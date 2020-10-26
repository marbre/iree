// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "iree/compiler/Dialect/HAL/Target/LLVM/AOT/LLVMAOTTargetLinker.h"

#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"

#define DEBUG_TYPE "llvmaot-linker"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {

//===----------------------------------------------------------------------===//
// Unix linker (ld-like); for ELF files
//===----------------------------------------------------------------------===//

class UnixLinkerTool : public LinkerTool {
 public:
  using LinkerTool::LinkerTool;

  std::string getToolPath() const override {
    auto toolPath = LinkerTool::getToolPath();
    return toolPath.empty() ? "ld.lld" : toolPath;
  }

  LogicalResult configureModule(llvm::Module *llvmModule,
                                ArrayRef<StringRef> entryPointNames) override {
    // Possibly a no-op in ELF files; needs to be verified.
    return success();
  }

  Optional<Artifacts> linkDynamicLibrary(
      ArrayRef<Artifact> objectFiles) override {
    Artifacts artifacts;
    artifacts.libraryFile = Artifact::createTemporary("llvmaot", "so");

    SmallVector<std::string, 8> flags = {
        getToolPath(),
        "-shared",
        "-o " + artifacts.libraryFile.path,
    };

    // TODO(ataei): add flags based on targetTriple.isAndroid(), like
    //   -static-libstdc++ (if this is needed, which it shouldn't be).

    // Link all input objects. Note that we are not linking whole-archive as we
    // want to allow dropping of unused codegen outputs.
    for (auto &objectFile : objectFiles) {
      flags.push_back(objectFile.path);
    }

    auto commandLine = llvm::join(flags, " ");
    if (failed(runLinkCommand(commandLine))) return llvm::None;
    return artifacts;
  }
};

//===----------------------------------------------------------------------===//
// Windows linker (MSVC link.exe-like); for DLL files
//===----------------------------------------------------------------------===//

class WindowsLinkerTool : public LinkerTool {
 public:
  using LinkerTool::LinkerTool;

  std::string getToolPath() const override {
    auto toolPath = LinkerTool::getToolPath();
    return toolPath.empty() ? "lld-link" : toolPath;
  }

  LogicalResult configureModule(llvm::Module *llvmModule,
                                ArrayRef<StringRef> entryPointNames) override {
    auto &ctx = llvmModule->getContext();

    // Create a _DllMainCRTStartup replacement that does not initialize the CRT.
    // This is required to prevent a bunch of CRT junk (locale, errno, TLS, etc)
    // from getting emitted in such a way that it cannot be stripped by LTCG.
    // Since we don't emit code using the CRT (beyond memset/memcpy) this is
    // fine and can reduce binary sizes by 50-100KB.
    //
    // More info:
    // https://docs.microsoft.com/en-us/cpp/build/run-time-library-behavior?view=vs-2019
    {
      auto dwordType = llvm::IntegerType::get(ctx, 32);
      auto ptrType = llvm::PointerType::getUnqual(dwordType);
      auto entry = cast<llvm::Function>(
          llvmModule
              ->getOrInsertFunction("IREEDLLMain", dwordType, ptrType,
                                    dwordType, ptrType)
              .getCallee());
      entry->setCallingConv(llvm::CallingConv::X86_StdCall);
      entry->setDLLStorageClass(
          llvm::GlobalValue::DLLStorageClassTypes::DLLExportStorageClass);
      entry->setLinkage(llvm::GlobalValue::LinkageTypes::ExternalLinkage);
      auto *block = llvm::BasicBlock::Create(ctx, "entry", entry);
      llvm::IRBuilder<> builder(block);
      auto one = llvm::ConstantInt::get(dwordType, 12345678, false);
      builder.CreateRet(one);
    }

    // For now we ensure that our entry points are exported (via linker
    // directives embedded in the object file) and in a compatible calling
    // convention.
    // TODO(benvanik): switch to executable libraries w/ internal functions.
    for (auto entryPointName : entryPointNames) {
      auto *entryPointFn = llvmModule->getFunction(entryPointName);
      entryPointFn->setCallingConv(llvm::CallingConv::X86_StdCall);
      entryPointFn->setDLLStorageClass(
          llvm::GlobalValue::DLLStorageClassTypes::DLLExportStorageClass);
      entryPointFn->setLinkage(
          llvm::GlobalValue::LinkageTypes::ExternalLinkage);
      entryPointFn->setVisibility(
          llvm::GlobalValue::VisibilityTypes::DefaultVisibility);
    }

    return success();
  }

  Optional<Artifacts> linkDynamicLibrary(
      ArrayRef<Artifact> objectFiles) override {
    Artifacts artifacts;
    artifacts.libraryFile = Artifact::createTemporary("llvmaot", "dll");
    artifacts.debugFile =
        Artifact::createVariant(artifacts.libraryFile.path, "pdb");

    // We currently discard the .lib file (only needed for dll import linking).
    artifacts.otherFiles.push_back(
        Artifact::createVariant(artifacts.libraryFile.path, "lib"));

    SmallVector<std::string, 8> flags = {
        getToolPath(),

        // Useful when debugging linking/loading issues:
        // "/verbose",

        // https://docs.microsoft.com/en-us/cpp/build/reference/dll-build-a-dll?view=vs-2019
        // Builds a DLL and exports functions with the dllexport storage class.
        "/dll",

        // Forces a fixed timestamp to ensure files are reproducable across
        // builds. Undocumented but accepted by both link and lld-link.
        // https://blog.conan.io/2019/09/02/Deterministic-builds-with-C-C++.html
        "/Brepro",

        // https://docs.microsoft.com/en-us/cpp/build/reference/nodefaultlib-ignore-libraries?view=vs-2019
        // Ignore any libraries that are specified by the platform as we
        // directly provide the ones we want.
        "/nodefaultlib",

        // https://docs.microsoft.com/en-us/cpp/build/reference/incremental-link-incrementally?view=vs-2019
        // Disable incremental linking as we are only ever linking in one-shot
        // mode to temp files. This avoids additional file padding and ordering
        // restrictions that enable incremental linking. Our other options will
        // prevent incremental linking in most cases, but it doesn't hurt to be
        // explicit.
        "/incremental:no",

        // https://docs.microsoft.com/en-us/cpp/build/reference/guard-enable-guard-checks?view=vs-2019
        // No control flow guard lookup (indirect branch verification).
        "/guard:no",

        // https://docs.microsoft.com/en-us/cpp/build/reference/safeseh-image-has-safe-exception-handlers?view=vs-2019
        // We don't want exception unwind tables in our output.
        "/safeseh:no",

        // https://docs.microsoft.com/en-us/cpp/build/reference/entry-entry-point-symbol?view=vs-2019
        // Use our entry point instead of the standard CRT one; ensures that we
        // pull in no global state from the CRT.
        "/entry:IREEDLLMain",

        // https://docs.microsoft.com/en-us/cpp/build/reference/debug-generate-debug-info?view=vs-2019
        // Copies all PDB information into the final PDB so that we can use the
        // same PDB across multiple machines.
        "/debug:full",

        // https://docs.microsoft.com/en-us/cpp/build/reference/pdbaltpath-use-alternate-pdb-path?view=vs-2019
        // Forces the PDB we generate to be referenced in the DLL as just a
        // relative path to the DLL itself. This allows us to move the PDBs
        // along with the build DLLs across machines.
        "/pdbaltpath:%_PDB%",

        // https://docs.microsoft.com/en-us/cpp/build/reference/out-output-file-name?view=vs-2019
        // Target for linker output. The base name of this path will be used for
        // additional output files (like the map and pdb).
        "/out:" + artifacts.libraryFile.path,
    };

    // DO NOT SUBMIT
    bool debug = false;

    if (!debug) {
      // https://docs.microsoft.com/en-us/cpp/build/reference/opt-optimizations?view=vs-2019
      // Enable all the fancy optimizations.
      flags.push_back("/opt:ref,icf,lbr");
    }

    flags.push_back(
        "/libpath:\"C:\\Program Files (x86)\\Microsoft Visual "
        "Studio\\2019\\Preview\\VC\\Tools\\MSVC\\14.28.29304\\lib\\x64\"");
    if (debug) {
      flags.push_back("vcruntimed.lib");
      flags.push_back("msvcrtd.lib");
    } else {
      flags.push_back("vcruntime.lib");
      flags.push_back("msvcrt.lib");
    }

    flags.push_back(
        "/libpath:\"C:\\Program Files (x86)\\Windows "
        "Kits\\10\\Lib\\10.0.18362.0\\ucrt\\x64\"");
    if (debug) {
      flags.push_back("ucrtd.lib");
    } else {
      flags.push_back("ucrt.lib");
    }

    flags.push_back(
        "/libpath:\"C:\\Program Files (x86)\\Windows "
        "Kits\\10\\Lib\\10.0.18362.0\\um\\x64\"");
    flags.push_back("kernel32.lib");

    // Link all input objects. Note that we are not linking whole-archive as we
    // want to allow dropping of unused codegen outputs.
    for (auto &objectFile : objectFiles) {
      flags.push_back(objectFile.path);
    }

    auto commandLine = llvm::join(flags, " ");
    if (failed(runLinkCommand(commandLine))) return llvm::None;
    return artifacts;
  }
};

// TODO(benvanik): add other platforms:
class MacLinkerTool;   // ld64.lld
class WasmLinkerTool;  // wasm-ld

//===----------------------------------------------------------------------===//
// Linker tool discovery
//===----------------------------------------------------------------------===//

// static
std::unique_ptr<LinkerTool> LinkerTool::getForTarget(
    llvm::Triple &targetTriple, LLVMTargetOptions &targetOptions) {
  if (targetTriple.isOSWindows() || targetTriple.isWindowsMSVCEnvironment()) {
    return std::make_unique<WindowsLinkerTool>(targetTriple, targetOptions);
  }
  return std::make_unique<UnixLinkerTool>(targetTriple, targetOptions);
}

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
