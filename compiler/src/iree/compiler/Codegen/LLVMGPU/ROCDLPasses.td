// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_CODEGEN_LLVMGPU_ROCDLPASSES
#define IREE_CODEGEN_LLVMGPU_ROCDLPASSES

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// ROCDL Passes (keep alphabetical)
//===----------------------------------------------------------------------===//

def ROCDLLowerExecutableTargetPass : InterfacePass<
    "iree-rocdl-lower-executable-target", "mlir::FunctionOpInterface"> {
  let summary = "Lower an IREE hal.executable.variant op using a suitable "
                "pass pipeline";
}

def ROCDLSelectLoweringStrategyPass :
    Pass<"iree-rocdl-select-lowering-strategy", "ModuleOp"> {
  let summary = "Select a suitable lowering strategy for an IREE "
                "hal.executable.variant op";
}


#endif // IREE_CODEGEN_LLVMGPU_ROCDLPASSES
