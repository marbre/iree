// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECTS_DIALECT_IREEPYDM_UTILS_TYPE_INFERENCE_H
#define IREE_DIALECTS_DIALECT_IREEPYDM_UTILS_TYPE_INFERENCE_H

#include "mlir/IR/Block.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Allocator.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace PYDM {

/// Holds state and manages updates for performing permuted type propagation.
/// This is used by various local and global type inference passes. A key
/// feature of algorithms implemented with this class relates to permutation:
/// generally, duplicating/permuting blocks or regions is preferred over
/// unifying.
class PermutedTypePropagator {
public:
  PermutedTypePropagator(MLIRContext *context) : context(context) {}

  // ---------------------------------------------------------------------------
  // Block permutations.
  // Every block being operated on is either a parent block (pre-existing) or
  // permuted (generated by applying some transformation to the parent and
  // duplicating it).
  // ---------------------------------------------------------------------------
  using BlockPermuteCallback = std::function<void(
      Block *newBlock, Block *origBlock, BlockAndValueMapping &mapping)>;
  struct ParentBlockInfo;
  struct PermutedParentBlockInfo;

  struct PermutedBlockInfo {
    Block *permutedBlock;
    ParentBlockInfo *parentInfo;
    FunctionType signature;
    PermutedBlockInfo *next = nullptr;
  };

  struct ParentBlockInfo {
    Block *parentBlock = nullptr;
    PermutedBlockInfo *permutationHead = nullptr;
    int size = 0;
  };

  struct BlockPredecessor {
    BranchOpInterface terminator;
    unsigned successorIndex;
    FunctionType signature;
  };

  /// Finds any predecessor blocks which are mismatched with a predecessor
  /// signature.
  SmallVector<BlockPredecessor> findMismatchedBlockPredecessors(Block *block);

  /// For an arbitrary Block, looks up the parent block info record. If no
  /// such record exists, this is assumed to be a parent block and a record
  /// is established and returned.
  ParentBlockInfo *lookupParentBlock(Block *forBlock);

  /// Finds an existing block permutation which matches the argument types.
  /// Returns nullptr if none exists.
  Block *findBlockPermutation(ParentBlockInfo *parentInfo,
                              FunctionType signature);

  /// Creates a new block permutation. The initialize callback must populate
  /// the mapping for all original arguments.
  Block *createBlockPermutation(Location loc, ParentBlockInfo *parentInfo,
                                TypeRange newArgumentTypes,
                                BlockPermuteCallback initializeCallback);

private:
  MLIRContext *context;
  llvm::BumpPtrAllocator allocator;

  DenseMap<Block *, PermutedBlockInfo *> permutedBlocks;

  PermutedBlockInfo *addPermutedBlockToParent(ParentBlockInfo *parentInfo,
                                              Block *block);
};

} // namespace PYDM
} // namespace IREE
} // namespace iree_compiler
} // namespace mlir

#endif // IREE_DIALECTS_DIALECT_IREEPYDM_UTILS_TYPE_INFERENCE_H
