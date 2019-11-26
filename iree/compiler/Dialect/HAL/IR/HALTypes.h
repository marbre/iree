// Copyright 2019 Google LLC
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

#ifndef IREE_COMPILER_DIALECT_HAL_IR_HALTYPES_H_
#define IREE_COMPILER_DIALECT_HAL_IR_HALTYPES_H_

#include <cstdint>

#include "iree/compiler/Dialect/Types.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSwitch.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"

// Order matters.
#include "iree/compiler/Dialect/HAL/IR/HALEnums.h.inc"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {

//===----------------------------------------------------------------------===//
// RefObject types
//===----------------------------------------------------------------------===//

class AllocatorType : public Type::TypeBase<AllocatorType, RefObjectType> {
 public:
  using Base::Base;
  static AllocatorType get(MLIRContext *context) {
    return Base::get(context, TypeKind::Allocator);
  }
  static bool kindof(unsigned kind) { return kind == TypeKind::Allocator; }
};

class BufferType : public Type::TypeBase<BufferType, RefObjectType> {
 public:
  using Base::Base;
  static BufferType get(MLIRContext *context) {
    return Base::get(context, TypeKind::Buffer);
  }
  static bool kindof(unsigned kind) { return kind == TypeKind::Buffer; }
};

class CommandBufferType
    : public Type::TypeBase<CommandBufferType, RefObjectType> {
 public:
  using Base::Base;
  static CommandBufferType get(MLIRContext *context) {
    return Base::get(context, TypeKind::CommandBuffer);
  }
  static bool kindof(unsigned kind) { return kind == TypeKind::CommandBuffer; }
};

class DeviceType : public Type::TypeBase<DeviceType, RefObjectType> {
 public:
  using Base::Base;
  static DeviceType get(MLIRContext *context) {
    return Base::get(context, TypeKind::Device);
  }
  static bool kindof(unsigned kind) { return kind == TypeKind::Device; }
};

class EventType : public Type::TypeBase<EventType, RefObjectType> {
 public:
  using Base::Base;
  static EventType get(MLIRContext *context) {
    return Base::get(context, TypeKind::Event);
  }
  static bool kindof(unsigned kind) { return kind == TypeKind::Event; }
};

class ExecutableType : public Type::TypeBase<ExecutableType, RefObjectType> {
 public:
  using Base::Base;
  static ExecutableType get(MLIRContext *context) {
    return Base::get(context, TypeKind::Executable);
  }
  static bool kindof(unsigned kind) { return kind == TypeKind::Executable; }
};

class ExecutableCacheType
    : public Type::TypeBase<ExecutableCacheType, RefObjectType> {
 public:
  using Base::Base;
  static ExecutableCacheType get(MLIRContext *context) {
    return Base::get(context, TypeKind::ExecutableCache);
  }
  static bool kindof(unsigned kind) {
    return kind == TypeKind::ExecutableCache;
  }
};

class FenceType : public Type::TypeBase<FenceType, RefObjectType> {
 public:
  using Base::Base;
  static FenceType get(MLIRContext *context) {
    return Base::get(context, TypeKind::Fence);
  }
  static bool kindof(unsigned kind) { return kind == TypeKind::Fence; }
};

class RingBufferType : public Type::TypeBase<RingBufferType, RefObjectType> {
 public:
  using Base::Base;
  static RingBufferType get(MLIRContext *context) {
    return Base::get(context, TypeKind::RingBuffer);
  }
  static bool kindof(unsigned kind) { return kind == TypeKind::RingBuffer; }
};

class SemaphoreType : public Type::TypeBase<SemaphoreType, RefObjectType> {
 public:
  using Base::Base;
  static SemaphoreType get(MLIRContext *context) {
    return Base::get(context, TypeKind::Semaphore);
  }
  static bool kindof(unsigned kind) { return kind == TypeKind::Semaphore; }
};

//===----------------------------------------------------------------------===//
// Struct types
//===----------------------------------------------------------------------===//

class BufferBarrierType {
 public:
  static TupleType get(MLIRContext *context) {
    return TupleType::get(
        {
            IntegerType::get(32, context),
            IntegerType::get(32, context),
            RefPtrType::get(BufferType::get(context)),
            IntegerType::get(32, context),
            IntegerType::get(32, context),
        },
        context);
  }
};

class BufferBarrierListType {
 public:
  static TupleType get(size_t count, MLIRContext *context) {
    SmallVector<Type, 4> elementTypes(count, BufferBarrierType::get(context));
    return TupleType::get(elementTypes, context);
  }
};

class MemoryBarrierType {
 public:
  static TupleType get(MLIRContext *context) {
    return TupleType::get(
        {
            IntegerType::get(32, context),
            IntegerType::get(32, context),
        },
        context);
  }
};

class MemoryBarrierListType {
 public:
  static TupleType get(size_t count, MLIRContext *context) {
    SmallVector<Type, 4> elementTypes(count, MemoryBarrierType::get(context));
    return TupleType::get(elementTypes, context);
  }
};

class BufferBindingType {
 public:
  static TupleType get(MLIRContext *context) {
    return TupleType::get(
        {
            IntegerType::get(32, context),
            RefPtrType::get(BufferType::get(context)),
            IntegerType::get(32, context),
            IntegerType::get(32, context),
        },
        context);
  }
};

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_HAL_IR_HALTYPES_H_
