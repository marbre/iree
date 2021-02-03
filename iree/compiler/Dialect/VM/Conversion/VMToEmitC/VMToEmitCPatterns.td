// Copyright 2021 Google LLC
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

include "mlir/IR/OpBase.td"
include "iree/compiler/Dialect/VM/IR/VMOps.td"
include "emitc/Dialect/EmitC/EmitC.td"

class StringAttrOfValue<string value>:
    NativeCodeCall<"$_builder.getStringAttr(\"" # value # "\")">;

def NullArrayAttr :
	NativeCodeCall<"ArrayAttr()">;

def ArrayRef :
    //NativeCodeCall<"$_builder.getArrayAttr({$0, $1})">;
    NativeCodeCall<"ArrayRef<int32_t> ($0, $1)">;
    //NativeCodeCall<"$_builder, lhs, rhs">;


class NoAttributeOpConversion<VM_BinaryArithmeticOp op, string name>:
    Pat<(op $lhs, $rhs),
          (EmitC_CallOp (StringAttrOfValue<name>), (NullArrayAttr), (NullArrayAttr), (ArrayRef $lhs, $rhs))>;

def : NoAttributeOpConversion<VM_AddI32Op, "vm_add_i32">;
def : NoAttributeOpConversion<VM_SubI32Op, "vm_sub_i32">;
def : NoAttributeOpConversion<VM_MulI32Op, "vm_mul_i32">;
def : NoAttributeOpConversion<VM_DivI32SOp, "vm_div_i32s">;
def : NoAttributeOpConversion<VM_DivI32UOp, "vm_div_i32u">;
def : NoAttributeOpConversion<VM_RemI32SOp, "vm_rem_i32s">;
def : NoAttributeOpConversion<VM_RemI32UOp, "vm_rem_i32u">;
//def : NoAttributeOpConversion<VM_NotI32Op, "vm_not_i32">;
def : NoAttributeOpConversion<VM_AndI32Op, "vm_and_i32">;
def : NoAttributeOpConversion<VM_OrI32Op, "vm_or_i32">;
def : NoAttributeOpConversion<VM_XorI32Op, "vm_xor_i32">;

