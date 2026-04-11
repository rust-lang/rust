/*
 * Copyright (c) 2026 Teenygrad.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

use melior::Context;
use melior::dialect::arith;
use melior::dialect::ods::arith::{ConstantOperation, MaxSIOperation, MaximumFOperation};
use melior::ir::attribute::DenseI32ArrayAttribute;
use melior::ir::operation::OperationMutLike;
use melior::ir::r#type::{IntegerType, RankedTensorType};
use melior::ir::{Attribute, Location, Operation, Type, TypeLike, Value, ValueLike};

use crate::errors::Error;
use crate::ffi::mlirCreateTritonPointerType;
use crate::shared::builtin::tensor_type;
use crate::triton::attr_i32;
use crate::triton::tt::{
    AddPtrOperation, LoadOperation, MakeRangeOperation, MulhiUIOperation, SplatOperation,
    StoreOperation,
};

pub fn make_range<'ctx>(
    context: &'ctx Context,
    location: Location<'ctx>,
    start: i32,
    end: i32,
) -> Result<MakeRangeOperation<'ctx>, Error> {
    let start_attr = attr_i32(context, start);
    let end_attr = attr_i32(context, end);
    let element_type = IntegerType::new(context, 32).into();
    let dimensions = &[(end - start) as i64];

    let result = tensor_type(dimensions, element_type).into();
    Ok(MakeRangeOperation::builder(context, location)
        .start(start_attr)
        .end(end_attr)
        .result(result)
        .build())
}

pub fn splat<'ctx>(
    context: &'ctx Context,
    location: Location<'ctx>,
    src: Value<'ctx, 'ctx>,
    result_ty: Type<'ctx>,
) -> Result<SplatOperation<'ctx>, Error> {
    Ok(SplatOperation::builder(context, location).src(src).result(result_ty).build())
}

pub fn add_ptr<'ctx>(
    context: &'ctx Context,
    location: Location<'ctx>,
    ptr: Value<'ctx, 'ctx>,
    offset: Value<'ctx, 'ctx>,
    result_ty: Type<'ctx>,
) -> Result<AddPtrOperation<'ctx>, Error> {
    Ok(AddPtrOperation::builder(context, location)
        .ptr(ptr)
        .offset(offset)
        .result(result_ty)
        .build())
}

pub fn load<'ctx>(
    context: &'ctx Context,
    location: Location<'ctx>,
    ptr: Value<'ctx, 'ctx>,
    mask: Value<'ctx, 'ctx>,
    // result_ty: Type<'ctx>,
) -> Result<LoadOperation<'ctx>, Error> {
    let mut op: Operation<'ctx> =
        LoadOperation::builder(context, location).ptr(ptr).mask(mask).build().into();

    // ptr=1, mask=1, other=0
    let seg_sizes = DenseI32ArrayAttribute::new(context, &[1, 1, 0]);
    op.set_attribute("operandSegmentSizes", Attribute::from(seg_sizes));
    Ok(LoadOperation::try_from(op).expect("valid tt.load"))
}

pub fn store<'ctx>(
    context: &'ctx Context,
    location: Location<'ctx>,
    ptr: Value<'ctx, 'ctx>,
    value: Value<'ctx, 'ctx>,
    mask: Value<'ctx, 'ctx>,
) -> Result<StoreOperation<'ctx>, Error> {
    Ok(StoreOperation::builder(context, location).ptr(ptr).value(value).mask(mask).build())
}

pub fn mulhiui<'ctx>(
    context: &'ctx Context,
    location: Location<'ctx>,
    x: Value<'ctx, 'ctx>,
    y: Value<'ctx, 'ctx>,
) -> Result<MulhiUIOperation<'ctx>, Error> {
    // SameOperandsAndResultType: result type is inferred from operand types
    Ok(MulhiUIOperation::builder(context, location).x(x).y(y).build())
}

pub fn maximumf<'ctx>(
    context: &'ctx Context,
    location: Location<'ctx>,
    a: Value<'ctx, 'ctx>,
    b: Value<'ctx, 'ctx>,
) -> Result<MaximumFOperation<'ctx>, Error> {
    todo!()
    // Ok(MaximumOperation::builder(context, location).a(a).b(b).build())
}

pub fn maxsi<'ctx>(
    context: &'ctx Context,
    location: Location<'ctx>,
    a: Value<'ctx, 'ctx>,
    b: Value<'ctx, 'ctx>,
) -> Result<MaxSIOperation<'ctx>, Error> {
    todo!()
    // Ok(MaximumOperation::builder(context, location).a(a).b(b).build())
}

pub fn zeros_like<'ctx>(
    context: &'ctx Context,
    location: Location<'ctx>,
    src: Value<'ctx, 'ctx>,
) -> Result<ConstantOperation<'ctx>, Error> {
    let ty: RankedTensorType<'ctx> = src
        .r#type()
        .try_into()
        .map_err(|_| Error::InvalidType { msg: "Invalid tensor type".to_string() })?;
    todo!()
}

#[cfg(test)]
mod tests {
    use melior::Context;
    use melior::ir::operation::OperationLike;
    use melior::ir::{BlockLike, Location, Module, Operation};

    use super::*;
    use crate::shared::arith::{Int, create_int_constant};
    use crate::test::create_test_context;
    use crate::triton::{int_to_ptr, load_triton_dialect, pointer_type};

    #[test]
    fn test_make_range_generic() {
        let context = Context::new();
        let location = Location::unknown(&context);

        let op = make_range(&context, location, 0, 5).unwrap();

        let output = op.as_operation().to_string();
        let expected =
            "%0 = \"tt.make_range\"() {end = 5 : i32, start = 0 : i32} : () -> tensor<5xi32>\n";
        assert_eq!(expected, output);
    }

    #[test]
    fn test_make_range() {
        let context = create_test_context();
        load_triton_dialect(&context);

        let location = Location::unknown(&context);
        let module = Module::new(location);

        let range_op = make_range(&context, location, 0, 8).unwrap();
        module.body().append_operation(range_op.into());

        let output = module.as_operation().to_string();
        let expected = "module {\n  %0 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>\n}\n";
        assert_eq!(expected, output);
    }

    #[test]
    fn test_create_splat() {
        let context = create_test_context();
        load_triton_dialect(&context);

        let location = Location::unknown(&context);
        let module = Module::new(location);

        let src_op: Operation<'_> =
            create_int_constant(&context, location, Int::I32(0)).unwrap().into();
        let src = src_op.result(0).unwrap().into();
        let result = tensor_type(&[5], IntegerType::new(&context, 32).into()).into();
        let splat_op = splat(&context, location, src, result);
        assert!(splat_op.is_ok());
        let splat_op = splat_op.unwrap().into();

        module.body().append_operation(src_op);
        module.body().append_operation(splat_op);

        let output = module.as_operation().to_string();
        let expected = "\"builtin.module\"() ({\n  %0 = \"arith.constant\"() <{value = 0 : i32}> : () -> i32\n  %1 = \"tt.splat\"(%0) : (i32) -> tensor<5xi32>\n}) : () -> ()\n";
        assert_eq!(expected, output);
    }

    #[test]
    fn test_mulhiui() {
        let context = create_test_context();
        load_triton_dialect(&context);

        let location = Location::unknown(&context);
        let module = Module::new(location);

        // Create two i32 constants as operands
        let x_op: Operation<'_> =
            create_int_constant(&context, location, Int::I32(3)).unwrap().into();
        let x_val = x_op.result(0).unwrap().into();

        let y_op: Operation<'_> =
            create_int_constant(&context, location, Int::I32(5)).unwrap().into();
        let y_val = y_op.result(0).unwrap().into();

        let mulhi_op = mulhiui(&context, location, x_val, y_val).unwrap();

        module.body().append_operation(x_op);
        module.body().append_operation(y_op);
        module.body().append_operation(mulhi_op.into());

        let output = module.as_operation().to_string();
        let expected = "module {\n  %c3_i32 = arith.constant 3 : i32\n  %c5_i32 = arith.constant 5 : i32\n  %0 = tt.mulhiui %c3_i32, %c5_i32 : i32\n}\n";
        assert_eq!(expected, output);
    }

    #[test]
    fn test_add_ptr() {
        let context = create_test_context();
        load_triton_dialect(&context);

        let location = Location::unknown(&context);
        let module = Module::new(location);

        // Create an i64 zero constant to cast to a pointer
        let i64_zero_op: Operation<'_> =
            create_int_constant(&context, location, Int::I64(0)).unwrap().into();
        let i64_zero = i64_zero_op.result(0).unwrap().into();

        // Cast i64 → !tt.ptr<f32>
        let f32_type = melior::ir::Type::float32(&context);
        let ptr_f32_type = pointer_type(f32_type);
        let int_to_ptr_op: Operation<'_> =
            int_to_ptr(&context, location, i64_zero, ptr_f32_type).unwrap().into();
        let ptr_val = int_to_ptr_op.result(0).unwrap().into();

        // i32 offset constant
        let offset_op: Operation<'_> =
            create_int_constant(&context, location, Int::I32(0)).unwrap().into();
        let offset_val = offset_op.result(0).unwrap().into();

        // tt.addptr: result type is the same pointer type
        let add_ptr_op = add_ptr(&context, location, ptr_val, offset_val, ptr_f32_type).unwrap();

        module.body().append_operation(i64_zero_op);
        module.body().append_operation(int_to_ptr_op);
        module.body().append_operation(offset_op);
        module.body().append_operation(add_ptr_op.into());

        let output = module.as_operation().to_string();
        let expected = "module {\n  %c0_i64 = arith.constant 0 : i64\n  %0 = tt.int_to_ptr %c0_i64 : i64 -> !tt.ptr<f32>\n  %c0_i32 = arith.constant 0 : i32\n  %1 = tt.addptr %0, %c0_i32 : !tt.ptr<f32>, i32\n}\n";
        assert_eq!(expected, output);
    }
}
