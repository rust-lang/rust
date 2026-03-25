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
use melior::ir::attribute::DenseI32ArrayAttribute;
use melior::ir::operation::OperationMutLike;
use melior::ir::r#type::IntegerType;
use melior::ir::{Attribute, Location, Operation, Type, TypeLike, Value};

use crate::errors::Error;
use crate::ffi::mlirCreateTritonPointerType;
use crate::shared::builtin::tensor_type;
use crate::triton::attr_i32;
use crate::triton::tt::{
    AddPtrOperation, LoadOperation, MakeRangeOperation, SplatOperation, StoreOperation,
};

pub fn arange<'ctx>(
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
    fn test_create_arange() {
        let context = Context::new();
        let location = Location::unknown(&context);
        let start = 0;
        let end = 5;

        let arange_op = arange(&context, location, start, end);
        assert!(arange_op.is_ok());
        let op = arange_op.unwrap();

        let output = op.as_operation().to_string();
        let expected =
            "%0 = \"tt.make_range\"() {end = 5 : i32, start = 0 : i32} : () -> tensor<5xi32>\n";
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
}
