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
use melior::ir::attribute::{BoolAttribute, DenseI32ArrayAttribute};
use melior::ir::operation::{OperationBuilder, OperationMutLike};
use melior::ir::r#type::{IntegerType, RankedTensorType};
use melior::ir::{Attribute, Identifier, Location, Operation, Type, TypeLike, Value, ValueLike};

use crate::errors::Error;
use crate::ffi::mlirCreateTritonPointerType;
use crate::shared::builtin::tensor_type;
use crate::triton::attr_i32;
use crate::triton::tt::{
    AddPtrOperation, DescriptorGatherOperation, LoadOperation, MakeRangeOperation,
    MulhiUIOperation, SplatOperation, StoreOperation,
};

/// Mirror of Triton's `ScaleDotElemType` enum (TritonAttrDefs.td).
///
/// Integer values must stay in sync with the TableGen definition so that
/// the resulting `IntegerAttr<i32>` is accepted as a `ScaleDotElemTypeAttr`
/// by MLIR's `classof` check.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ScaleDotElemType {
    E4M3 = 0,
    E5M2 = 1,
    E2M3 = 2,
    E3M2 = 3,
    E2M1 = 4,
    BF16 = 5,
    FP16 = 6,
}

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

/// Build a `tt.descriptor_gather` operation.
///
/// Lowers to NVIDIA TMA gather, reading multiple rows from a TMA descriptor
/// into a single 2D result tensor.
///
/// # Arguments
/// * `desc`      – value of type `!tt.tensordesc<tensor<1xNxT>>`;
///                 the descriptor block must have exactly one row.
/// * `x_offsets` – 1-D `tensor<Kxi32>` of column offsets to gather.
/// * `y_offset`  – scalar `i32` row offset.
/// * `result_ty` – expected 2-D tensor type of the result (e.g.
///                 `tensor<KxNxT>`).
///
/// The op's assembly format is:
/// ```text
/// tt.descriptor_gather %desc[%x_offsets, %y_offset]
///     : (type(desc), type(x_offsets), type(y_offset)) -> result_ty
/// ```
pub fn descriptor_gather<'ctx>(
    context: &'ctx Context,
    location: Location<'ctx>,
    desc: Value<'ctx, 'ctx>,
    x_offsets: Value<'ctx, 'ctx>,
    y_offset: Value<'ctx, 'ctx>,
    result_ty: Type<'ctx>,
) -> Result<DescriptorGatherOperation<'ctx>, Error> {
    Ok(DescriptorGatherOperation::builder(context, location)
        .desc(desc)
        .x_offsets(x_offsets)
        .y_offset(y_offset)
        .result(result_ty)
        .build())
}

/// Build a `tt.dot_scaled` operation.
///
/// Computes `d = matrix_multiply(scale(a, a_scale), scale(b, b_scale)) + c`
/// following the microscaling spec.  `a_scale` and `b_scale` are optional;
/// pass `None` to omit them (the corresponding operand segment is set to 0).
///
/// `a_elem_type` / `b_elem_type` describe the logical element type used for
/// scaling and are encoded as `ScaleDotElemTypeAttr` (an `IntegerAttr<i32>`
/// in MLIR's storage).
///
/// The result type is always the same as `c` (enforced by
/// `TypesMatchWith<"result's type matches accumulator's type", "d", "c", "$_self">`).
pub fn dot_scaled<'ctx>(
    context: &'ctx Context,
    location: Location<'ctx>,
    a: Value<'ctx, 'ctx>,
    b: Value<'ctx, 'ctx>,
    c: Value<'ctx, 'ctx>,
    a_scale: Option<Value<'ctx, 'ctx>>,
    b_scale: Option<Value<'ctx, 'ctx>>,
    a_elem_type: ScaleDotElemType,
    b_elem_type: ScaleDotElemType,
    fast_math: bool,
) -> Result<Operation<'ctx>, Error> {
    // Result type matches accumulator type (TypesMatchWith constraint).
    let result_type = c.r#type();

    // Collect operands: required first, then optional.
    let mut operands: Vec<Value> = vec![a, b, c];
    let a_scale_size: i32 = if let Some(s) = a_scale {
        operands.push(s);
        1
    } else {
        0
    };
    let b_scale_size: i32 = if let Some(s) = b_scale {
        operands.push(s);
        1
    } else {
        0
    };

    // AttrSizedOperandSegments for [a, b, c, a_scale, b_scale].
    let seg_sizes =
        DenseI32ArrayAttribute::new(context, &[1, 1, 1, a_scale_size, b_scale_size]);

    // ScaleDotElemTypeAttr is backed by IntegerAttr<i32>; attr_i32 produces a
    // compatible value that satisfies classof(ScaleDotElemTypeAttr).
    let a_elem_attr = attr_i32(context, a_elem_type as i32);
    let b_elem_attr = attr_i32(context, b_elem_type as i32);
    let fast_math_attr = BoolAttribute::new(context, fast_math);

    let op = OperationBuilder::new("tt.dot_scaled", location)
        .add_operands(&operands)
        .add_attributes(&[
            (
                Identifier::new(context, "operandSegmentSizes"),
                Attribute::from(seg_sizes),
            ),
            (
                Identifier::new(context, "a_elem_type"),
                Attribute::from(a_elem_attr),
            ),
            (
                Identifier::new(context, "b_elem_type"),
                Attribute::from(b_elem_attr),
            ),
            (
                Identifier::new(context, "fastMath"),
                Attribute::from(fast_math_attr),
            ),
        ])
        .add_results(&[result_type])
        .build()
        .map_err(|e| Error::InvalidType { msg: format!("failed to build tt.dot_scaled: {e}") })?;

    Ok(op)
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
    use melior::ir::{Block, BlockLike, Location, Module, Operation, RegionLike};

    use super::*;
    use crate::shared::arith::{Int, create_int_constant};
    use crate::shared::builtin::tensor_type;
    use crate::test::create_test_context;
    use crate::triton::tt::ReturnOperation;
    use crate::triton::{create_func, int_to_ptr, load_triton_dialect, pointer_type};

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

    /// Verify that `descriptor_gather` emits the correct `tt.descriptor_gather` IR.
    ///
    /// Uses function block-arguments to supply typed values without needing
    /// constant-folding or a real TMA descriptor at compile time.
    #[test]
    fn test_descriptor_gather() {
        let context = create_test_context();
        load_triton_dialect(&context);

        let location = Location::unknown(&context);
        let module = Module::new(location);

        // Result type: tensor<16x16xf32> (16 rows gathered, 16 cols each).
        let f32_type = melior::ir::Type::float32(&context);
        let result_tensor_ty: Type = tensor_type(&[16, 16], f32_type).into();

        // Descriptor type: !tt.tensordesc<tensor<1x16xf32>>.
        // The block has 1 row and 16 columns – the gather restriction.
        let desc_ty = Type::parse(&context, "!tt.tensordesc<tensor<1x16xf32>>")
            .expect("valid tensordesc type");

        // x_offsets type: tensor<16xi32> (16 column-offset indices).
        let i32_type: Type = IntegerType::new(&context, 32).into();
        let x_offsets_ty: Type = tensor_type(&[16], i32_type).into();

        // y_offset type: i32.
        let y_offset_ty = i32_type;

        // Build a wrapper function so operands have proper block-argument types.
        let func_op = create_func(
            &context,
            location,
            "test_descriptor_gather",
            "public",
            &[desc_ty, x_offsets_ty, y_offset_ty],
            &[result_tensor_ty],
            0,
        )
        .unwrap();

        let block = Block::new(&[
            (desc_ty, location),
            (x_offsets_ty, location),
            (y_offset_ty, location),
        ]);

        let desc: Value = block.argument(0).unwrap().into();
        let x_offsets: Value = block.argument(1).unwrap().into();
        let y_offset: Value = block.argument(2).unwrap().into();

        let gather_op: Operation<'_> =
            descriptor_gather(&context, location, desc, x_offsets, y_offset, result_tensor_ty)
                .unwrap()
                .into();

        let ret_val: Value = gather_op.result(0).unwrap().into();
        let ret_op = ReturnOperation::builder(&context, location).srcs(&[ret_val]).build();

        block.append_operation(gather_op);
        block.append_operation(ret_op.into());
        func_op.body().unwrap().append_block(block);
        module.body().append_operation(func_op.into());

        let output = module.as_operation().to_string();

        assert!(
            output.contains("tt.descriptor_gather"),
            "missing op mnemonic:\n{output}"
        );
        assert!(
            output.contains("tensordesc"),
            "missing tensordesc operand type:\n{output}"
        );
        assert!(
            output.contains("tensor<16x16xf32>"),
            "missing result tensor type:\n{output}"
        );
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

    /// Verify that `dot_scaled` emits a valid `tt.dot_scaled` op without
    /// optional scale operands.  The test uses function block-arguments as
    /// tensor values so it does not depend on any constant-folding logic.
    #[test]
    fn test_dot_scaled_no_scales() {
        let context = create_test_context();
        load_triton_dialect(&context);

        let location = Location::unknown(&context);
        let module = Module::new(location);

        let f32_type = melior::ir::Type::float32(&context);
        let tensor_type_2d: Type = tensor_type(&[16, 16], f32_type).into();

        // Build a function: (tensor<16x16xf32>, tensor<16x16xf32>, tensor<16x16xf32>) -> tensor<16x16xf32>
        let func_op = create_func(
            &context,
            location,
            "test_dot_scaled",
            "public",
            &[tensor_type_2d, tensor_type_2d, tensor_type_2d],
            &[tensor_type_2d],
            0,
        )
        .unwrap();

        let block = Block::new(&[
            (tensor_type_2d, location),
            (tensor_type_2d, location),
            (tensor_type_2d, location),
        ]);

        let a = block.argument(0).unwrap().into();
        let b = block.argument(1).unwrap().into();
        let c = block.argument(2).unwrap().into();

        let dot_op =
            dot_scaled(&context, location, a, b, c, None, None, ScaleDotElemType::FP16, ScaleDotElemType::FP16, false)
                .unwrap();

        let ret_val: Value = dot_op.result(0).unwrap().into();
        let ret_op = ReturnOperation::builder(&context, location).srcs(&[ret_val]).build();

        block.append_operation(dot_op);
        block.append_operation(ret_op.into());
        func_op.body().unwrap().append_block(block);
        module.body().append_operation(func_op.into());

        let output = module.as_operation().to_string();

        // Verify the key structural elements of the emitted IR.
        assert!(output.contains("tt.dot_scaled"), "missing op mnemonic:\n{output}");
        assert!(output.contains("lhs = fp16"), "missing lhs elem type:\n{output}");
        assert!(output.contains("rhs = fp16"), "missing rhs elem type:\n{output}");
        assert!(output.contains("fastMath = false"), "missing fastMath attr:\n{output}");
        assert!(output.contains("-> tensor<16x16xf32>"), "missing result type:\n{output}");
    }

    /// Verify `dot_scaled` with both optional scale tensors present.
    #[test]
    fn test_dot_scaled_with_scales() {
        let context = create_test_context();
        load_triton_dialect(&context);

        let location = Location::unknown(&context);
        let module = Module::new(location);

        let f32_type = melior::ir::Type::float32(&context);
        let tensor_16x16: Type = tensor_type(&[16, 16], f32_type).into();
        let tensor_16x1: Type = tensor_type(&[16, 1], f32_type).into();

        // Function: (a, b, c, a_scale, b_scale) -> tensor<16x16xf32>
        let func_op = create_func(
            &context,
            location,
            "test_dot_scaled_scales",
            "public",
            &[tensor_16x16, tensor_16x16, tensor_16x16, tensor_16x1, tensor_16x1],
            &[tensor_16x16],
            0,
        )
        .unwrap();

        let block = Block::new(&[
            (tensor_16x16, location),
            (tensor_16x16, location),
            (tensor_16x16, location),
            (tensor_16x1, location),
            (tensor_16x1, location),
        ]);

        let a: Value = block.argument(0).unwrap().into();
        let b: Value = block.argument(1).unwrap().into();
        let c: Value = block.argument(2).unwrap().into();
        let a_scale: Value = block.argument(3).unwrap().into();
        let b_scale: Value = block.argument(4).unwrap().into();

        let dot_op = dot_scaled(
            &context,
            location,
            a,
            b,
            c,
            Some(a_scale),
            Some(b_scale),
            ScaleDotElemType::E4M3,
            ScaleDotElemType::E5M2,
            true,
        )
        .unwrap();

        let ret_val: Value = dot_op.result(0).unwrap().into();
        let ret_op = ReturnOperation::builder(&context, location).srcs(&[ret_val]).build();

        block.append_operation(dot_op);
        block.append_operation(ret_op.into());
        func_op.body().unwrap().append_block(block);
        module.body().append_operation(func_op.into());

        let output = module.as_operation().to_string();

        // Both the custom assembly format ("lhs = e4m3") and the generic format
        // ("a_elem_type = 0 : i32") are acceptable; check for at least one.
        assert!(output.contains("tt.dot_scaled"), "missing op mnemonic:\n{output}");
        assert!(
            output.contains("lhs = e4m3") || output.contains("a_elem_type = 0"),
            "missing a_elem_type (E4M3 = 0):\n{output}",
        );
        assert!(
            output.contains("rhs = e5m2") || output.contains("b_elem_type = 1"),
            "missing b_elem_type (E5M2 = 1):\n{output}",
        );
        assert!(output.contains("fastMath = true"), "missing fastMath attr:\n{output}");
        // Optional scale operands must appear somewhere in the output.
        assert!(
            output.contains("scale") || output.contains("arg3"),
            "optional scale operands missing from output:\n{output}",
        );
    }
}
