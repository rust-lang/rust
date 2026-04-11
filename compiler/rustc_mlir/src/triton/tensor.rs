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
    MulhiUIOperation, SplatOperation,
};

/// Mirror of Triton's `InputPrecision` enum (TritonAttrDefs.td).
///
/// Controls how tensor cores are used when inputs are f32.
/// Integer values must stay in sync with the TableGen definition so that
/// the resulting `IntegerAttr<i32>` is accepted as `InputPrecisionAttr`
/// by MLIR's `classof` check.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum InputPrecision {
    TF32   = 0,
    TF32x3 = 1,
    IEEE   = 2,
    BF16x3 = 3,
    BF16x6 = 4,
}

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

/// Mirror of Triton's `CacheModifier` enum (TritonAttrDefs.td).
///
/// Controls L1/L2 cache behaviour for load/store ops.
/// Integer values must stay in sync with the TableGen definition so that
/// the resulting `IntegerAttr<i32>` is accepted as a `CacheModifierAttr`.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum CacheModifier {
    None = 1,
    Ca   = 2,
    Cg   = 3,
    Wb   = 4,
    Cs   = 5,
    Wt   = 6,
    Cv   = 7,
}

/// Mirror of Triton's `EvictionPolicy` enum (TritonAttrDefs.td).
///
/// Controls cache eviction policy for load/store ops.
/// Integer values must stay in sync with the TableGen definition so that
/// the resulting `IntegerAttr<i32>` is accepted as an `EvictionPolicyAttr`.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum EvictionPolicy {
    Normal     = 1,
    EvictFirst = 2,
    EvictLast  = 3,
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

/// Build a `tt.store` operation.
///
/// Stores `value` to the memory location(s) described by `ptr`.
/// An optional `mask` (same shape as `ptr`) guards individual lanes; elements
/// whose mask bit is `false` are not written.
///
/// # Arguments
/// * `ptr`   – pointer value (`!tt.ptr<T>` or `tensor<Nx!tt.ptr<T>>`).
/// * `value` – value to store; must match the pointee type of `ptr`.
/// * `mask`  – optional boolean lane mask (`i1` or `tensor<Nxi1>`).
/// * `cache` – L1/L2 cache modifier (use `CacheModifier::None` for default).
/// * `evict` – cache eviction policy (use `EvictionPolicy::Normal` for default).
///
/// Assembly format:
/// ```text
/// tt.store %ptr, %value [, %mask] {cache = <cache>, evict = <evict>}
///     : type(%ptr), type(%value)
/// ```
pub fn store<'ctx>(
    context: &'ctx Context,
    location: Location<'ctx>,
    ptr: Value<'ctx, 'ctx>,
    value: Value<'ctx, 'ctx>,
    mask: Option<Value<'ctx, 'ctx>>,
    cache: CacheModifier,
    evict: EvictionPolicy,
) -> Result<Operation<'ctx>, Error> {
    let mask_size = if mask.is_some() { 1i32 } else { 0i32 };

    let mut operands: Vec<Value> = vec![ptr, value];
    if let Some(m) = mask {
        operands.push(m);
    }

    // operandSegmentSizes encodes [ptr=1, value=1, mask=0|1] so the verifier
    // knows which optional operand slots are populated.
    let seg_sizes = DenseI32ArrayAttribute::new(context, &[1, 1, mask_size]);

    // CacheModifierAttr and EvictionPolicyAttr are both backed by IntegerAttr<i32>;
    // attr_i32 produces a compatible value that satisfies classof for each kind.
    let cache_attr = attr_i32(context, cache as i32);
    let evict_attr = attr_i32(context, evict as i32);

    OperationBuilder::new("tt.store", location)
        .add_operands(&operands)
        .add_attributes(&[
            (
                Identifier::new(context, "operandSegmentSizes"),
                Attribute::from(seg_sizes),
            ),
            (Identifier::new(context, "cache"), Attribute::from(cache_attr)),
            (Identifier::new(context, "evict"), Attribute::from(evict_attr)),
        ])
        .build()
        .map_err(|e| Error::InvalidType { msg: format!("failed to build tt.store: {e}") })
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

/// Build a `tt.descriptor_load` operation.
///
/// Lowers to NVIDIA TMA load, reading a tile from global memory into a
/// register tensor using a TMA descriptor.
///
/// # Arguments
/// * `desc`      – value of type `!tt.tensordesc<tensor<...>>`; describes the
///                 source tile layout and strides in global memory.
/// * `indices`   – slice of scalar `i32` values giving the multi-dimensional
///                 load offset (one index per descriptor dimension).
/// * `result_ty` – the tensor type of the loaded result; must match the
///                 element type and shape encoded in the descriptor.
/// * `cache`     – L1/L2 cache modifier (default: `CacheModifier::None`).
/// * `evict`     – cache eviction hint (default: `EvictionPolicy::Normal`).
///
/// Assembly format:
/// ```text
/// %result = tt.descriptor_load %desc[%i0, %i1, ...]
///     [cacheModifier = <cache>] [evictionPolicy = <evict>]
///     : !tt.tensordesc<tensor<...>> -> tensor<...>
/// ```
pub fn descriptor_load<'ctx>(
    context: &'ctx Context,
    location: Location<'ctx>,
    desc: Value<'ctx, 'ctx>,
    indices: &[Value<'ctx, 'ctx>],
    result_ty: Type<'ctx>,
    cache: CacheModifier,
    evict: EvictionPolicy,
) -> Result<Operation<'ctx>, Error> {
    let mut operands: Vec<Value> = Vec::with_capacity(1 + indices.len());
    operands.push(desc);
    operands.extend_from_slice(indices);

    // CacheModifierAttr and EvictionPolicyAttr are both backed by
    // IntegerAttr<i32>; attr_i32 produces a compatible value that satisfies
    // classof for each attribute kind.
    let cache_attr = attr_i32(context, cache as i32);
    let evict_attr = attr_i32(context, evict as i32);

    let op = OperationBuilder::new("tt.descriptor_load", location)
        .add_operands(&operands)
        .add_attributes(&[
            (Identifier::new(context, "cache"), Attribute::from(cache_attr)),
            (Identifier::new(context, "evict"), Attribute::from(evict_attr)),
        ])
        .add_results(&[result_ty])
        .build()
        .map_err(|e| Error::InvalidType {
            msg: format!("failed to build tt.descriptor_load: {e}"),
        })?;

    Ok(op)
}

/// Build a `tt.dot` operation.
///
/// Computes `d = matrix_multiply(a, b) + c`.
///
/// # Arguments
/// * `a`, `b`                 – input matrices (TT_FpIntTensor)
/// * `c`                      – accumulator; result `d` has the same type
/// * `input_precision`        – how to use tensor cores for f32 inputs
///                              (default: `InputPrecision::IEEE`)
/// * `max_num_imprecise_acc`  – maximum allowed imprecise accumulations
///                              (0 = unlimited; default)
///
/// The result type is identical to `c` (enforced by the TableGen
/// `TypesMatchWith<"result's type matches accumulator's type", "d", "c", "$_self">`
/// constraint).
///
/// Assembly format:
/// ```text
/// %d = tt.dot %a, %b, %c [, inputPrecision = <prec>]
///     { maxNumImpreciseAcc = N : i32 }
///     : type(a) * type(b) -> type(d)
/// ```
pub fn dot<'ctx>(
    context: &'ctx Context,
    location: Location<'ctx>,
    a: Value<'ctx, 'ctx>,
    b: Value<'ctx, 'ctx>,
    c: Value<'ctx, 'ctx>,
    input_precision: InputPrecision,
    max_num_imprecise_acc: i32,
) -> Result<Operation<'ctx>, Error> {
    // Result type matches accumulator type (TypesMatchWith constraint).
    let result_type = c.r#type();

    // InputPrecisionAttr is backed by IntegerAttr<i32>; attr_i32 produces a
    // compatible value that satisfies classof(InputPrecisionAttr).
    let input_prec_attr = attr_i32(context, input_precision as i32);
    let max_imprecise_attr = attr_i32(context, max_num_imprecise_acc);

    let op = OperationBuilder::new("tt.dot", location)
        .add_operands(&[a, b, c])
        .add_attributes(&[
            (
                Identifier::new(context, "inputPrecision"),
                Attribute::from(input_prec_attr),
            ),
            (
                Identifier::new(context, "maxNumImpreciseAcc"),
                Attribute::from(max_imprecise_attr),
            ),
        ])
        .add_results(&[result_type])
        .build()
        .map_err(|e| Error::InvalidType { msg: format!("failed to build tt.dot: {e}") })?;

    Ok(op)
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

/// Build a `tt.split` operation.
///
/// Splits a tensor along its last dimension into two equal halves.
/// The input must be a tensor whose last dimension has size 2.
///
/// # Arguments
/// * `src`      – source tensor; last dimension must be 2
/// * `out_type` – element type for both `outLHS` and `outRHS`
///                (the src type with its last dimension removed)
///
/// # Example
///
/// Input `tensor<4x2xf32>` → two results of type `tensor<4xf32>`.
///
/// Assembly format:
/// ```text
/// %lhs, %rhs = tt.split %src : tensor<4x2xf32> -> tensor<4xf32>
/// ```
pub fn split<'ctx>(
    context: &'ctx Context,
    location: Location<'ctx>,
    src: Value<'ctx, 'ctx>,
    out_type: Type<'ctx>,
) -> Result<Operation<'ctx>, Error> {
    // Both outLHS and outRHS have the same type (TypesMatchWith constraint).
    let op = OperationBuilder::new("tt.split", location)
        .add_operands(&[src])
        .add_results(&[out_type, out_type])
        .build()
        .map_err(|e| Error::InvalidType { msg: format!("failed to build tt.split: {e}") })?;

    Ok(op)
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

    /// Verify that `dot` emits a valid `tt.dot` op with the default IEEE precision.
    ///
    /// Uses function block-arguments as tensor values to avoid constant-folding.
    #[test]
    fn test_dot_ieee() {
        let context = create_test_context();
        load_triton_dialect(&context);

        let location = Location::unknown(&context);
        let module = Module::new(location);

        let f32_type = melior::ir::Type::float32(&context);
        let tensor_16x16: Type = tensor_type(&[16, 16], f32_type).into();

        // Function: (a, b, c: tensor<16x16xf32>) -> tensor<16x16xf32>
        let func_op = create_func(
            &context,
            location,
            "test_dot_ieee",
            "public",
            &[tensor_16x16, tensor_16x16, tensor_16x16],
            &[tensor_16x16],
            0,
        )
        .unwrap();

        let block = Block::new(&[
            (tensor_16x16, location),
            (tensor_16x16, location),
            (tensor_16x16, location),
        ]);

        let a: Value = block.argument(0).unwrap().into();
        let b: Value = block.argument(1).unwrap().into();
        let c: Value = block.argument(2).unwrap().into();

        let dot_op =
            dot(&context, location, a, b, c, InputPrecision::IEEE, 0).unwrap();

        let ret_val: Value = dot_op.result(0).unwrap().into();
        let ret_op = ReturnOperation::builder(&context, location).srcs(&[ret_val]).build();

        block.append_operation(dot_op);
        block.append_operation(ret_op.into());
        func_op.body().unwrap().append_block(block);
        module.body().append_operation(func_op.into());

        let output = module.as_operation().to_string();

        assert!(output.contains("tt.dot"), "missing op mnemonic:\n{output}");
        assert!(
            output.contains("tensor<16x16xf32> * tensor<16x16xf32>"),
            "missing operand types:\n{output}"
        );
        assert!(
            output.contains("-> tensor<16x16xf32>"),
            "missing result type:\n{output}"
        );
    }

    /// Verify that `dot` emits `inputPrecision = tf32` for TF32 precision.
    #[test]
    fn test_dot_tf32_precision() {
        let context = create_test_context();
        load_triton_dialect(&context);

        let location = Location::unknown(&context);
        let module = Module::new(location);

        let f32_type = melior::ir::Type::float32(&context);
        let tensor_16x16: Type = tensor_type(&[16, 16], f32_type).into();

        let func_op = create_func(
            &context,
            location,
            "test_dot_tf32",
            "public",
            &[tensor_16x16, tensor_16x16, tensor_16x16],
            &[tensor_16x16],
            0,
        )
        .unwrap();

        let block = Block::new(&[
            (tensor_16x16, location),
            (tensor_16x16, location),
            (tensor_16x16, location),
        ]);

        let a: Value = block.argument(0).unwrap().into();
        let b: Value = block.argument(1).unwrap().into();
        let c: Value = block.argument(2).unwrap().into();

        let dot_op =
            dot(&context, location, a, b, c, InputPrecision::TF32, 0).unwrap();

        let ret_val: Value = dot_op.result(0).unwrap().into();
        let ret_op = ReturnOperation::builder(&context, location).srcs(&[ret_val]).build();

        block.append_operation(dot_op);
        block.append_operation(ret_op.into());
        func_op.body().unwrap().append_block(block);
        module.body().append_operation(func_op.into());

        let output = module.as_operation().to_string();

        assert!(output.contains("tt.dot"), "missing op mnemonic:\n{output}");
        // TF32 is non-default so inputPrecision must appear in the output.
        assert!(
            output.contains("inputPrecision = tf32") || output.contains("inputPrecision = 0"),
            "missing inputPrecision = tf32:\n{output}"
        );
        assert!(
            output.contains("-> tensor<16x16xf32>"),
            "missing result type:\n{output}"
        );
    }

    /// Verify pretty-printed `tt.dot` with IEEE precision and exact IR form.
    #[test]
    fn test_dot_pretty_format() {
        let context = create_test_context();
        load_triton_dialect(&context);

        let location = Location::unknown(&context);
        let module = Module::new(location);

        let f32_type = melior::ir::Type::float32(&context);
        let tensor_16x16: Type = tensor_type(&[16, 16], f32_type).into();

        let func_op = create_func(
            &context,
            location,
            "test_dot_pretty",
            "public",
            &[tensor_16x16, tensor_16x16, tensor_16x16],
            &[tensor_16x16],
            0,
        )
        .unwrap();

        let block = Block::new(&[
            (tensor_16x16, location),
            (tensor_16x16, location),
            (tensor_16x16, location),
        ]);

        let a: Value = block.argument(0).unwrap().into();
        let b: Value = block.argument(1).unwrap().into();
        let c: Value = block.argument(2).unwrap().into();

        let dot_op =
            dot(&context, location, a, b, c, InputPrecision::IEEE, 0).unwrap();

        let ret_val: Value = dot_op.result(0).unwrap().into();
        let ret_op = ReturnOperation::builder(&context, location).srcs(&[ret_val]).build();

        block.append_operation(dot_op);
        block.append_operation(ret_op.into());
        func_op.body().unwrap().append_block(block);
        module.body().append_operation(func_op.into());

        let output = module.as_operation().to_string();

        // The pretty-printed form must include the tt.dot mnemonic with proper types.
        // inputPrecision = ieee may be omitted when it equals the default value.
        let expected_fragment =
            "tt.dot %arg0, %arg1, %arg2 : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>";
        assert!(
            output.contains(expected_fragment),
            "expected pretty fragment not found.\nExpected fragment:\n  {expected_fragment}\nActual output:\n{output}"
        );
    }

    /// Verify that `descriptor_load` emits the correct `tt.descriptor_load` IR.
    ///
    /// Uses function block-arguments to supply typed values without needing a
    /// real TMA descriptor at compile time.
    #[test]
    fn test_descriptor_load() {
        let context = create_test_context();
        load_triton_dialect(&context);

        let location = Location::unknown(&context);
        let module = Module::new(location);

        // Result type: tensor<16x16xf32>.
        let f32_type = melior::ir::Type::float32(&context);
        let result_tensor_ty: Type = tensor_type(&[16, 16], f32_type).into();

        // Descriptor type: !tt.tensordesc<tensor<16x16xf32>>.
        let desc_ty = Type::parse(&context, "!tt.tensordesc<tensor<16x16xf32>>")
            .expect("valid tensordesc type");

        // Two i32 indices for a 2-D descriptor.
        let i32_type: Type = IntegerType::new(&context, 32).into();

        // Build a wrapper function so operands have proper block-argument types.
        let func_op = create_func(
            &context,
            location,
            "test_descriptor_load",
            "public",
            &[desc_ty, i32_type, i32_type],
            &[result_tensor_ty],
            0,
        )
        .unwrap();

        let block = Block::new(&[
            (desc_ty, location),
            (i32_type, location),
            (i32_type, location),
        ]);

        let desc: Value = block.argument(0).unwrap().into();
        let idx0: Value = block.argument(1).unwrap().into();
        let idx1: Value = block.argument(2).unwrap().into();

        let load_op: Operation<'_> = descriptor_load(
            &context,
            location,
            desc,
            &[idx0, idx1],
            result_tensor_ty,
            CacheModifier::None,
            EvictionPolicy::Normal,
        )
        .unwrap();

        let ret_val: Value = load_op.result(0).unwrap().into();
        let ret_op = ReturnOperation::builder(&context, location).srcs(&[ret_val]).build();

        block.append_operation(load_op);
        block.append_operation(ret_op.into());
        func_op.body().unwrap().append_block(block);
        module.body().append_operation(func_op.into());

        let output = module.as_operation().to_string();

        assert!(
            output.contains("tt.descriptor_load"),
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

    /// Verify that `store` without a mask emits a valid `tt.store` op.
    ///
    /// Uses function block-arguments to supply typed pointer and value operands
    /// so the test does not depend on constant-folding or a real GPU pointer.
    #[test]
    fn test_store_no_mask() {
        let context = create_test_context();
        load_triton_dialect(&context);

        let location = Location::unknown(&context);
        let module = Module::new(location);

        let f32_type = melior::ir::Type::float32(&context);
        // Tensor pointer: tensor<8x!tt.ptr<f32>>
        let ptr_elem_ty = pointer_type(f32_type);
        let ptr_tensor_ty: Type = tensor_type(&[8], ptr_elem_ty).into();
        // Value tensor: tensor<8xf32>
        let val_tensor_ty: Type = tensor_type(&[8], f32_type).into();

        // Wrapper function so block-arguments carry their types.
        let func_op = create_func(
            &context,
            location,
            "test_store_no_mask",
            "public",
            &[ptr_tensor_ty, val_tensor_ty],
            &[],
            0,
        )
        .unwrap();

        let block = Block::new(&[(ptr_tensor_ty, location), (val_tensor_ty, location)]);

        let ptr_val: melior::ir::Value = block.argument(0).unwrap().into();
        let val_val: melior::ir::Value = block.argument(1).unwrap().into();

        let store_op: Operation<'_> =
            store(&context, location, ptr_val, val_val, None, CacheModifier::None, EvictionPolicy::Normal)
                .unwrap();

        let ret_op = ReturnOperation::builder(&context, location).srcs(&[]).build();

        block.append_operation(store_op);
        block.append_operation(ret_op.into());
        func_op.body().unwrap().append_block(block);
        module.body().append_operation(func_op.into());

        let output = module.as_operation().to_string();

        assert!(output.contains("tt.store"), "missing op mnemonic:\n{output}");
        assert!(
            output.contains("tensor<8x!tt.ptr<f32>>"),
            "missing ptr tensor type:\n{output}"
        );
        assert!(
            output.contains("tensor<8xf32>"),
            "missing value tensor type:\n{output}"
        );
    }

    /// Verify that `store` with a mask emits a valid `tt.store` op including the
    /// mask operand.
    #[test]
    fn test_store_with_mask() {
        let context = create_test_context();
        load_triton_dialect(&context);

        let location = Location::unknown(&context);
        let module = Module::new(location);

        let f32_type = melior::ir::Type::float32(&context);
        let i1_type: Type = melior::ir::r#type::IntegerType::new(&context, 1).into();

        // Tensor types: tensor<16x!tt.ptr<f32>>, tensor<16xf32>, tensor<16xi1>.
        let ptr_elem_ty = pointer_type(f32_type);
        let ptr_tensor_ty: Type = tensor_type(&[16], ptr_elem_ty).into();
        let val_tensor_ty: Type = tensor_type(&[16], f32_type).into();
        let mask_tensor_ty: Type = tensor_type(&[16], i1_type).into();

        let func_op = create_func(
            &context,
            location,
            "test_store_masked",
            "public",
            &[ptr_tensor_ty, val_tensor_ty, mask_tensor_ty],
            &[],
            0,
        )
        .unwrap();

        let block = Block::new(&[
            (ptr_tensor_ty, location),
            (val_tensor_ty, location),
            (mask_tensor_ty, location),
        ]);

        let ptr_val: melior::ir::Value = block.argument(0).unwrap().into();
        let val_val: melior::ir::Value = block.argument(1).unwrap().into();
        let mask_val: melior::ir::Value = block.argument(2).unwrap().into();

        let store_op: Operation<'_> = store(
            &context,
            location,
            ptr_val,
            val_val,
            Some(mask_val),
            CacheModifier::None,
            EvictionPolicy::Normal,
        )
        .unwrap();

        let ret_op = ReturnOperation::builder(&context, location).srcs(&[]).build();

        block.append_operation(store_op);
        block.append_operation(ret_op.into());
        func_op.body().unwrap().append_block(block);
        module.body().append_operation(func_op.into());

        let output = module.as_operation().to_string();

        assert!(output.contains("tt.store"), "missing op mnemonic:\n{output}");
        assert!(
            output.contains("tensor<16x!tt.ptr<f32>>"),
            "missing ptr tensor type:\n{output}"
        );
        assert!(
            output.contains("tensor<16xf32>"),
            "missing value tensor type:\n{output}"
        );
        assert!(
            output.contains("tensor<16xi1>"),
            "missing mask tensor type:\n{output}"
        );
    }

    /// Verify that `store` with a non-default cache modifier emits the cache
    /// attribute in the IR output.
    #[test]
    fn test_store_cache_modifier() {
        let context = create_test_context();
        load_triton_dialect(&context);

        let location = Location::unknown(&context);
        let module = Module::new(location);

        let f32_type = melior::ir::Type::float32(&context);
        let ptr_elem_ty = pointer_type(f32_type);
        let ptr_tensor_ty: Type = tensor_type(&[8], ptr_elem_ty).into();
        let val_tensor_ty: Type = tensor_type(&[8], f32_type).into();

        let func_op = create_func(
            &context,
            location,
            "test_store_cache",
            "public",
            &[ptr_tensor_ty, val_tensor_ty],
            &[],
            0,
        )
        .unwrap();

        let block = Block::new(&[(ptr_tensor_ty, location), (val_tensor_ty, location)]);

        let ptr_val: melior::ir::Value = block.argument(0).unwrap().into();
        let val_val: melior::ir::Value = block.argument(1).unwrap().into();

        // CacheModifier::Ca (write-back cache) and EvictionPolicy::EvictFirst.
        let store_op: Operation<'_> = store(
            &context,
            location,
            ptr_val,
            val_val,
            None,
            CacheModifier::Ca,
            EvictionPolicy::EvictFirst,
        )
        .unwrap();

        let ret_op = ReturnOperation::builder(&context, location).srcs(&[]).build();

        block.append_operation(store_op);
        block.append_operation(ret_op.into());
        func_op.body().unwrap().append_block(block);
        module.body().append_operation(func_op.into());

        let output = module.as_operation().to_string();

        assert!(output.contains("tt.store"), "missing op mnemonic:\n{output}");
        // Non-default cache/evict attributes must appear in the emitted IR.
        // Pretty-printed form: "cacheModifier = ca" / "evictionPolicy = evict_first".
        // Generic form: "cache = 2 : i32" / "evict = 2 : i32".
        assert!(
            output.contains("cacheModifier = ca") || output.contains("cache = 2"),
            "missing cacheModifier = ca (CacheModifier::Ca = 2):\n{output}"
        );
        assert!(
            output.contains("evictionPolicy = evict_first") || output.contains("evict = 2"),
            "missing evictionPolicy = evict_first (EvictionPolicy::EvictFirst = 2):\n{output}"
        );
    }

    /// Verify that `split` emits the correct `tt.split` op.
    ///
    /// Input tensor `tensor<4x2xf32>` → two outputs of type `tensor<4xf32>`.
    #[test]
    fn test_split() {
        let context = create_test_context();
        load_triton_dialect(&context);

        let location = Location::unknown(&context);
        let module = Module::new(location);

        let f32_type = melior::ir::Type::float32(&context);
        // Input: tensor<4x2xf32>  (last dimension must be 2)
        let src_ty: Type = tensor_type(&[4, 2], f32_type).into();
        // Each output: tensor<4xf32>
        let out_ty: Type = tensor_type(&[4], f32_type).into();

        let func_op = create_func(
            &context,
            location,
            "test_split",
            "public",
            &[src_ty],
            &[out_ty, out_ty],
            0,
        )
        .unwrap();

        let block = Block::new(&[(src_ty, location)]);
        let src: Value = block.argument(0).unwrap().into();

        let split_op: Operation<'_> = split(&context, location, src, out_ty).unwrap();

        let lhs: Value = split_op.result(0).unwrap().into();
        let rhs: Value = split_op.result(1).unwrap().into();
        let ret_op = ReturnOperation::builder(&context, location).srcs(&[lhs, rhs]).build();

        block.append_operation(split_op);
        block.append_operation(ret_op.into());
        func_op.body().unwrap().append_block(block);
        module.body().append_operation(func_op.into());

        let output = module.as_operation().to_string();

        assert!(output.contains("tt.split"), "missing op mnemonic:\n{output}");
        assert!(
            output.contains("tensor<4x2xf32>"),
            "missing src tensor type:\n{output}"
        );
        assert!(
            output.contains("tensor<4xf32>"),
            "missing output tensor type:\n{output}"
        );
    }

    /// Verify pretty-printed `tt.split` with exact IR form.
    #[test]
    fn test_split_pretty_format() {
        let context = create_test_context();
        load_triton_dialect(&context);

        let location = Location::unknown(&context);
        let module = Module::new(location);

        let f32_type = melior::ir::Type::float32(&context);
        let src_ty: Type = tensor_type(&[4, 2], f32_type).into();
        let out_ty: Type = tensor_type(&[4], f32_type).into();

        let func_op = create_func(
            &context,
            location,
            "test_split_pretty",
            "public",
            &[src_ty],
            &[out_ty, out_ty],
            0,
        )
        .unwrap();

        let block = Block::new(&[(src_ty, location)]);
        let src: Value = block.argument(0).unwrap().into();

        let split_op: Operation<'_> = split(&context, location, src, out_ty).unwrap();

        let lhs: Value = split_op.result(0).unwrap().into();
        let rhs: Value = split_op.result(1).unwrap().into();
        let ret_op = ReturnOperation::builder(&context, location).srcs(&[lhs, rhs]).build();

        block.append_operation(split_op);
        block.append_operation(ret_op.into());
        func_op.body().unwrap().append_block(block);
        module.body().append_operation(func_op.into());

        let output = module.as_operation().to_string();

        // Pretty-printed form: "tt.split %arg0 : tensor<4x2xf32> -> tensor<4xf32>"
        assert!(
            output.contains("tt.split"),
            "missing tt.split mnemonic:\n{output}"
        );
        assert!(
            output.contains("tensor<4x2xf32>") && output.contains("tensor<4xf32>"),
            "missing type annotations in tt.split output:\n{output}"
        );
    }
}
