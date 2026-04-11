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
use melior::ir::attribute::{BoolAttribute, DenseI32ArrayAttribute, StringAttribute};
use melior::ir::operation::{OperationBuilder, OperationMutLike};
use melior::ir::r#type::{IntegerType, RankedTensorType};
use melior::ir::{Attribute, Identifier, Location, Operation, Type, TypeLike, Value, ValueLike};

use crate::errors::Error;
use crate::ffi::mlirCreateTritonPointerType;
use crate::shared::builtin::tensor_type;
use crate::triton::attr_i32;
use crate::triton::tt::{
    AddPtrOperation, DescriptorGatherOperation, DescriptorScatterOperation, MapElementwiseOperation,
    MapElementwiseReturnOperation, MakeRangeOperation, MulhiUIOperation, PreciseDivFOperation,
    PreciseSqrtOperation, ReduceOperation, ReduceReturnOperation, ReturnOperation, ScanOperation,
    ScanReturnOperation, SplatOperation,
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

/// Mirror of Triton's `PaddingOption` enum (TritonAttrDefs.td).
///
/// Controls out-of-bounds padding behaviour for `tt.make_tensor_descriptor`.
/// Integer values must stay in sync with the TableGen definition so that
/// the resulting `IntegerAttr<i32>` is accepted as a `PaddingOptionAttr`.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum PaddingOption {
    PadZero = 1,
    PadNan  = 2,
}

/// Mirror of Triton's `RoundingMode` enum (TritonAttrDefs.td).
///
/// Controls the rounding mode for `tt.fp_to_fp` floating-point casts.
/// Integer values must stay in sync with the TableGen definition so that
/// the resulting `IntegerAttr<i32>` is accepted as a `RoundingModeAttr`.
///
/// - `RTZ`  = 0 – round toward zero
/// - `RTNE` = 1 – round to nearest, ties to even
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum RoundingMode {
    RTZ  = 0,
    RTNE = 1,
}

/// Mirror of Triton's `PropagateNan` enum (TritonAttrDefs.td).
///
/// Controls NaN propagation behaviour for `tt.clampf`.
/// Integer values must stay in sync with the TableGen definition so that
/// the resulting `IntegerAttr<i32>` is accepted as a `PropagateNanAttr`.
///
/// - `None` = 0      – do not propagate NaN; clamp NaN inputs as if they were in-range
/// - `All`  = 0xFFFF – propagate NaN; a NaN input produces a NaN output
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum PropagateNan {
    None = 0,
    All  = 0xFFFF,
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

/// Build a `tt.load` operation.
///
/// Loads from a tensor of pointers (or a tensor pointer).  An optional `mask`
/// guards individual lanes; masked-out lanes are not read.  An optional `other`
/// value fills masked-out lanes in the result.
///
/// # Arguments
/// * `ptr`         – pointer operand (`!tt.ptr<T>` or `tensor<Nx!tt.ptr<T>>`).
/// * `mask`        – optional boolean lane mask (`i1` or `tensor<Nxi1>`).
/// * `other`       – optional fill value for masked lanes; type must match
///                   the pointee type of `ptr`.
/// * `result_ty`   – result type (pointee type of `ptr`, e.g. `f32` or
///                   `tensor<Nxf32>`).
/// * `cache`       – L1/L2 cache modifier (use `CacheModifier::None` for
///                   the default behaviour).
/// * `evict`       – cache eviction policy (use `EvictionPolicy::Normal` for
///                   the default behaviour).
/// * `is_volatile` – if `true`, the load is treated as volatile (not
///                   reordered or cached by the compiler).
///
/// Assembly format:
/// ```text
/// tt.load %ptr [, %mask [, %other]]
///   [cacheModifier = <cache>] [evictionPolicy = <evict>]
///   attr-dict : type(%ptr)
/// ```
pub fn load<'ctx>(
    context: &'ctx Context,
    location: Location<'ctx>,
    ptr: Value<'ctx, 'ctx>,
    mask: Option<Value<'ctx, 'ctx>>,
    other: Option<Value<'ctx, 'ctx>>,
    result_ty: Type<'ctx>,
    cache: CacheModifier,
    evict: EvictionPolicy,
    is_volatile: bool,
) -> Result<Operation<'ctx>, Error> {
    let mask_size = if mask.is_some() { 1i32 } else { 0i32 };
    let other_size = if other.is_some() { 1i32 } else { 0i32 };

    let mut operands: Vec<Value> = vec![ptr];
    if let Some(m) = mask {
        operands.push(m);
    }
    if let Some(o) = other {
        operands.push(o);
    }

    // operandSegmentSizes encodes [ptr=1, mask=0|1, other=0|1] so the
    // verifier knows which optional operand slots are populated.
    let seg_sizes = DenseI32ArrayAttribute::new(context, &[1, mask_size, other_size]);

    // CacheModifierAttr and EvictionPolicyAttr are both backed by
    // IntegerAttr<i32>; attr_i32 produces a compatible value that satisfies
    // classof for each attribute kind.
    let cache_attr = attr_i32(context, cache as i32);
    let evict_attr = attr_i32(context, evict as i32);
    let volatile_attr = BoolAttribute::new(context, is_volatile);

    OperationBuilder::new("tt.load", location)
        .add_operands(&operands)
        .add_attributes(&[
            (
                Identifier::new(context, "operandSegmentSizes"),
                Attribute::from(seg_sizes),
            ),
            (Identifier::new(context, "cache"), Attribute::from(cache_attr)),
            (Identifier::new(context, "evict"), Attribute::from(evict_attr)),
            (
                Identifier::new(context, "isVolatile"),
                Attribute::from(volatile_attr),
            ),
        ])
        .add_results(&[result_ty])
        .build()
        .map_err(|e| Error::InvalidType { msg: format!("failed to build tt.load: {e}") })
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

/// Build a `tt.descriptor_scatter` operation.
///
/// Lowers to NVIDIA TMA scatter, writing multiple rows from a single 2D
/// source tensor back to global memory via a TMA descriptor.
///
/// # Arguments
/// * `desc`      – value of type `!tt.tensordesc<tensor<1xNxT>>`;
///                 the descriptor block must have exactly one row.
/// * `x_offsets` – 1-D `tensor<Kxi32>` of column offsets to scatter to.
/// * `y_offset`  – scalar `i32` row offset.
/// * `src`       – 2-D source tensor to scatter (e.g. `tensor<KxNxT>`).
///
/// The op's assembly format is:
/// ```text
/// tt.descriptor_scatter %desc[%x_offsets, %y_offset], %src
///     : type(desc), type(x_offsets), type(y_offset), type(src)
/// ```
pub fn descriptor_scatter<'ctx>(
    context: &'ctx Context,
    location: Location<'ctx>,
    desc: Value<'ctx, 'ctx>,
    x_offsets: Value<'ctx, 'ctx>,
    y_offset: Value<'ctx, 'ctx>,
    src: Value<'ctx, 'ctx>,
) -> Result<DescriptorScatterOperation<'ctx>, Error> {
    Ok(DescriptorScatterOperation::builder(context, location)
        .desc(desc)
        .x_offsets(x_offsets)
        .y_offset(y_offset)
        .src(src)
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

/// Build a `tt.descriptor_store` operation.
///
/// Lowers to NVIDIA TMA store, writing a register tensor tile back to global
/// memory using a TMA descriptor.
///
/// # Arguments
/// * `desc`    – value of type `!tt.tensordesc<tensor<...>>`; describes the
///               destination tile layout and strides in global memory.
/// * `src`     – the tensor value to store; element type and shape must match
///               the descriptor.
/// * `indices` – slice of scalar `i32` values giving the multi-dimensional
///               store offset (one index per descriptor dimension).
///
/// Assembly format:
/// ```text
/// tt.descriptor_store %desc[%i0, %i1, ...], %src
///     : !tt.tensordesc<tensor<...>>, tensor<...>
/// ```
pub fn descriptor_store<'ctx>(
    context: &'ctx Context,
    location: Location<'ctx>,
    desc: Value<'ctx, 'ctx>,
    src: Value<'ctx, 'ctx>,
    indices: &[Value<'ctx, 'ctx>],
) -> Result<Operation<'ctx>, Error> {
    // Operand order: desc, src, indices (matches TableGen $desc, $src, $indices).
    let mut operands: Vec<Value> = Vec::with_capacity(2 + indices.len());
    operands.push(desc);
    operands.push(src);
    operands.extend_from_slice(indices);

    let op = OperationBuilder::new("tt.descriptor_store", location)
        .add_operands(&operands)
        .build()
        .map_err(|e| Error::InvalidType {
            msg: format!("failed to build tt.descriptor_store: {e}"),
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

/// Build a `tt.join` operation.
///
/// Joins two tensors of the same shape along a new, minor (last) dimension.
/// For example, two `tensor<4x8xf32>` inputs produce a `tensor<4x8x2xf32>`
/// result.  Both inputs must have identical shape (`SameTypeOperands` trait).
///
/// # Arguments
/// * `lhs`       – first input tensor.
/// * `rhs`       – second input tensor (same type as `lhs`).
/// * `result_ty` – result tensor type (same element type, one new trailing dim
///                 of size 2).
///
/// # Example
///
/// ```text
/// %result = tt.join %lhs, %rhs : tensor<4x8xf32> -> tensor<4x8x2xf32>
/// ```
///
/// Assembly format (from TableGen):
/// ```text
/// $lhs `,` $rhs attr-dict `:` type($lhs) `->` type($result)
/// ```
pub fn join<'ctx>(
    _context: &'ctx Context,
    location: Location<'ctx>,
    lhs: Value<'ctx, '_>,
    rhs: Value<'ctx, '_>,
    result_ty: Type<'ctx>,
) -> Result<Operation<'ctx>, Error> {
    OperationBuilder::new("tt.join", location)
        .add_operands(&[lhs, rhs])
        .add_results(&[result_ty])
        .build()
        .map_err(|e| Error::InvalidType { msg: format!("failed to build tt.join: {e}") })
}

/// Build a `tt.reshape` operation.
///
/// Reinterprets the elements of a tensor to a different shape.  The total
/// element count must be identical in source and result; the element type is
/// unchanged.
///
/// # Arguments
/// * `src`              – source tensor to reshape.
/// * `result_ty`        – target tensor type (same element type, new shape).
/// * `allow_reorder`    – when `true` the compiler may change element order
///                        to produce more efficient code.
/// * `efficient_layout` – hint that the destination layout should be
///                        preserved for performance (compiler may override).
///
/// # Example
///
/// ```text
/// // Without flags:
/// %1 = tt.reshape %0 : tensor<8xf32> -> tensor<2x4xf32>
///
/// // With allow_reorder:
/// %1 = tt.reshape %0 allow_reorder : tensor<8xf32> -> tensor<2x4xf32>
/// ```
///
/// Assembly format (from TableGen):
/// ```text
/// $src (`allow_reorder` $allow_reorder^)? (`efficient_layout` $efficient_layout^)? attr-dict `:` type($src) `->` type($result)
/// ```
pub fn reshape<'ctx>(
    context: &'ctx Context,
    location: Location<'ctx>,
    src: Value<'ctx, 'ctx>,
    result_ty: Type<'ctx>,
    allow_reorder: bool,
    efficient_layout: bool,
) -> Result<Operation<'ctx>, Error> {
    let mut builder = OperationBuilder::new("tt.reshape", location)
        .add_operands(&[src])
        .add_results(&[result_ty]);

    if allow_reorder {
        builder = builder.add_attributes(&[(
            Identifier::new(context, "allow_reorder"),
            Attribute::unit(context),
        )]);
    }
    if efficient_layout {
        builder = builder.add_attributes(&[(
            Identifier::new(context, "efficient_layout"),
            Attribute::unit(context),
        )]);
    }

    builder
        .build()
        .map_err(|e| Error::InvalidType { msg: format!("failed to build tt.reshape: {e}") })
}

/// Build a `tt.scan.return` operation.
///
/// `tt.scan.return` is the terminator for the `combineOp` region inside a
/// `tt.scan` op.  It returns the element-wise combined values back to the
/// scan loop.  The operands must match the element types declared by the
/// enclosing `tt.scan` op (one scalar per input tensor).
///
/// # Arguments
/// * `result` – the values to return; their types must match the element
///              types of the enclosing `tt.scan` combineOp block arguments.
///
/// # Example
///
/// ```text
/// // Inside a tt.scan combineOp region where %arg0 and %arg1 : f32:
/// tt.scan.return %arg0 : f32
/// ```
///
/// Assembly format (from TableGen):
/// ```text
/// tt.scan.return $result attr-dict : type($result)
/// ```
pub fn scan_return<'ctx>(
    context: &'ctx Context,
    location: Location<'ctx>,
    result: &[Value<'ctx, '_>],
) -> Result<ScanReturnOperation<'ctx>, Error> {
    Ok(ScanReturnOperation::builder(context, location).result(result).build())
}

/// Build a `tt.reduce.return` operation.
///
/// `tt.reduce.return` is the terminator for the `combineOp` region inside a
/// `tt.reduce` op.  It returns the element-wise combined values back to the
/// reduction loop.  The operands must match the element types declared by the
/// enclosing `tt.reduce` op (one scalar per input tensor).
///
/// # Arguments
/// * `result` – the values to return; their types must match the element
///              types of the enclosing `tt.reduce` combineOp block arguments.
///
/// # Example
///
/// ```text
/// // Inside a tt.reduce combineOp region where %arg0 and %arg1 : f32:
/// tt.reduce.return %arg0 : f32
/// ```
///
/// Assembly format (from TableGen):
/// ```text
/// tt.reduce.return $result attr-dict : type($result)
/// ```
pub fn reduce_return<'ctx>(
    context: &'ctx Context,
    location: Location<'ctx>,
    result: &[Value<'ctx, '_>],
) -> Result<ReduceReturnOperation<'ctx>, Error> {
    Ok(ReduceReturnOperation::builder(context, location).result(result).build())
}

/// Build a `tt.map_elementwise.return` operation.
///
/// `tt.map_elementwise.return` is the terminator for the `scalarOp` region
/// inside a `tt.map_elementwise` op.  It yields the scalar results produced
/// by the body back to the enclosing map operation.  The operand types must
/// match the element types of the corresponding result tensors declared by
/// the enclosing `tt.map_elementwise` op.
///
/// # Arguments
/// * `result` – scalar values to yield; types must match the element types of
///              the enclosing `tt.map_elementwise` result tensors.
///
/// # Example
///
/// ```text
/// // Inside a tt.map_elementwise scalarOp region where %arg0 : f32:
/// tt.map_elementwise.return %arg0 : f32
/// ```
///
/// Assembly format (from TableGen):
/// ```text
/// tt.map_elementwise.return attr-dict ($result^ `:` type($result))?
/// ```
pub fn map_elementwise_return<'ctx>(
    context: &'ctx Context,
    location: Location<'ctx>,
    result: &[Value<'ctx, '_>],
) -> Result<MapElementwiseReturnOperation<'ctx>, Error> {
    Ok(MapElementwiseReturnOperation::builder(context, location).result(result).build())
}

/// Build a `tt.map_elementwise` operation.
///
/// `tt.map_elementwise` applies a scalar computation (expressed as a region)
/// element-wise over one or more input tensors, producing one output tensor per
/// input.  The `scalarOp` region receives the element values as block arguments
/// and must be terminated by a `tt.map_elementwise.return`.
///
/// # Arguments
/// * `srcs`         – input tensors; all must share the same shape and encoding.
/// * `result_types` – result tensor types, one per source tensor (same shape/encoding).
/// * `pack`         – packing factor attribute (`I32Attr`); use `0` for unpacked.
/// * `scalar_op`    – the `scalarOp` region: a single block whose arguments are
///                    the element values of each source tensor, terminated by
///                    `tt.map_elementwise.return`.
///
/// # Example
///
/// ```text
/// %result = tt.map_elementwise %src {pack = 0 : i32} (
///   ^bb0(%elem: f32):
///     tt.map_elementwise.return %elem : f32
/// ) : (tensor<4xf32>) -> tensor<4xf32>
/// ```
///
/// # Panics / Errors
///
/// Returns an error if the operation builder rejects the supplied operands or
/// result types.
pub fn map_elementwise<'ctx>(
    context: &'ctx Context,
    location: Location<'ctx>,
    srcs: &[Value<'ctx, '_>],
    result_types: &[Type<'ctx>],
    pack: i32,
    scalar_op: melior::ir::Region<'ctx>,
) -> Result<MapElementwiseOperation<'ctx>, Error> {
    let pack_attr = attr_i32(context, pack);

    Ok(MapElementwiseOperation::builder(context, location)
        .result(result_types)
        .srcs(srcs)
        .scalar_op(scalar_op)
        .pack(pack_attr)
        .build())
}

/// Build a `tt.scan` operation.
///
/// `tt.scan` performs an associative prefix scan over tensors along a given
/// axis using a user-supplied combination region.  Each input tensor produces
/// one output tensor of the same shape and element type.
///
/// # Arguments
/// * `srcs`        – input tensors; must all have the same shape and encoding.
/// * `result_types` – result types, one per source tensor (same type as the
///                   corresponding `src`).
/// * `axis`        – the tensor axis to scan over (0-based, i32).
/// * `reverse`     – when `true` the scan is performed in reverse order.
/// * `combine_op`  – the `combineOp` region: a single block whose arguments
///                   are 2 × len(srcs) scalars (lhs₀, rhs₀, lhs₁, rhs₁, …)
///                   terminated by a `tt.scan.return`.
///
/// # Example
///
/// ```text
/// %result = tt.scan %src {axis = 0 : i32, reverse = false} (
///   ^bb0(%lhs: f32, %rhs: f32):
///     tt.scan.return %lhs : f32
/// ) : (tensor<4xf32>) -> tensor<4xf32>
/// ```
///
/// # Panics / Errors
///
/// Returns an error if the operation builder rejects the supplied operands
/// or result types.
pub fn scan<'ctx>(
    context: &'ctx Context,
    location: Location<'ctx>,
    srcs: &[Value<'ctx, '_>],
    result_types: &[Type<'ctx>],
    axis: i32,
    reverse: bool,
    combine_op: melior::ir::Region<'ctx>,
) -> Result<ScanOperation<'ctx>, Error> {
    let axis_attr = attr_i32(context, axis);
    let reverse_attr: Attribute = BoolAttribute::new(context, reverse).into();

    Ok(ScanOperation::builder(context, location)
        .result(result_types)
        .srcs(srcs)
        .combine_op(combine_op)
        .axis(axis_attr)
        .reverse(reverse_attr)
        .build())
}

/// Build a `tt.reduce` operation.
///
/// `tt.reduce` performs a reduction over tensors along a given axis using a
/// user-supplied combination region.  Each input tensor produces one scalar
/// result whose type is the element type of the corresponding source tensor.
///
/// # Arguments
/// * `srcs`         – input tensors; must all have the same shape and encoding.
/// * `result_types` – result types, one per source tensor (element type of the
///                    corresponding source, i.e. the axis dimension is collapsed).
/// * `axis`         – the tensor axis to reduce along (0-based, i32).
/// * `combine_op`   – the `combineOp` region: a single block whose arguments
///                    are 2 × len(srcs) scalars (lhs₀, rhs₀, lhs₁, rhs₁, …)
///                    terminated by a `tt.reduce.return`.
///
/// # Example
///
/// ```text
/// %result = tt.reduce(%src) {axis = 0 : i32} (
///   ^bb0(%lhs: f32, %rhs: f32):
///     tt.reduce.return %lhs : f32
/// ) : (tensor<4xf32>) -> f32
/// ```
///
/// # Panics / Errors
///
/// Returns an error if the operation builder rejects the supplied operands
/// or result types.
pub fn reduce<'ctx>(
    context: &'ctx Context,
    location: Location<'ctx>,
    srcs: &[Value<'ctx, '_>],
    result_types: &[Type<'ctx>],
    axis: i32,
    combine_op: melior::ir::Region<'ctx>,
) -> Result<ReduceOperation<'ctx>, Error> {
    let axis_attr = attr_i32(context, axis);

    Ok(ReduceOperation::builder(context, location)
        .result(result_types)
        .srcs(srcs)
        .combine_op(combine_op)
        .axis(axis_attr)
        .build())
}

/// Build a `tt.return` operation.
///
/// `tt.return` terminates a Triton function (`tt.func`) and returns zero or
/// more values to the caller.  The operand types must match the enclosing
/// `tt.func`'s result type signature.
///
/// # Arguments
/// * `srcs` – values to return; may be empty for `() -> ()` functions.
///
/// # Example
///
/// ```text
/// tt.return %0, %1 : i32, f32
/// ```
///
/// Build a `tt.precise_sqrt` operation.
///
/// Computes the IEEE-precise square root of a floating-point scalar or tensor.
/// The operation has the `Elementwise`, `SameOperandsAndResultType`, and `Pure`
/// traits, so the result type is inferred from the operand type.
///
/// # Arguments
/// * `x` – floating-point scalar or tensor operand
///
/// # Example
///
/// ```text
/// %result = tt.precise_sqrt %x : f32
/// %result = tt.precise_sqrt %x : tensor<8xf32>
/// ```
///
/// Assembly format (from TableGen):
/// ```text
/// tt.precise_sqrt $x attr-dict `:` type($x)
/// ```
pub fn precise_sqrt<'ctx>(
    context: &'ctx Context,
    location: Location<'ctx>,
    x: Value<'ctx, 'ctx>,
) -> Result<PreciseSqrtOperation<'ctx>, Error> {
    // SameOperandsAndResultType: result type is inferred from the operand type.
    Ok(PreciseSqrtOperation::builder(context, location).x(x).build())
}

/// Build a `tt.precise_divf` operation.
///
/// Computes the IEEE-precise floating-point division of two scalars or tensors.
/// The operation has the `Elementwise`, `SameOperandsAndResultType`, and `Pure`
/// traits, so the result type is inferred from the operand types.
///
/// # Arguments
/// * `x` – dividend: floating-point scalar or tensor
/// * `y` – divisor: floating-point scalar or tensor (same type as `x`)
///
/// # Example
///
/// ```text
/// %result = tt.precise_divf %x, %y : f32
/// %result = tt.precise_divf %x, %y : tensor<8xf32>
/// ```
///
/// Assembly format (from TableGen):
/// ```text
/// tt.precise_divf $x, $y attr-dict `:` type($x)
/// ```
pub fn precise_divf<'ctx>(
    context: &'ctx Context,
    location: Location<'ctx>,
    x: Value<'ctx, 'ctx>,
    y: Value<'ctx, 'ctx>,
) -> Result<PreciseDivFOperation<'ctx>, Error> {
    // SameOperandsAndResultType: result type is inferred from the operand types.
    Ok(PreciseDivFOperation::builder(context, location).x(x).y(y).build())
}

/// Assembly format (from TableGen):
/// ```text
/// tt.return ($srcs^ `:` type($srcs))? attr-dict
/// ```
///
/// # Panics / Errors
///
/// Returns an error if the operation builder rejects the supplied operands.
pub fn return_op<'ctx>(
    context: &'ctx Context,
    location: Location<'ctx>,
    srcs: &[Value<'ctx, '_>],
) -> Result<ReturnOperation<'ctx>, Error> {
    Ok(ReturnOperation::builder(context, location).srcs(srcs).build())
}

/// Build a `tt.make_tensor_ptr` operation.
///
/// Creates a block-tensor pointer from a flat base pointer and per-dimension
/// metadata.  The result type is `!tt.ptr<tensor<D0x…xT>>`, where the
/// dimension sizes are encoded in the result type's tensor shape.
///
/// # Arguments
/// * `base`      – base pointer `!tt.ptr<T>` into the parent tensor.
/// * `shape`     – variadic `i64` values: dimension sizes of the parent tensor.
/// * `strides`   – variadic `i64` values: strides (in elements) of each dimension.
/// * `offsets`   – variadic `i32` values: initial offsets into each dimension.
/// * `order`     – permutation of `[0..rank-1]` specifying the memory layout
///                 (e.g. `[1, 0]` for row-major 2-D).
/// * `result_ty` – the tensor-pointer result type, `!tt.ptr<tensor<…xT>>`.
///
/// `shape`, `strides`, and `offsets` must all have the same length, equal to
/// the rank of the result tensor.
///
/// # Assembly format
/// ```text
/// tt.make_tensor_ptr %base, [%s0, …], [%str0, …], [%off0, …]
///     {order = array<i32: …>} : !tt.ptr<tensor<…>>
/// ```
///
/// # Errors
/// Returns an [`Error`] if the underlying MLIR operation builder fails.
pub fn make_tensor_ptr<'ctx>(
    context: &'ctx Context,
    location: Location<'ctx>,
    base: Value<'ctx, '_>,
    shape: &[Value<'ctx, '_>],
    strides: &[Value<'ctx, '_>],
    offsets: &[Value<'ctx, '_>],
    order: &[i32],
    result_ty: Type<'ctx>,
) -> Result<Operation<'ctx>, Error> {
    debug_assert_eq!(
        shape.len(),
        strides.len(),
        "make_tensor_ptr: shape and strides must have the same length"
    );
    debug_assert_eq!(
        shape.len(),
        offsets.len(),
        "make_tensor_ptr: shape and offsets must have the same length"
    );

    let mut operands: Vec<Value> =
        Vec::with_capacity(1 + shape.len() + strides.len() + offsets.len());
    operands.push(base);
    operands.extend_from_slice(shape);
    operands.extend_from_slice(strides);
    operands.extend_from_slice(offsets);

    let order_attr = DenseI32ArrayAttribute::new(context, order);

    OperationBuilder::new("tt.make_tensor_ptr", location)
        .add_operands(&operands)
        .add_attributes(&[(
            Identifier::new(context, "order"),
            Attribute::from(order_attr),
        )])
        .add_results(&[result_ty])
        .build()
        .map_err(|e| Error::InvalidType {
            msg: format!("failed to build tt.make_tensor_ptr: {e}"),
        })
}

/// Build a `tt.print` operation.
///
/// `tt.print` is a device-side print statement for debugging Triton kernels.
/// It prints the given string prefix followed by the formatted values of any
/// supplied tensor or scalar arguments.
///
/// # Arguments
/// * `prefix`    – string literal prefix printed before the argument values.
/// * `hex`       – when `true`, integer arguments are printed in hexadecimal.
/// * `args`      – zero or more scalar / tensor values to print.
/// * `is_signed` – one `i32` flag per element of `args`; non-zero means the
///                 corresponding argument should be printed as signed.
///
/// # Example
///
/// ```text
/// tt.print "x: " {hex = false, isSigned = array<i32: 1>} : %0 : tensor<8xi32>
/// ```
///
/// Assembly format (from TableGen):
/// ```text
/// tt.print $prefix attr-dict (`:` $args^ `:` type($args))?
/// ```
///
/// # Notes
///
/// The `isSigned` array must have exactly one entry per element of `args`.
/// Passing an empty slice for both `args` and `is_signed` produces a plain
/// prefix-only print with no operands.
pub fn print<'ctx>(
    context: &'ctx Context,
    location: Location<'ctx>,
    prefix: &str,
    hex: bool,
    args: &[Value<'ctx, '_>],
    is_signed: &[i32],
) -> Result<Operation<'ctx>, Error> {
    let prefix_attr = StringAttribute::new(context, prefix);
    let hex_attr = BoolAttribute::new(context, hex);
    let is_signed_attr = DenseI32ArrayAttribute::new(context, is_signed);

    OperationBuilder::new("tt.print", location)
        .add_operands(args)
        .add_attributes(&[
            (Identifier::new(context, "prefix"), Attribute::from(prefix_attr)),
            (Identifier::new(context, "hex"), Attribute::from(hex_attr)),
            (Identifier::new(context, "isSigned"), Attribute::from(is_signed_attr)),
        ])
        .build()
        .map_err(|e| Error::InvalidType { msg: format!("failed to build tt.print: {e}") })
}

/// Build a `tt.make_tensor_descriptor` operation.
///
/// `tt.make_tensor_descriptor` takes meta information of a parent tensor (its
/// base pointer, dynamic shape, and dynamic strides) together with a statically
/// known block shape encoded in the result type, and returns a descriptor
/// object (`!tt.tensordesc<tensor<...>>`) that can be used by
/// `tt.descriptor_load` / `tt.descriptor_store` / `tt.descriptor_gather` /
/// `tt.descriptor_scatter`.
///
/// # Arguments
/// * `base`      – base pointer value (`!tt.ptr<T>`).
/// * `shape`     – one `i32` operand per dimension giving the dynamic extent.
/// * `strides`   – one `i64` operand per dimension giving the dynamic stride.
/// * `padding`   – out-of-bounds padding mode (default: `PaddingOption::PadZero`).
/// * `result_ty` – complete `!tt.tensordesc<tensor<...>>` result type.
///
/// # Panics (debug only)
///
/// Asserts that `shape.len() == strides.len()`, matching the
/// `SameVariadicOperandSize` trait on the Triton op.
///
/// # Assembly format
///
/// ```text
/// tt.make_tensor_descriptor %base, [%s0, ...], [%str0, ...] {padding = ...}
///     : !tt.ptr<T>, !tt.tensordesc<tensor<...>>
/// ```
pub fn make_tensor_descriptor<'ctx>(
    context: &'ctx Context,
    location: Location<'ctx>,
    base: Value<'ctx, '_>,
    shape: &[Value<'ctx, '_>],
    strides: &[Value<'ctx, '_>],
    padding: PaddingOption,
    result_ty: Type<'ctx>,
) -> Result<Operation<'ctx>, Error> {
    debug_assert_eq!(
        shape.len(),
        strides.len(),
        "make_tensor_descriptor: shape and strides must have the same length"
    );

    let mut operands: Vec<Value> = Vec::with_capacity(1 + shape.len() + strides.len());
    operands.push(base);
    operands.extend_from_slice(shape);
    operands.extend_from_slice(strides);

    let padding_attr = attr_i32(context, padding as i32);

    OperationBuilder::new("tt.make_tensor_descriptor", location)
        .add_operands(&operands)
        .add_attributes(&[(
            Identifier::new(context, "padding"),
            Attribute::from(padding_attr),
        )])
        .add_results(&[result_ty])
        .build()
        .map_err(|e| Error::InvalidType {
            msg: format!("failed to build tt.make_tensor_descriptor: {e}"),
        })
}

/// Build a `tt.int_to_ptr` operation.
///
/// Casts an integer value (scalar `i64` or a tensor of `i64`) to a Triton
/// pointer type.  The operation is elementwise: for tensor inputs every lane
/// is individually converted.
///
/// Traits: `Elementwise`, `SameOperandsAndResultShape`,
/// `SameOperandsAndResultEncoding`, `Pure`.
///
/// # Arguments
/// * `src`       – integer operand (`i64` scalar or `tensor<Nxi64>`).
/// * `result_ty` – pointer result type (`!tt.ptr<T>` or `tensor<Nx!tt.ptr<T>>`);
///                 must share the same shape as `src` when both are tensors.
///
/// # Example
///
/// ```text
/// // Scalar cast:
/// %ptr  = tt.int_to_ptr %addr  : i64          -> !tt.ptr<f32>
/// // Tensor cast:
/// %ptrs = tt.int_to_ptr %addrs : tensor<8xi64> -> tensor<8x!tt.ptr<f32>>
/// ```
///
/// Assembly format (from TableGen):
/// ```text
/// $src attr-dict `:` type($src) `->` type($result)
/// ```
pub fn int_to_ptr<'ctx>(
    _context: &'ctx Context,
    location: Location<'ctx>,
    src: Value<'ctx, '_>,
    result_ty: Type<'ctx>,
) -> Result<Operation<'ctx>, Error> {
    OperationBuilder::new("tt.int_to_ptr", location)
        .add_operands(&[src])
        .add_results(&[result_ty])
        .build()
        .map_err(|e| Error::InvalidType { msg: format!("failed to build tt.int_to_ptr: {e}") })
}

/// Build a `tt.gather` operation.
///
/// Gathers elements from `src` using `indices` along a single `axis`.  The
/// output tensor has the same shape as `indices`.  Each dimension of `indices`
/// that is *not* the gather axis must be no greater than the corresponding
/// dimension of `src`.
///
/// The optional `efficient_layout` flag signals that the compiler has already
/// selected an optimised layout for this gather and it should not be changed
/// by subsequent passes.
///
/// # Arguments
/// * `src`              – source tensor to gather from.
/// * `indices`          – integer tensor of indices (same rank as `src`).
/// * `axis`             – axis along which to gather (i32).
/// * `efficient_layout` – when `true`, sets the `efficient_layout` unit attr.
/// * `result_ty`        – result tensor type; same shape as `indices`.
///
/// # Example
///
/// ```text
/// // Gather 8 elements from a 16-element f32 tensor along axis 0:
/// %result = tt.gather %src[%idx] {axis = 0 : i32}
///     : (tensor<16xf32>, tensor<8xi32>) -> tensor<8xf32>
/// ```
///
/// Assembly format (from TableGen):
/// ```text
/// $src `[` $indices `]` attr-dict `:` functional-type(operands, results)
/// ```
pub fn gather<'ctx>(
    context: &'ctx Context,
    location: Location<'ctx>,
    src: Value<'ctx, '_>,
    indices: Value<'ctx, '_>,
    axis: i32,
    efficient_layout: bool,
    result_ty: Type<'ctx>,
) -> Result<Operation<'ctx>, Error> {
    let axis_attr = attr_i32(context, axis);

    let mut attrs: Vec<(Identifier<'_>, Attribute<'_>)> = vec![(
        Identifier::new(context, "axis"),
        Attribute::from(axis_attr),
    )];

    if efficient_layout {
        attrs.push((
            Identifier::new(context, "efficient_layout"),
            Attribute::unit(context),
        ));
    }

    OperationBuilder::new("tt.gather", location)
        .add_operands(&[src, indices])
        .add_attributes(&attrs)
        .add_results(&[result_ty])
        .build()
        .map_err(|e| Error::InvalidType { msg: format!("failed to build tt.gather: {e}") })
}

/// Build a `tt.histogram` operation.
///
/// Returns a histogram of the input integer tensor. The number of bins is
/// equal to the size of the 1-D result tensor; each bin has width 1 and
/// starts at 0.  An optional boolean mask (same shape as `src`) suppresses
/// individual lanes from contributing to the histogram.
///
/// # Arguments
/// * `src`       – source 1-D integer tensor whose elements are histogrammed.
/// * `mask`      – optional boolean mask tensor (same shape as `src`); only
///                 lanes where the mask is `true` contribute to the histogram.
/// * `result_ty` – result integer tensor type (determines the number of bins).
///
/// # Examples
///
/// ```text
/// // Without mask:
/// %hist = tt.histogram %src : tensor<16xi32> -> tensor<16xi32>
///
/// // With mask:
/// %hist = tt.histogram %src, %mask : tensor<16xi32> -> tensor<16xi32>
/// ```
///
/// Assembly format (from TableGen):
/// ```text
/// $src (`,` $mask^)? attr-dict `:` type($src) `->` type($result)
/// ```
pub fn histogram<'ctx>(
    context: &'ctx Context,
    location: Location<'ctx>,
    src: Value<'ctx, '_>,
    mask: Option<Value<'ctx, '_>>,
    result_ty: Type<'ctx>,
) -> Result<Operation<'ctx>, Error> {
    let mask_size = if mask.is_some() { 1i32 } else { 0i32 };

    let mut operands: Vec<Value> = vec![src];
    if let Some(m) = mask {
        operands.push(m);
    }

    // operandSegmentSizes encodes [src=1, mask=0|1] so the verifier knows
    // which optional operand slots are populated.
    let seg_sizes = DenseI32ArrayAttribute::new(context, &[1, mask_size]);

    OperationBuilder::new("tt.histogram", location)
        .add_operands(&operands)
        .add_attributes(&[(
            Identifier::new(context, "operandSegmentSizes"),
            Attribute::from(seg_sizes),
        )])
        .add_results(&[result_ty])
        .build()
        .map_err(|e| Error::InvalidType { msg: format!("failed to build tt.histogram: {e}") })
}

/// Build a `tt.fp_to_fp` operation.
///
/// Performs a floating-point cast between custom or non-standard types.
/// Primarily used for F8 ↔ FP16/BF16/FP32/FP64 conversions, and for
/// non-default rounding modes.
///
/// Traits: `Elementwise`, `SameOperandsAndResultShape`,
/// `SameOperandsAndResultEncoding`, `Pure`.
///
/// # Arguments
/// * `src`       – source floating-point operand (scalar or tensor).
/// * `result_ty` – destination floating-point type; must share the same shape
///                 as `src` when both are tensors.
/// * `rounding`  – optional rounding mode ([`RoundingMode`]).  When `None` the
///                 attribute is omitted and the verifier applies the default
///                 (implementation-defined) rounding for the chosen type pair.
///
/// # Examples
///
/// ```text
/// // Scalar cast with no rounding attribute:
/// %0 = tt.fp_to_fp %arg0 : f32 -> f16
///
/// // Scalar cast with explicit RTZ rounding:
/// %1 = tt.fp_to_fp %arg0, rounding = rtz : f32 -> f8e4m3fnuz
///
/// // Tensor cast:
/// %2 = tt.fp_to_fp %arg0 : tensor<8xf32> -> tensor<8xf16>
/// ```
///
/// Assembly format (from TableGen):
/// ```text
/// $src attr-dict (`,` `rounding` `=` $rounding^)? `:` type($src) `->` type($result)
/// ```
pub fn fp_to_fp<'ctx>(
    context: &'ctx Context,
    location: Location<'ctx>,
    src: Value<'ctx, '_>,
    result_ty: Type<'ctx>,
    rounding: Option<RoundingMode>,
) -> Result<Operation<'ctx>, Error> {
    let mut attrs: Vec<(Identifier<'_>, Attribute<'_>)> = Vec::new();

    if let Some(mode) = rounding {
        // RoundingModeAttr is backed by IntegerAttr<i32>; attr_i32 produces a
        // compatible value that satisfies classof for the RoundingModeAttr check.
        let rounding_attr = attr_i32(context, mode as i32);
        attrs.push((
            Identifier::new(context, "rounding"),
            Attribute::from(rounding_attr),
        ));
    }

    let mut builder = OperationBuilder::new("tt.fp_to_fp", location)
        .add_operands(&[src])
        .add_results(&[result_ty]);

    if !attrs.is_empty() {
        builder = builder.add_attributes(&attrs);
    }

    builder
        .build()
        .map_err(|e| Error::InvalidType { msg: format!("failed to build tt.fp_to_fp: {e}") })
}

/// Build a `tt.extern_elementwise` operation.
///
/// Calls an external elementwise function identified by `symbol` from the
/// shared library `libpath/libname`, passing `srcs` as arguments and
/// returning a value of `result_ty`.
///
/// Traits: `Elementwise`, `SameOperandsAndResultEncoding`,
/// `SameVariadicOperandSize`, `MemoryEffectsOpInterface`,
/// `ConditionallySpeculatable`.
///
/// # Arguments
/// * `srcs`      – variadic source operands (scalar or tensor, any `TT_Type`).
/// * `libname`   – name of the shared library (e.g. `"libdevice"`).
/// * `libpath`   – filesystem path to the library (e.g. `"/usr/local/cuda/nvvm/libdevice"`).
/// * `symbol`    – symbol name inside the library (e.g. `"__nv_sinf"`).
/// * `pure`      – when `true` the call has no side effects and may be
///                 hoisted / CSE'd.
/// * `result_ty` – result element/tensor type.
///
/// # Assembly format
///
/// ```text
/// tt.extern_elementwise %arg0, %arg1 {libname = "libdevice", libpath = "/path", symbol = "__nv_sinf", pure = true}
///     : (f32, f32) -> f32
/// ```
pub fn extern_elementwise<'ctx>(
    context: &'ctx Context,
    location: Location<'ctx>,
    srcs: &[Value<'ctx, '_>],
    libname: &str,
    libpath: &str,
    symbol: &str,
    pure: bool,
    result_ty: Type<'ctx>,
) -> Result<Operation<'ctx>, Error> {
    let libname_attr = StringAttribute::new(context, libname);
    let libpath_attr = StringAttribute::new(context, libpath);
    let symbol_attr = StringAttribute::new(context, symbol);
    let pure_attr = BoolAttribute::new(context, pure);

    OperationBuilder::new("tt.extern_elementwise", location)
        .add_operands(srcs)
        .add_attributes(&[
            (Identifier::new(context, "libname"), Attribute::from(libname_attr)),
            (Identifier::new(context, "libpath"), Attribute::from(libpath_attr)),
            (Identifier::new(context, "symbol"), Attribute::from(symbol_attr)),
            (Identifier::new(context, "pure"), Attribute::from(pure_attr)),
        ])
        .add_results(&[result_ty])
        .build()
        .map_err(|e| Error::InvalidType {
            msg: format!("failed to build tt.extern_elementwise: {e}"),
        })
}

/// Build a `tt.elementwise_inline_asm` operation.
///
/// Runs an inline assembly block to generate one or more tensors, applying an
/// elementwise operation to a group of `packed_element` elements at a time.
/// Exactly which elements the asm block receives is unspecified.
///
/// Traits: `Elementwise`, `SameOperandsAndResultEncoding`,
/// `MemoryEffectsOpInterface`, `ConditionallySpeculatable`.
///
/// # Arguments
/// * `asm_string`     – inline assembly source string (e.g. `"add.f32 $0, $1, $2;"`).
/// * `constraints`    – PTX constraint string (e.g. `"=r,r,r"`).
/// * `pure`           – when `true` the op has no side effects and may be hoisted / CSE'd.
/// * `packed_element` – number of elements processed per asm invocation (i32 ≥ 1).
/// * `args`           – variadic input operands (scalar or tensor, any `TT_Type`).
/// * `result_types`   – result types; may be empty or multiple for multi-output asm.
///
/// # Assembly format
///
/// ```text
/// tt.elementwise_inline_asm "add.f32 $0, $1, $2;" constraints="=r,r,r" {pure = true, packed_element = 1}
///     %arg0, %arg1 : tensor<8xf32>, tensor<8xf32> -> tensor<8xf32>
/// ```
pub fn elementwise_inline_asm<'ctx>(
    context: &'ctx Context,
    location: Location<'ctx>,
    asm_string: &str,
    constraints: &str,
    pure: bool,
    packed_element: i32,
    args: &[Value<'ctx, '_>],
    result_types: &[Type<'ctx>],
) -> Result<Operation<'ctx>, Error> {
    let asm_string_attr = StringAttribute::new(context, asm_string);
    let constraints_attr = StringAttribute::new(context, constraints);
    let pure_attr = BoolAttribute::new(context, pure);
    let packed_element_attr = attr_i32(context, packed_element);

    OperationBuilder::new("tt.elementwise_inline_asm", location)
        .add_operands(args)
        .add_attributes(&[
            (Identifier::new(context, "asm_string"), Attribute::from(asm_string_attr)),
            (Identifier::new(context, "constraints"), Attribute::from(constraints_attr)),
            (Identifier::new(context, "pure"), Attribute::from(pure_attr)),
            (Identifier::new(context, "packed_element"), Attribute::from(packed_element_attr)),
        ])
        .add_results(result_types)
        .build()
        .map_err(|e| Error::InvalidType {
            msg: format!("failed to build tt.elementwise_inline_asm: {e}"),
        })
}

/// Build a `tt.expand_dims` operation.
///
/// Inserts a new dimension of size 1 into the tensor `src` at position `axis`.
/// The element type is preserved.  The result rank is `rank(src) + 1`, with
/// all original dimensions in their original relative order and a new size-1
/// dimension at index `axis`.
///
/// Traits: `Pure`, `InferTypeOpInterface`, `SameOperandsAndResultElementType`.
///
/// # Arguments
/// * `src`       – source tensor to expand.
/// * `axis`      – position at which to insert the new dimension (i32).
/// * `result_ty` – result tensor type (must be `src` type with a size-1 dim
///                 inserted at `axis`).
///
/// # Example
///
/// ```text
/// // Expand a 1-D tensor into 2-D by inserting a size-1 dimension at axis 1:
/// %1 = tt.expand_dims %src {axis = 1 : i32} : tensor<8xf32> -> tensor<8x1xf32>
/// ```
///
/// Assembly format (from TableGen):
/// ```text
/// $src attr-dict `:` type($src) `->` type($result)
/// ```
pub fn expand_dims<'ctx>(
    context: &'ctx Context,
    location: Location<'ctx>,
    src: Value<'ctx, '_>,
    axis: i32,
    result_ty: Type<'ctx>,
) -> Result<Operation<'ctx>, Error> {
    let axis_attr = attr_i32(context, axis);

    OperationBuilder::new("tt.expand_dims", location)
        .add_operands(&[src])
        .add_attributes(&[(Identifier::new(context, "axis"), Attribute::from(axis_attr))])
        .add_results(&[result_ty])
        .build()
        .map_err(|e| Error::InvalidType { msg: format!("failed to build tt.expand_dims: {e}") })
}

/// Build a `tt.broadcast` operation.
///
/// Broadcasts a tensor by expanding dimensions of size 1 to a larger size.
/// For example, `tensor<1x32x1xf32>` can be broadcast to `tensor<2x32x4xf32>`.
/// Non-size-1 dimensions must remain unchanged.
///
/// Traits: `Pure`, `SameOperandsAndResultElementType`, `SameOperandsAndResultEncoding`.
///
/// # Arguments
/// * `src`       – source tensor with one or more size-1 dimensions to broadcast.
/// * `result_ty` – result tensor type (same rank and element type as `src`, with
///                 size-1 dimensions expanded to new sizes).
///
/// # Example
///
/// ```text
/// // Broadcast a tensor<1x32xf32> to tensor<4x32xf32>:
/// %1 = tt.broadcast %src : tensor<1x32xf32> -> tensor<4x32xf32>
/// ```
///
/// Assembly format (from TableGen):
/// ```text
/// $src attr-dict `:` type($src) `->` type($result)
/// ```
pub fn broadcast<'ctx>(
    context: &'ctx Context,
    location: Location<'ctx>,
    src: Value<'ctx, '_>,
    result_ty: Type<'ctx>,
) -> Result<Operation<'ctx>, Error> {
    OperationBuilder::new("tt.broadcast", location)
        .add_operands(&[src])
        .add_results(&[result_ty])
        .build()
        .map_err(|e| Error::InvalidType { msg: format!("failed to build tt.broadcast: {e}") })
}

/// Build a `tt.bitcast` operation.
///
/// Reinterprets the bits of `src` as `result_ty` without any conversion.
/// Both types must have the same bitwidth.  Unlike `arith.bitcast`, this op
/// supports Triton pointer types in addition to ordinary integer and
/// floating-point types.
///
/// Traits: `Elementwise`, `SameOperandsAndResultShape`,
/// `SameOperandsAndResultEncoding`, `Pure`.
///
/// # Arguments
/// * `src`       – source value (scalar or tensor of any Triton-supported type).
/// * `result_ty` – target type; must have the same bitwidth as `src`'s element
///                 type and the same shape when both are tensors.
///
/// # Example
///
/// ```text
/// // Reinterpret f32 bits as i32:
/// %i = tt.bitcast %f : f32 -> i32
/// // Reinterpret a tensor of f32 as tensor of i32:
/// %is = tt.bitcast %fs : tensor<8xf32> -> tensor<8xi32>
/// ```
///
/// Assembly format (from TableGen):
/// ```text
/// $src attr-dict `:` type($src) `->` type($result)
/// ```
pub fn bitcast<'ctx>(
    _context: &'ctx Context,
    location: Location<'ctx>,
    src: Value<'ctx, '_>,
    result_ty: Type<'ctx>,
) -> Result<Operation<'ctx>, Error> {
    OperationBuilder::new("tt.bitcast", location)
        .add_operands(&[src])
        .add_results(&[result_ty])
        .build()
        .map_err(|e| Error::InvalidType { msg: format!("failed to build tt.bitcast: {e}") })
}

/// Build a `tt.clampf` operation.
///
/// Clamps a floating-point scalar or tensor `x` to the closed interval
/// `[min, max]`.  All three operands and the result share the same type
/// (`SameOperandsAndResultType`).
///
/// The `propagate_nan` argument controls NaN handling:
/// - [`PropagateNan::None`] – NaN inputs are treated as in-range values
///   (behaviour is implementation-defined per backend).
/// - [`PropagateNan::All`]  – a NaN in any operand produces a NaN result.
///
/// Traits: `Elementwise`, `SameOperandsAndResultType`, `Pure`.
///
/// # Arguments
/// * `x`             – value to clamp (scalar or tensor float type).
/// * `min`           – lower bound; same type as `x`.
/// * `max`           – upper bound; same type as `x`.
/// * `propagate_nan` – NaN propagation mode.
///
/// # Assembly format
///
/// ```text
/// %result = tt.clampf %x, %min, %max, propagateNan = none : f32
/// %result = tt.clampf %x, %min, %max, propagateNan = all  : tensor<8xf32>
/// ```
pub fn clampf<'ctx>(
    context: &'ctx Context,
    location: Location<'ctx>,
    x: Value<'ctx, '_>,
    min: Value<'ctx, '_>,
    max: Value<'ctx, '_>,
    propagate_nan: PropagateNan,
    result_ty: Type<'ctx>,
) -> Result<Operation<'ctx>, Error> {
    let propagate_nan_attr = attr_i32(context, propagate_nan as i32);

    OperationBuilder::new("tt.clampf", location)
        .add_operands(&[x, min, max])
        .add_attributes(&[(
            Identifier::new(context, "propagateNan"),
            Attribute::from(propagate_nan_attr),
        )])
        .add_results(&[result_ty])
        .build()
        .map_err(|e| Error::InvalidType { msg: format!("failed to build tt.clampf: {e}") })
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

    /// Verify pretty-printed `tt.splat` with exact IR form.
    ///
    /// Uses a function block argument as the scalar source to avoid `arith.constant`
    /// mixing generic and pretty formats.  The expected assembly form is:
    /// `tt.splat %arg0 : i32 -> tensor<8xi32>`
    #[test]
    fn test_splat_pretty_format() {
        let context = create_test_context();
        load_triton_dialect(&context);

        let location = Location::unknown(&context);
        let module = Module::new(location);

        let i32_type: Type = IntegerType::new(&context, 32).into();
        let result_ty: Type = tensor_type(&[8], i32_type).into();

        let func_op = create_func(
            &context,
            location,
            "test_splat_pretty",
            "public",
            &[i32_type],
            &[result_ty],
            0,
        )
        .unwrap();

        let block = Block::new(&[(i32_type, location)]);
        let src: Value = block.argument(0).unwrap().into();

        let splat_op: Operation<'_> = splat(&context, location, src, result_ty).unwrap().into();
        let ret_val: Value = splat_op.result(0).unwrap().into();
        let ret_op = ReturnOperation::builder(&context, location).srcs(&[ret_val]).build();

        block.append_operation(splat_op);
        block.append_operation(ret_op.into());
        func_op.body().unwrap().append_block(block);
        module.body().append_operation(func_op.into());

        let output = module.as_operation().to_string();

        // Pretty-printed form: "tt.splat %arg0 : i32 -> tensor<8xi32>"
        assert!(output.contains("tt.splat"), "missing op mnemonic:\n{output}");
        assert!(output.contains("i32"), "missing src type:\n{output}");
        assert!(output.contains("tensor<8xi32>"), "missing result type:\n{output}");
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

    /// Verify that `descriptor_scatter` emits the correct `tt.descriptor_scatter` IR.
    ///
    /// Uses function block-arguments to supply typed values without needing
    /// constant-folding or a real TMA descriptor at compile time.
    #[test]
    fn test_descriptor_scatter() {
        let context = create_test_context();
        load_triton_dialect(&context);

        let location = Location::unknown(&context);
        let module = Module::new(location);

        // Source tensor type: tensor<16x16xf32> (16 rows to scatter, 16 cols each).
        let f32_type = melior::ir::Type::float32(&context);
        let src_tensor_ty: Type = tensor_type(&[16, 16], f32_type).into();

        // Descriptor type: !tt.tensordesc<tensor<1x16xf32>>.
        // The block has 1 row and 16 columns – the scatter restriction.
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
            "test_descriptor_scatter",
            "public",
            &[desc_ty, x_offsets_ty, y_offset_ty, src_tensor_ty],
            &[],
            0,
        )
        .unwrap();

        let block = Block::new(&[
            (desc_ty, location),
            (x_offsets_ty, location),
            (y_offset_ty, location),
            (src_tensor_ty, location),
        ]);

        let desc: Value = block.argument(0).unwrap().into();
        let x_offsets: Value = block.argument(1).unwrap().into();
        let y_offset: Value = block.argument(2).unwrap().into();
        let src: Value = block.argument(3).unwrap().into();

        let scatter_op: Operation<'_> =
            descriptor_scatter(&context, location, desc, x_offsets, y_offset, src)
                .unwrap()
                .into();

        // tt.descriptor_scatter has no results; emit a void return.
        let ret_op = ReturnOperation::builder(&context, location).srcs(&[]).build();

        block.append_operation(scatter_op);
        block.append_operation(ret_op.into());
        func_op.body().unwrap().append_block(block);
        module.body().append_operation(func_op.into());

        let output = module.as_operation().to_string();

        assert!(
            output.contains("tt.descriptor_scatter"),
            "missing op mnemonic:\n{output}"
        );
        assert!(
            output.contains("tensordesc"),
            "missing tensordesc operand type:\n{output}"
        );
        assert!(
            output.contains("tensor<16x16xf32>"),
            "missing src tensor type:\n{output}"
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

    /// Verify that `tensor::int_to_ptr` emits the correct `tt.int_to_ptr` IR
    /// for a scalar `i64 -> !tt.ptr<f32>` cast inside a `tt.func`.
    ///
    /// Expected assembly:
    /// ```text
    /// %0 = tt.int_to_ptr %arg0 : i64 -> !tt.ptr<f32>
    /// ```
    #[test]
    fn test_int_to_ptr() {
        let context = create_test_context();
        load_triton_dialect(&context);

        let location = Location::unknown(&context);
        let module = Module::new(location);

        let f32_type = melior::ir::Type::float32(&context);
        let ptr_f32_type = pointer_type(f32_type);
        let i64_type: Type = IntegerType::new(&context, 64).into();

        // Function: (i64) -> !tt.ptr<f32>
        let func_op = create_func(
            &context,
            location,
            "test_int_to_ptr_fn",
            "public",
            &[i64_type],
            &[ptr_f32_type],
            0,
        )
        .unwrap();

        let block = Block::new(&[(i64_type, location)]);
        let src: melior::ir::Value = block.argument(0).unwrap().into();

        // Use the tensor-module int_to_ptr (super::int_to_ptr, not crate::triton::int_to_ptr).
        let cast_op: Operation<'_> =
            super::int_to_ptr(&context, location, src, ptr_f32_type).unwrap();
        let ptr_val: melior::ir::Value = cast_op.result(0).unwrap().into();
        let ret_op = ReturnOperation::builder(&context, location).srcs(&[ptr_val]).build();

        block.append_operation(cast_op);
        block.append_operation(ret_op.into());
        func_op.body().unwrap().append_block(block);
        module.body().append_operation(func_op.into());

        let output = module.as_operation().to_string();

        // The op mnemonic must appear in the emitted IR.
        assert!(output.contains("tt.int_to_ptr"), "missing op mnemonic:\n{output}");
        // The source and result types must be visible in the assembly.
        assert!(
            output.contains("i64 -> !tt.ptr<f32>"),
            "wrong type signature in IR:\n{output}"
        );
    }

    /// Verify that `tensor::int_to_ptr` works for a tensor operand:
    /// `tensor<8xi64> -> tensor<8x!tt.ptr<f32>>`.
    #[test]
    fn test_int_to_ptr_tensor() {
        let context = create_test_context();
        load_triton_dialect(&context);

        let location = Location::unknown(&context);
        let module = Module::new(location);

        let f32_type = melior::ir::Type::float32(&context);
        let ptr_f32_type = pointer_type(f32_type);
        let i64_type: Type = IntegerType::new(&context, 64).into();

        let src_ty: Type = tensor_type(&[8], i64_type).into();
        let res_ty: Type = tensor_type(&[8], ptr_f32_type).into();

        let func_op = create_func(
            &context,
            location,
            "test_int_to_ptr_tensor_fn",
            "public",
            &[src_ty],
            &[res_ty],
            0,
        )
        .unwrap();

        let block = Block::new(&[(src_ty, location)]);
        let src: melior::ir::Value = block.argument(0).unwrap().into();

        let cast_op: Operation<'_> =
            super::int_to_ptr(&context, location, src, res_ty).unwrap();
        let result_val: melior::ir::Value = cast_op.result(0).unwrap().into();
        let ret_op = ReturnOperation::builder(&context, location).srcs(&[result_val]).build();

        block.append_operation(cast_op);
        block.append_operation(ret_op.into());
        func_op.body().unwrap().append_block(block);
        module.body().append_operation(func_op.into());

        let output = module.as_operation().to_string();

        assert!(output.contains("tt.int_to_ptr"), "missing op mnemonic:\n{output}");
        assert!(
            output.contains("tensor<8xi64>") && output.contains("tensor<8x!tt.ptr<f32>>"),
            "wrong tensor types in IR:\n{output}"
        );
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

    /// Verify that `join` emits the correct `tt.join` op.
    ///
    /// Two `tensor<4x8xf32>` inputs → `tensor<4x8x2xf32>`.
    #[test]
    fn test_join() {
        let context = create_test_context();
        load_triton_dialect(&context);

        let location = Location::unknown(&context);
        let module = Module::new(location);

        let f32_type = melior::ir::Type::float32(&context);
        // Inputs: tensor<4x8xf32>
        let in_ty: Type = tensor_type(&[4, 8], f32_type).into();
        // Result: tensor<4x8x2xf32>
        let out_ty: Type = tensor_type(&[4, 8, 2], f32_type).into();

        let func_op = create_func(
            &context,
            location,
            "test_join",
            "public",
            &[in_ty, in_ty],
            &[out_ty],
            0,
        )
        .unwrap();

        let block = Block::new(&[(in_ty, location), (in_ty, location)]);
        let lhs: Value = block.argument(0).unwrap().into();
        let rhs: Value = block.argument(1).unwrap().into();

        let join_op: Operation<'_> = join(&context, location, lhs, rhs, out_ty).unwrap();

        let result: Value = join_op.result(0).unwrap().into();
        let ret_op = ReturnOperation::builder(&context, location).srcs(&[result]).build();

        block.append_operation(join_op);
        block.append_operation(ret_op.into());
        func_op.body().unwrap().append_block(block);
        module.body().append_operation(func_op.into());

        let output = module.as_operation().to_string();

        assert!(output.contains("tt.join"), "missing op mnemonic:\n{output}");
        assert!(
            output.contains("tensor<4x8xf32>"),
            "missing input tensor type:\n{output}"
        );
        assert!(
            output.contains("tensor<4x8x2xf32>"),
            "missing result tensor type:\n{output}"
        );
    }

    /// Verify pretty-printed `tt.join` with exact IR form.
    #[test]
    fn test_join_pretty_format() {
        let context = create_test_context();
        load_triton_dialect(&context);

        let location = Location::unknown(&context);
        let module = Module::new(location);

        let f32_type = melior::ir::Type::float32(&context);
        let in_ty: Type = tensor_type(&[4, 8], f32_type).into();
        let out_ty: Type = tensor_type(&[4, 8, 2], f32_type).into();

        let func_op = create_func(
            &context,
            location,
            "test_join_pretty",
            "public",
            &[in_ty, in_ty],
            &[out_ty],
            0,
        )
        .unwrap();

        let block = Block::new(&[(in_ty, location), (in_ty, location)]);
        let lhs: Value = block.argument(0).unwrap().into();
        let rhs: Value = block.argument(1).unwrap().into();

        let join_op: Operation<'_> = join(&context, location, lhs, rhs, out_ty).unwrap();

        let result: Value = join_op.result(0).unwrap().into();
        let ret_op = ReturnOperation::builder(&context, location).srcs(&[result]).build();

        block.append_operation(join_op);
        block.append_operation(ret_op.into());
        func_op.body().unwrap().append_block(block);
        module.body().append_operation(func_op.into());

        let output = module.as_operation().to_string();

        // Pretty-printed form: "tt.join %arg0, %arg1 : tensor<4x8xf32> -> tensor<4x8x2xf32>"
        assert!(
            output.contains("tt.join"),
            "missing tt.join mnemonic:\n{output}"
        );
        assert!(
            output.contains("tensor<4x8xf32>") && output.contains("tensor<4x8x2xf32>"),
            "missing type annotations in tt.join output:\n{output}"
        );
    }

    /// Verify that `reshape` emits the correct `tt.reshape` op without flags.
    ///
    /// Input `tensor<8xf32>` → `tensor<2x4xf32>` with no optional attributes.
    #[test]
    fn test_reshape_no_flags() {
        let context = create_test_context();
        load_triton_dialect(&context);

        let location = Location::unknown(&context);
        let module = Module::new(location);

        let f32_type = melior::ir::Type::float32(&context);
        let src_ty: Type = tensor_type(&[8], f32_type).into();
        let dst_ty: Type = tensor_type(&[2, 4], f32_type).into();

        let func_op = create_func(
            &context,
            location,
            "test_reshape_no_flags",
            "public",
            &[src_ty],
            &[dst_ty],
            0,
        )
        .unwrap();

        let block = Block::new(&[(src_ty, location)]);
        let src: Value = block.argument(0).unwrap().into();

        let reshape_op: Operation<'_> =
            reshape(&context, location, src, dst_ty, false, false).unwrap();

        let result: Value = reshape_op.result(0).unwrap().into();
        let ret_op = ReturnOperation::builder(&context, location).srcs(&[result]).build();

        block.append_operation(reshape_op);
        block.append_operation(ret_op.into());
        func_op.body().unwrap().append_block(block);
        module.body().append_operation(func_op.into());

        let output = module.as_operation().to_string();

        assert!(output.contains("tt.reshape"), "missing op mnemonic:\n{output}");
        assert!(output.contains("tensor<8xf32>"), "missing src type:\n{output}");
        assert!(output.contains("tensor<2x4xf32>"), "missing result type:\n{output}");
        // No optional attributes should appear.
        assert!(!output.contains("allow_reorder"), "unexpected allow_reorder flag:\n{output}");
        assert!(!output.contains("efficient_layout"), "unexpected efficient_layout flag:\n{output}");
    }

    /// Verify that `reshape` with `allow_reorder = true` emits the flag.
    #[test]
    fn test_reshape_allow_reorder() {
        let context = create_test_context();
        load_triton_dialect(&context);

        let location = Location::unknown(&context);
        let module = Module::new(location);

        let f32_type = melior::ir::Type::float32(&context);
        let src_ty: Type = tensor_type(&[8], f32_type).into();
        let dst_ty: Type = tensor_type(&[2, 4], f32_type).into();

        let func_op = create_func(
            &context,
            location,
            "test_reshape_allow_reorder",
            "public",
            &[src_ty],
            &[dst_ty],
            0,
        )
        .unwrap();

        let block = Block::new(&[(src_ty, location)]);
        let src: Value = block.argument(0).unwrap().into();

        let reshape_op: Operation<'_> =
            reshape(&context, location, src, dst_ty, true, false).unwrap();

        let result: Value = reshape_op.result(0).unwrap().into();
        let ret_op = ReturnOperation::builder(&context, location).srcs(&[result]).build();

        block.append_operation(reshape_op);
        block.append_operation(ret_op.into());
        func_op.body().unwrap().append_block(block);
        module.body().append_operation(func_op.into());

        let output = module.as_operation().to_string();

        assert!(output.contains("tt.reshape"), "missing op mnemonic:\n{output}");
        assert!(output.contains("allow_reorder"), "missing allow_reorder flag:\n{output}");
        assert!(!output.contains("efficient_layout"), "unexpected efficient_layout flag:\n{output}");
    }

    /// Verify that `reshape` with both flags emits both attributes.
    #[test]
    fn test_reshape_both_flags() {
        let context = create_test_context();
        load_triton_dialect(&context);

        let location = Location::unknown(&context);
        let module = Module::new(location);

        let f32_type = melior::ir::Type::float32(&context);
        let src_ty: Type = tensor_type(&[8], f32_type).into();
        let dst_ty: Type = tensor_type(&[2, 4], f32_type).into();

        let func_op = create_func(
            &context,
            location,
            "test_reshape_both_flags",
            "public",
            &[src_ty],
            &[dst_ty],
            0,
        )
        .unwrap();

        let block = Block::new(&[(src_ty, location)]);
        let src: Value = block.argument(0).unwrap().into();

        let reshape_op: Operation<'_> =
            reshape(&context, location, src, dst_ty, true, true).unwrap();

        let result: Value = reshape_op.result(0).unwrap().into();
        let ret_op = ReturnOperation::builder(&context, location).srcs(&[result]).build();

        block.append_operation(reshape_op);
        block.append_operation(ret_op.into());
        func_op.body().unwrap().append_block(block);
        module.body().append_operation(func_op.into());

        let output = module.as_operation().to_string();

        assert!(output.contains("tt.reshape"), "missing op mnemonic:\n{output}");
        assert!(output.contains("allow_reorder"), "missing allow_reorder flag:\n{output}");
        assert!(output.contains("efficient_layout"), "missing efficient_layout flag:\n{output}");
    }

    /// Verify that `scan_return` emits `tt.scan.return` with the correct
    /// operands and type annotation.
    ///
    /// The test builds a minimal `tt.scan` over a `tensor<4xf32>` with
    /// axis=0 and reverse=false.  The `combineOp` region holds one block
    /// with two `f32` block arguments (lhs, rhs); `tt.scan.return` yields
    /// the lhs value back to the scan loop.
    ///
    /// Expected pretty-printed form inside the region:
    /// ```text
    /// tt.scan.return %arg0 : f32
    /// ```
    #[test]
    fn test_scan_return() {
        use melior::ir::attribute::BoolAttribute;
        use melior::ir::operation::OperationMutLike;
        use crate::triton::tt::{ScanOperation, ScanReturnOperation};

        let context = create_test_context();
        load_triton_dialect(&context);

        let location = Location::unknown(&context);
        let module = Module::new(location);

        let f32_type = melior::ir::Type::float32(&context);
        // Source tensor: tensor<4xf32>
        let src_ty: Type = tensor_type(&[4], f32_type).into();

        // Wrap in a tt.func so block-arguments are valid SSA values.
        let func_op = create_func(
            &context,
            location,
            "test_scan_return",
            "public",
            &[src_ty],
            &[src_ty],
            0,
        )
        .unwrap();

        // --- Build the combineOp region ---
        // The region has a single block with two f32 arguments (lhs, rhs).
        let combine_block = Block::new(&[(f32_type, location), (f32_type, location)]);
        let lhs: Value = combine_block.argument(0).unwrap().into();

        // tt.scan.return %lhs : f32
        let scan_ret_op: Operation<'_> =
            scan_return(&context, location, &[lhs]).unwrap().into();
        combine_block.append_operation(scan_ret_op);

        let combine_region = melior::ir::Region::new();
        combine_region.append_block(combine_block);

        // --- Build the tt.scan op ---
        let func_block = Block::new(&[(src_ty, location)]);
        let src: Value = func_block.argument(0).unwrap().into();

        let axis_attr = attr_i32(&context, 0);
        let reverse_attr: melior::ir::Attribute =
            BoolAttribute::new(&context, false).into();

        let scan_op: Operation<'_> = ScanOperation::builder(&context, location)
            .result(&[src_ty])
            .srcs(&[src])
            .combine_op(combine_region)
            .axis(axis_attr)
            .reverse(reverse_attr)
            .build()
            .into();

        let scan_result: Value = scan_op.result(0).unwrap().into();
        let ret_op = ReturnOperation::builder(&context, location)
            .srcs(&[scan_result])
            .build();

        func_block.append_operation(scan_op);
        func_block.append_operation(ret_op.into());
        func_op.body().unwrap().append_block(func_block);
        module.body().append_operation(func_op.into());

        let output = module.as_operation().to_string();

        assert!(
            output.contains("tt.scan"),
            "missing tt.scan mnemonic:\n{output}"
        );
        assert!(
            output.contains("tt.scan.return"),
            "missing tt.scan.return terminator:\n{output}"
        );
        assert!(
            output.contains("f32"),
            "missing f32 type annotation:\n{output}"
        );
    }

    /// Verify pretty-printed `tt.scan.return` with exact IR form.
    ///
    /// Checks that the emitted IR contains the canonical assembly-format
    /// string for `tt.scan.return`:
    /// ```text
    /// tt.scan.return %arg0 : f32
    /// ```
    #[test]
    fn test_scan_return_pretty_format() {
        use melior::ir::attribute::BoolAttribute;
        use crate::triton::tt::{ScanOperation, ScanReturnOperation};

        let context = create_test_context();
        load_triton_dialect(&context);

        let location = Location::unknown(&context);
        let module = Module::new(location);

        let f32_type = melior::ir::Type::float32(&context);
        let src_ty: Type = tensor_type(&[4], f32_type).into();

        let func_op = create_func(
            &context,
            location,
            "test_scan_return_pretty",
            "public",
            &[src_ty],
            &[src_ty],
            0,
        )
        .unwrap();

        // combineOp region: block(%arg0: f32, %arg1: f32) { tt.scan.return %arg0 : f32 }
        let combine_block = Block::new(&[(f32_type, location), (f32_type, location)]);
        let lhs: Value = combine_block.argument(0).unwrap().into();
        let scan_ret_op: Operation<'_> =
            scan_return(&context, location, &[lhs]).unwrap().into();
        combine_block.append_operation(scan_ret_op);

        let combine_region = melior::ir::Region::new();
        combine_region.append_block(combine_block);

        let func_block = Block::new(&[(src_ty, location)]);
        let src: Value = func_block.argument(0).unwrap().into();

        let axis_attr = attr_i32(&context, 0);
        let reverse_attr: melior::ir::Attribute =
            BoolAttribute::new(&context, false).into();

        let scan_op: Operation<'_> = ScanOperation::builder(&context, location)
            .result(&[src_ty])
            .srcs(&[src])
            .combine_op(combine_region)
            .axis(axis_attr)
            .reverse(reverse_attr)
            .build()
            .into();

        let scan_result: Value = scan_op.result(0).unwrap().into();
        let ret_op = ReturnOperation::builder(&context, location)
            .srcs(&[scan_result])
            .build();

        func_block.append_operation(scan_op);
        func_block.append_operation(ret_op.into());
        func_op.body().unwrap().append_block(func_block);
        module.body().append_operation(func_op.into());

        let output = module.as_operation().to_string();

        // Pretty-printed form: "tt.scan.return %arg0 : f32"
        assert!(
            output.contains("tt.scan.return"),
            "missing tt.scan.return mnemonic:\n{output}"
        );
        assert!(
            output.contains(": f32"),
            "missing f32 type annotation in tt.scan.return:\n{output}"
        );
        assert!(
            output.contains("axis = 0") || output.contains("axis = 0 :"),
            "missing axis attribute:\n{output}"
        );
    }

    /// Verify that `scan` emits a `tt.scan` op with the correct operands,
    /// result type, region, and attributes.
    ///
    /// The test builds a `tt.scan` over a `tensor<4xf32>` with axis=0 and
    /// reverse=false.  The combineOp region simply returns its first argument.
    ///
    /// Expected pretty-printed form (simplified):
    /// ```text
    /// tt.scan %arg0 {axis = 0 : i32, reverse = false} (
    ///   ^bb0(%arg0: f32, %arg1: f32):
    ///     tt.scan.return %arg0 : f32
    /// ) : (tensor<4xf32>) -> tensor<4xf32>
    /// ```
    #[test]
    fn test_scan() {
        let context = create_test_context();
        load_triton_dialect(&context);

        let location = Location::unknown(&context);
        let module = Module::new(location);

        let f32_type = melior::ir::Type::float32(&context);
        let src_ty: Type = tensor_type(&[4], f32_type).into();

        let func_op = create_func(
            &context,
            location,
            "test_scan",
            "public",
            &[src_ty],
            &[src_ty],
            0,
        )
        .unwrap();

        // combineOp region: two f32 block args, returns lhs via tt.scan.return
        let combine_block = Block::new(&[(f32_type, location), (f32_type, location)]);
        let lhs: Value = combine_block.argument(0).unwrap().into();
        let scan_ret_op: Operation<'_> =
            scan_return(&context, location, &[lhs]).unwrap().into();
        combine_block.append_operation(scan_ret_op);

        let combine_region = melior::ir::Region::new();
        combine_region.append_block(combine_block);

        let func_block = Block::new(&[(src_ty, location)]);
        let src: Value = func_block.argument(0).unwrap().into();

        let scan_op: Operation<'_> =
            scan(&context, location, &[src], &[src_ty], 0, false, combine_region)
                .unwrap()
                .into();

        let scan_result: Value = scan_op.result(0).unwrap().into();
        let ret_op = crate::triton::tt::ReturnOperation::builder(&context, location)
            .srcs(&[scan_result])
            .build();

        func_block.append_operation(scan_op);
        func_block.append_operation(ret_op.into());
        func_op.body().unwrap().append_block(func_block);
        module.body().append_operation(func_op.into());

        let output = module.as_operation().to_string();

        assert!(
            output.contains("tt.scan"),
            "missing tt.scan mnemonic:\n{output}"
        );
        assert!(
            output.contains("tt.scan.return"),
            "missing tt.scan.return terminator:\n{output}"
        );
        assert!(
            output.contains("f32"),
            "missing f32 type annotation:\n{output}"
        );
    }

    /// Verify pretty-printed `tt.scan` with exact IR form.
    ///
    /// Checks that the emitted IR contains the canonical pretty-print strings
    /// for a `tt.scan` op with axis=0, reverse=false.
    #[test]
    fn test_scan_pretty_format() {
        let context = create_test_context();
        load_triton_dialect(&context);

        let location = Location::unknown(&context);
        let module = Module::new(location);

        let f32_type = melior::ir::Type::float32(&context);
        let src_ty: Type = tensor_type(&[4], f32_type).into();

        let func_op = create_func(
            &context,
            location,
            "test_scan_pretty",
            "public",
            &[src_ty],
            &[src_ty],
            0,
        )
        .unwrap();

        let combine_block = Block::new(&[(f32_type, location), (f32_type, location)]);
        let lhs: Value = combine_block.argument(0).unwrap().into();
        let scan_ret_op: Operation<'_> =
            scan_return(&context, location, &[lhs]).unwrap().into();
        combine_block.append_operation(scan_ret_op);

        let combine_region = melior::ir::Region::new();
        combine_region.append_block(combine_block);

        let func_block = Block::new(&[(src_ty, location)]);
        let src: Value = func_block.argument(0).unwrap().into();

        let scan_op: Operation<'_> =
            scan(&context, location, &[src], &[src_ty], 0, false, combine_region)
                .unwrap()
                .into();

        let scan_result: Value = scan_op.result(0).unwrap().into();
        let ret_op = crate::triton::tt::ReturnOperation::builder(&context, location)
            .srcs(&[scan_result])
            .build();

        func_block.append_operation(scan_op);
        func_block.append_operation(ret_op.into());
        func_op.body().unwrap().append_block(func_block);
        module.body().append_operation(func_op.into());

        let output = module.as_operation().to_string();

        assert!(
            output.contains("tt.scan"),
            "missing tt.scan mnemonic:\n{output}"
        );
        assert!(
            output.contains("axis = 0") || output.contains("axis = 0 :"),
            "missing axis attribute:\n{output}"
        );
        assert!(
            output.contains("reverse = false"),
            "missing reverse attribute:\n{output}"
        );
        assert!(
            output.contains("tensor<4xf32>"),
            "missing result type annotation:\n{output}"
        );
    }

    /// Verify `tt.return` with no operands (void function).
    #[test]
    fn test_return_op_no_values() {
        let context = create_test_context();
        load_triton_dialect(&context);

        let location = Location::unknown(&context);
        let module = Module::new(location);

        // tt.func @test_return_void() -> ()
        let func_op =
            create_func(&context, location, "test_return_void", "public", &[], &[], 0).unwrap();

        let func_block = Block::new(&[]);
        let ret_op: Operation<'_> = return_op(&context, location, &[]).unwrap().into();
        func_block.append_operation(ret_op);
        func_op.body().unwrap().append_block(func_block);
        module.body().append_operation(func_op.into());

        let output = module.as_operation().to_string();

        assert!(output.contains("tt.return"), "missing tt.return:\n{output}");
        // Void return: no type annotation expected after the mnemonic
        assert!(
            !output.contains("tt.return %"),
            "unexpected operands in void return:\n{output}"
        );
    }

    /// Verify `tt.return` with a single f32 operand.
    #[test]
    fn test_return_op_with_value() {
        use melior::dialect::ods::arith as ods_arith;
        use melior::ir::attribute::Attribute;

        let context = create_test_context();
        load_triton_dialect(&context);

        let location = Location::unknown(&context);
        let module = Module::new(location);

        let f32_type = melior::ir::Type::float32(&context);

        // tt.func @test_return_f32() -> f32
        let func_op =
            create_func(&context, location, "test_return_f32", "public", &[], &[f32_type], 0)
                .unwrap();

        let one_attr = Attribute::parse(&context, "1.0 : f32").expect("valid f32 literal");
        let const_op: Operation<'_> = ods_arith::ConstantOperation::builder(&context, location)
            .value(one_attr)
            .result(f32_type)
            .build()
            .into();
        let const_val: melior::ir::Value = const_op.result(0).unwrap().into();
        let ret_op: Operation<'_> = return_op(&context, location, &[const_val]).unwrap().into();

        let func_block = Block::new(&[]);
        func_block.append_operation(const_op);
        func_block.append_operation(ret_op);
        func_op.body().unwrap().append_block(func_block);
        module.body().append_operation(func_op.into());

        let output = module.as_operation().to_string();

        assert!(output.contains("tt.return"), "missing tt.return:\n{output}");
        assert!(output.contains("f32"), "missing f32 type annotation:\n{output}");
        // The return value should reference the constant result
        assert!(output.contains("tt.return %"), "missing operand in tt.return:\n{output}");
    }

    /// Verify that `reduce_return` emits `tt.reduce.return` with the correct
    /// operands and type annotation.
    ///
    /// The test builds a minimal `tt.reduce` over a `tensor<4xf32>` with
    /// axis=0.  The `combineOp` region holds one block with two `f32` block
    /// arguments (lhs, rhs); `tt.reduce.return` yields the lhs value back to
    /// the reduce loop.
    ///
    /// Expected pretty-printed form inside the region:
    /// ```text
    /// tt.reduce.return %arg0 : f32
    /// ```
    #[test]
    fn test_reduce_return() {
        use crate::triton::tt::{ReduceOperation, ReduceReturnOperation};

        let context = create_test_context();
        load_triton_dialect(&context);

        let location = Location::unknown(&context);
        let module = Module::new(location);

        let f32_type = melior::ir::Type::float32(&context);
        // Source tensor: tensor<4xf32>
        let src_ty: Type = tensor_type(&[4], f32_type).into();

        // Wrap in a tt.func so block-arguments are valid SSA values.
        let func_op = create_func(
            &context,
            location,
            "test_reduce_return",
            "public",
            &[src_ty],
            &[f32_type],
            0,
        )
        .unwrap();

        // --- Build the combineOp region ---
        // The region has a single block with two f32 arguments (lhs, rhs).
        let combine_block = Block::new(&[(f32_type, location), (f32_type, location)]);
        let lhs: Value = combine_block.argument(0).unwrap().into();

        // tt.reduce.return %lhs : f32
        let reduce_ret_op: Operation<'_> =
            reduce_return(&context, location, &[lhs]).unwrap().into();
        combine_block.append_operation(reduce_ret_op);

        let combine_region = melior::ir::Region::new();
        combine_region.append_block(combine_block);

        // --- Build the tt.reduce op ---
        let func_block = Block::new(&[(src_ty, location)]);
        let src: Value = func_block.argument(0).unwrap().into();

        let axis_attr = attr_i32(&context, 0);

        let reduce_op: Operation<'_> = ReduceOperation::builder(&context, location)
            .result(&[f32_type])
            .srcs(&[src])
            .combine_op(combine_region)
            .axis(axis_attr)
            .build()
            .into();

        let reduce_result: Value = reduce_op.result(0).unwrap().into();
        let ret_op = ReturnOperation::builder(&context, location)
            .srcs(&[reduce_result])
            .build();

        func_block.append_operation(reduce_op);
        func_block.append_operation(ret_op.into());
        func_op.body().unwrap().append_block(func_block);
        module.body().append_operation(func_op.into());

        let output = module.as_operation().to_string();

        assert!(
            output.contains("tt.reduce"),
            "missing tt.reduce mnemonic:\n{output}"
        );
        assert!(
            output.contains("tt.reduce.return"),
            "missing tt.reduce.return terminator:\n{output}"
        );
        assert!(
            output.contains("f32"),
            "missing f32 type annotation:\n{output}"
        );
    }

    /// Verify pretty-printed `tt.reduce.return` with exact IR form.
    ///
    /// Checks that the emitted IR contains the canonical assembly-format
    /// string for `tt.reduce.return`:
    /// ```text
    /// tt.reduce.return %arg0 : f32
    /// ```
    #[test]
    fn test_reduce_return_pretty_format() {
        use crate::triton::tt::{ReduceOperation, ReduceReturnOperation};

        let context = create_test_context();
        load_triton_dialect(&context);

        let location = Location::unknown(&context);
        let module = Module::new(location);

        let f32_type = melior::ir::Type::float32(&context);
        let src_ty: Type = tensor_type(&[4], f32_type).into();

        let func_op = create_func(
            &context,
            location,
            "test_reduce_return_pretty",
            "public",
            &[src_ty],
            &[f32_type],
            0,
        )
        .unwrap();

        // combineOp region: block(%arg0: f32, %arg1: f32) { tt.reduce.return %arg0 : f32 }
        let combine_block = Block::new(&[(f32_type, location), (f32_type, location)]);
        let lhs: Value = combine_block.argument(0).unwrap().into();
        let reduce_ret_op: Operation<'_> =
            reduce_return(&context, location, &[lhs]).unwrap().into();
        combine_block.append_operation(reduce_ret_op);

        let combine_region = melior::ir::Region::new();
        combine_region.append_block(combine_block);

        let func_block = Block::new(&[(src_ty, location)]);
        let src: Value = func_block.argument(0).unwrap().into();

        let axis_attr = attr_i32(&context, 0);

        let reduce_op: Operation<'_> = ReduceOperation::builder(&context, location)
            .result(&[f32_type])
            .srcs(&[src])
            .combine_op(combine_region)
            .axis(axis_attr)
            .build()
            .into();

        let reduce_result: Value = reduce_op.result(0).unwrap().into();
        let ret_op = ReturnOperation::builder(&context, location)
            .srcs(&[reduce_result])
            .build();

        func_block.append_operation(reduce_op);
        func_block.append_operation(ret_op.into());
        func_op.body().unwrap().append_block(func_block);
        module.body().append_operation(func_op.into());

        let output = module.as_operation().to_string();

        // Pretty-printed form: "tt.reduce.return %arg0 : f32"
        assert!(
            output.contains("tt.reduce.return"),
            "missing tt.reduce.return mnemonic:\n{output}"
        );
        assert!(
            output.contains(": f32"),
            "missing f32 type annotation in tt.reduce.return:\n{output}"
        );
        assert!(
            output.contains("axis = 0") || output.contains("axis = 0 :"),
            "missing axis attribute:\n{output}"
        );
    }

    /// Verify that `reduce` emits a valid `tt.reduce` op with a `combineOp` region.
    ///
    /// Reduces `tensor<4xf32>` to `f32` along axis 0.  The `combineOp` region
    /// passes through the lhs argument via `tt.reduce.return`.
    ///
    /// Expected IR contains:
    /// ```text
    /// tt.reduce(%arg0) {axis = 0 : i32} (
    ///   ^bb0(%arg0: f32, %arg1: f32):
    ///     tt.reduce.return %arg0 : f32
    /// ) : (tensor<4xf32>) -> f32
    /// ```
    #[test]
    fn test_reduce() {
        let context = create_test_context();
        load_triton_dialect(&context);

        let location = Location::unknown(&context);
        let module = Module::new(location);

        let f32_type = melior::ir::Type::float32(&context);
        let src_ty: Type = tensor_type(&[4], f32_type).into();

        let func_op = create_func(
            &context,
            location,
            "test_reduce",
            "public",
            &[src_ty],
            &[f32_type],
            0,
        )
        .unwrap();

        // combineOp region: two f32 block args, returns lhs via tt.reduce.return
        let combine_block = Block::new(&[(f32_type, location), (f32_type, location)]);
        let lhs: Value = combine_block.argument(0).unwrap().into();
        let reduce_ret_op: Operation<'_> =
            reduce_return(&context, location, &[lhs]).unwrap().into();
        combine_block.append_operation(reduce_ret_op);

        let combine_region = melior::ir::Region::new();
        combine_region.append_block(combine_block);

        let func_block = Block::new(&[(src_ty, location)]);
        let src: Value = func_block.argument(0).unwrap().into();

        let reduce_op: Operation<'_> =
            reduce(&context, location, &[src], &[f32_type], 0, combine_region)
                .unwrap()
                .into();

        let reduce_result: Value = reduce_op.result(0).unwrap().into();
        let ret_op = crate::triton::tt::ReturnOperation::builder(&context, location)
            .srcs(&[reduce_result])
            .build();

        func_block.append_operation(reduce_op);
        func_block.append_operation(ret_op.into());
        func_op.body().unwrap().append_block(func_block);
        module.body().append_operation(func_op.into());

        let output = module.as_operation().to_string();

        assert!(
            output.contains("tt.reduce"),
            "missing tt.reduce mnemonic:\n{output}"
        );
        assert!(
            output.contains("tt.reduce.return"),
            "missing tt.reduce.return terminator:\n{output}"
        );
        assert!(
            output.contains("f32"),
            "missing f32 type annotation:\n{output}"
        );
    }

    /// Verify pretty-printed `tt.reduce` with exact IR form.
    ///
    /// Checks that the emitted IR contains the canonical pretty-print strings
    /// for a `tt.reduce` op with axis=0.
    #[test]
    fn test_reduce_pretty_format() {
        let context = create_test_context();
        load_triton_dialect(&context);

        let location = Location::unknown(&context);
        let module = Module::new(location);

        let f32_type = melior::ir::Type::float32(&context);
        let src_ty: Type = tensor_type(&[4], f32_type).into();

        let func_op = create_func(
            &context,
            location,
            "test_reduce_pretty",
            "public",
            &[src_ty],
            &[f32_type],
            0,
        )
        .unwrap();

        let combine_block = Block::new(&[(f32_type, location), (f32_type, location)]);
        let lhs: Value = combine_block.argument(0).unwrap().into();
        let reduce_ret_op: Operation<'_> =
            reduce_return(&context, location, &[lhs]).unwrap().into();
        combine_block.append_operation(reduce_ret_op);

        let combine_region = melior::ir::Region::new();
        combine_region.append_block(combine_block);

        let func_block = Block::new(&[(src_ty, location)]);
        let src: Value = func_block.argument(0).unwrap().into();

        let reduce_op: Operation<'_> =
            reduce(&context, location, &[src], &[f32_type], 0, combine_region)
                .unwrap()
                .into();

        let reduce_result: Value = reduce_op.result(0).unwrap().into();
        let ret_op = crate::triton::tt::ReturnOperation::builder(&context, location)
            .srcs(&[reduce_result])
            .build();

        func_block.append_operation(reduce_op);
        func_block.append_operation(ret_op.into());
        func_op.body().unwrap().append_block(func_block);
        module.body().append_operation(func_op.into());

        let output = module.as_operation().to_string();

        assert!(
            output.contains("tt.reduce"),
            "missing tt.reduce mnemonic:\n{output}"
        );
        assert!(
            output.contains("axis = 0") || output.contains("axis = 0 :"),
            "missing axis attribute:\n{output}"
        );
        assert!(
            output.contains("tensor<4xf32>"),
            "missing source tensor type:\n{output}"
        );
        assert!(
            output.contains("-> f32") || output.contains(") : (tensor<4xf32>) -> f32"),
            "missing scalar result type:\n{output}"
        );
    }

    /// Verify that `print` without arguments emits a valid `tt.print` with only
    /// the prefix and attribute annotations (no operands).
    #[test]
    fn test_print_no_args() {
        let context = create_test_context();
        load_triton_dialect(&context);

        let location = Location::unknown(&context);
        let module = Module::new(location);

        let func_op = create_func(&context, location, "test_print_no_args", "public", &[], &[], 0)
            .unwrap();

        let block = Block::new(&[]);

        let print_op: Operation<'_> =
            print(&context, location, "hello: ", false, &[], &[]).unwrap();
        let ret_op = ReturnOperation::builder(&context, location).srcs(&[]).build();

        block.append_operation(print_op);
        block.append_operation(ret_op.into());
        func_op.body().unwrap().append_block(block);
        module.body().append_operation(func_op.into());

        let output = module.as_operation().to_string();

        assert!(output.contains("tt.print"), "missing tt.print mnemonic:\n{output}");
        assert!(output.contains("hello: "), "missing prefix string:\n{output}");
        assert!(
            output.contains("hex = false") || output.contains("hex = 0"),
            "missing hex attribute:\n{output}"
        );
    }

    /// Verify that `print` with a tensor argument emits the operand and its
    /// type in the IR, along with the `isSigned` array attribute.
    #[test]
    fn test_print_with_arg() {
        let context = create_test_context();
        load_triton_dialect(&context);

        let location = Location::unknown(&context);
        let module = Module::new(location);

        let i32_type: Type = IntegerType::new(&context, 32).into();
        let tensor_ty: Type = tensor_type(&[8], i32_type).into();

        let func_op = create_func(
            &context,
            location,
            "test_print_with_arg",
            "public",
            &[tensor_ty],
            &[],
            0,
        )
        .unwrap();

        let block = Block::new(&[(tensor_ty, location)]);
        let arg_val: Value = block.argument(0).unwrap().into();

        // is_signed = [1] means the single argument is treated as signed.
        let print_op: Operation<'_> =
            print(&context, location, "x: ", false, &[arg_val], &[1]).unwrap();
        let ret_op = ReturnOperation::builder(&context, location).srcs(&[]).build();

        block.append_operation(print_op);
        block.append_operation(ret_op.into());
        func_op.body().unwrap().append_block(block);
        module.body().append_operation(func_op.into());

        let output = module.as_operation().to_string();

        assert!(output.contains("tt.print"), "missing tt.print mnemonic:\n{output}");
        assert!(output.contains("x: "), "missing prefix string:\n{output}");
        assert!(output.contains("tensor<8xi32>"), "missing operand type:\n{output}");
        assert!(output.contains("isSigned"), "missing isSigned attribute:\n{output}");
    }

    /// Verify that `precise_sqrt` on a scalar f32 emits the canonical
    /// `tt.precise_sqrt %arg0 : f32` IR form.
    #[test]
    fn test_precise_sqrt_scalar() {
        let context = create_test_context();
        load_triton_dialect(&context);

        let location = Location::unknown(&context);
        let module = Module::new(location);

        let f32_type = Type::float32(&context);

        let func_op =
            create_func(&context, location, "test_precise_sqrt_scalar", "public", &[f32_type], &[f32_type], 0)
                .unwrap();

        let block = Block::new(&[(f32_type, location)]);
        let x: Value = block.argument(0).unwrap().into();

        let sqrt_op = precise_sqrt(&context, location, x).unwrap();
        let result: Value = sqrt_op.result().unwrap().into();
        let ret_op = ReturnOperation::builder(&context, location).srcs(&[result]).build();

        block.append_operation(sqrt_op.into());
        block.append_operation(ret_op.into());
        func_op.body().unwrap().append_block(block);
        module.body().append_operation(func_op.into());

        let output = module.as_operation().to_string();

        assert!(output.contains("tt.precise_sqrt"), "missing tt.precise_sqrt mnemonic:\n{output}");
        assert!(output.contains("f32"), "missing f32 type:\n{output}");
    }

    /// Verify that `precise_sqrt` on a tensor emits the canonical
    /// `tt.precise_sqrt %arg0 : tensor<8xf32>` IR form.
    #[test]
    fn test_precise_sqrt_tensor() {
        let context = create_test_context();
        load_triton_dialect(&context);

        let location = Location::unknown(&context);
        let module = Module::new(location);

        let f32_type = Type::float32(&context);
        let tensor_ty: Type = tensor_type(&[8], f32_type).into();

        let func_op = create_func(
            &context,
            location,
            "test_precise_sqrt_tensor",
            "public",
            &[tensor_ty],
            &[tensor_ty],
            0,
        )
        .unwrap();

        let block = Block::new(&[(tensor_ty, location)]);
        let x: Value = block.argument(0).unwrap().into();

        let sqrt_op = precise_sqrt(&context, location, x).unwrap();
        let result: Value = sqrt_op.result().unwrap().into();
        let ret_op = ReturnOperation::builder(&context, location).srcs(&[result]).build();

        block.append_operation(sqrt_op.into());
        block.append_operation(ret_op.into());
        func_op.body().unwrap().append_block(block);
        module.body().append_operation(func_op.into());

        let output = module.as_operation().to_string();

        assert!(output.contains("tt.precise_sqrt"), "missing tt.precise_sqrt mnemonic:\n{output}");
        assert!(
            output.contains("tensor<8xf32>"),
            "missing tensor<8xf32> type:\n{output}"
        );
    }

    /// Verify that `precise_divf` on scalar f32 emits the canonical
    /// `tt.precise_divf %arg0, %arg1 : f32` IR form.
    #[test]
    fn test_precise_divf_scalar() {
        let context = create_test_context();
        load_triton_dialect(&context);

        let location = Location::unknown(&context);
        let module = Module::new(location);

        let f32_type = Type::float32(&context);

        let func_op =
            create_func(&context, location, "test_precise_divf_scalar", "public", &[f32_type, f32_type], &[f32_type], 0)
                .unwrap();

        let block = Block::new(&[(f32_type, location), (f32_type, location)]);
        let x: Value = block.argument(0).unwrap().into();
        let y: Value = block.argument(1).unwrap().into();

        let divf_op = precise_divf(&context, location, x, y).unwrap();
        let result: Value = divf_op.result().unwrap().into();
        let ret_op = ReturnOperation::builder(&context, location).srcs(&[result]).build();

        block.append_operation(divf_op.into());
        block.append_operation(ret_op.into());
        func_op.body().unwrap().append_block(block);
        module.body().append_operation(func_op.into());

        let output = module.as_operation().to_string();

        assert!(output.contains("tt.precise_divf"), "missing tt.precise_divf mnemonic:\n{output}");
        assert!(output.contains("f32"), "missing f32 type:\n{output}");
    }

    /// Verify that `precise_divf` on a tensor emits the canonical
    /// `tt.precise_divf %arg0, %arg1 : tensor<8xf32>` IR form.
    #[test]
    fn test_precise_divf_tensor() {
        let context = create_test_context();
        load_triton_dialect(&context);

        let location = Location::unknown(&context);
        let module = Module::new(location);

        let f32_type = Type::float32(&context);
        let tensor_ty: Type = tensor_type(&[8], f32_type).into();

        let func_op = create_func(
            &context,
            location,
            "test_precise_divf_tensor",
            "public",
            &[tensor_ty, tensor_ty],
            &[tensor_ty],
            0,
        )
        .unwrap();

        let block = Block::new(&[(tensor_ty, location), (tensor_ty, location)]);
        let x: Value = block.argument(0).unwrap().into();
        let y: Value = block.argument(1).unwrap().into();

        let divf_op = precise_divf(&context, location, x, y).unwrap();
        let result: Value = divf_op.result().unwrap().into();
        let ret_op = ReturnOperation::builder(&context, location).srcs(&[result]).build();

        block.append_operation(divf_op.into());
        block.append_operation(ret_op.into());
        func_op.body().unwrap().append_block(block);
        module.body().append_operation(func_op.into());

        let output = module.as_operation().to_string();

        assert!(output.contains("tt.precise_divf"), "missing tt.precise_divf mnemonic:\n{output}");
        assert!(
            output.contains("tensor<8xf32>"),
            "missing tensor<8xf32> type:\n{output}"
        );
    }

    /// Verify that `map_elementwise` builds a valid `tt.map_elementwise` op.
    ///
    /// Constructs a `tt.map_elementwise` over a `tensor<4xf32>` whose `scalarOp`
    /// region passes the element through unchanged via `tt.map_elementwise.return`.
    ///
    /// Expected IR contains:
    /// ```text
    /// tt.map_elementwise
    /// tt.map_elementwise.return
    /// pack = 0
    /// ```
    #[test]
    fn test_map_elementwise() {
        let context = create_test_context();
        load_triton_dialect(&context);

        let location = Location::unknown(&context);
        let module = Module::new(location);

        let f32_type = melior::ir::Type::float32(&context);
        let src_ty: Type = tensor_type(&[4], f32_type).into();

        let func_op = create_func(
            &context,
            location,
            "test_map_elementwise",
            "public",
            &[src_ty],
            &[src_ty],
            0,
        )
        .unwrap();

        // scalarOp region: block(%arg0: f32) { tt.map_elementwise.return %arg0 : f32 }
        let scalar_block = Block::new(&[(f32_type, location)]);
        let elem: Value = scalar_block.argument(0).unwrap().into();
        let ret_op: Operation<'_> =
            map_elementwise_return(&context, location, &[elem]).unwrap().into();
        scalar_block.append_operation(ret_op);
        let scalar_region = melior::ir::Region::new();
        scalar_region.append_block(scalar_block);

        let func_block = Block::new(&[(src_ty, location)]);
        let src: Value = func_block.argument(0).unwrap().into();

        let map_op: Operation<'_> =
            map_elementwise(&context, location, &[src], &[src_ty], 0, scalar_region)
                .unwrap()
                .into();

        let map_result: Value = map_op.result(0).unwrap().into();
        let fn_ret = ReturnOperation::builder(&context, location).srcs(&[map_result]).build();

        func_block.append_operation(map_op);
        func_block.append_operation(fn_ret.into());
        func_op.body().unwrap().append_block(func_block);
        module.body().append_operation(func_op.into());

        let output = module.as_operation().to_string();

        assert!(
            output.contains("tt.map_elementwise"),
            "missing tt.map_elementwise mnemonic:\n{output}"
        );
        assert!(
            output.contains("tt.map_elementwise.return"),
            "missing tt.map_elementwise.return terminator:\n{output}"
        );
        assert!(
            output.contains("pack = 0") || output.contains("pack = 0 :"),
            "missing pack attribute:\n{output}"
        );
        assert!(output.contains("tensor<4xf32>"), "missing tensor<4xf32> type:\n{output}");
    }

    /// Verify that `map_elementwise_return` emits a valid `tt.map_elementwise.return` op.
    ///
    /// Builds a `tt.map_elementwise` op over a `tensor<4xf32>` whose `scalarOp`
    /// region contains a single block argument (`f32`) and a
    /// `tt.map_elementwise.return` that yields it unchanged.
    ///
    /// Expected IR contains:
    /// ```text
    /// tt.map_elementwise.return %arg0 : f32
    /// ```
    #[test]
    fn test_map_elementwise_return() {
        let context = create_test_context();
        load_triton_dialect(&context);

        let location = Location::unknown(&context);
        let module = Module::new(location);

        let f32_type = melior::ir::Type::float32(&context);
        let src_ty: Type = tensor_type(&[4], f32_type).into();

        // Wrap inside a tt.func so all SSA values are valid.
        let func_op = create_func(
            &context,
            location,
            "test_map_elementwise_return",
            "public",
            &[src_ty],
            &[src_ty],
            0,
        )
        .unwrap();

        // --- Build the scalarOp region ---
        // The block has one f32 argument (the element value); yield it back unchanged.
        let scalar_block = Block::new(&[(f32_type, location)]);
        let elem: Value = scalar_block.argument(0).unwrap().into();

        // tt.map_elementwise.return %elem : f32
        let ret_op: Operation<'_> =
            map_elementwise_return(&context, location, &[elem]).unwrap().into();
        scalar_block.append_operation(ret_op);

        let scalar_region = melior::ir::Region::new();
        scalar_region.append_block(scalar_block);

        // --- Build the tt.map_elementwise op ---
        let func_block = Block::new(&[(src_ty, location)]);
        let src: Value = func_block.argument(0).unwrap().into();

        let pack_attr = attr_i32(&context, 0);

        let map_op: Operation<'_> = MapElementwiseOperation::builder(&context, location)
            .result(&[src_ty])
            .srcs(&[src])
            .scalar_op(scalar_region)
            .pack(pack_attr)
            .build()
            .into();

        let map_result: Value = map_op.result(0).unwrap().into();
        let fn_ret = ReturnOperation::builder(&context, location).srcs(&[map_result]).build();

        func_block.append_operation(map_op);
        func_block.append_operation(fn_ret.into());
        func_op.body().unwrap().append_block(func_block);
        module.body().append_operation(func_op.into());

        let output = module.as_operation().to_string();

        assert!(
            output.contains("tt.map_elementwise"),
            "missing tt.map_elementwise mnemonic:\n{output}"
        );
        assert!(
            output.contains("tt.map_elementwise.return"),
            "missing tt.map_elementwise.return terminator:\n{output}"
        );
        assert!(output.contains("f32"), "missing f32 type annotation:\n{output}");
    }

    /// Verify the exact pretty-printed form of `tt.map_elementwise.return`.
    ///
    /// Checks that the emitted IR contains the canonical assembly-format string:
    /// ```text
    /// tt.map_elementwise.return %arg0 : f32
    /// ```
    #[test]
    fn test_map_elementwise_return_pretty_format() {
        let context = create_test_context();
        load_triton_dialect(&context);

        let location = Location::unknown(&context);
        let module = Module::new(location);

        let f32_type = melior::ir::Type::float32(&context);
        let src_ty: Type = tensor_type(&[4], f32_type).into();

        let func_op = create_func(
            &context,
            location,
            "test_map_elementwise_return_pretty",
            "public",
            &[src_ty],
            &[src_ty],
            0,
        )
        .unwrap();

        // scalarOp region: block(%arg0: f32) { tt.map_elementwise.return %arg0 : f32 }
        let scalar_block = Block::new(&[(f32_type, location)]);
        let elem: Value = scalar_block.argument(0).unwrap().into();
        let ret_op: Operation<'_> =
            map_elementwise_return(&context, location, &[elem]).unwrap().into();
        scalar_block.append_operation(ret_op);

        let scalar_region = melior::ir::Region::new();
        scalar_region.append_block(scalar_block);

        let func_block = Block::new(&[(src_ty, location)]);
        let src: Value = func_block.argument(0).unwrap().into();

        let pack_attr = attr_i32(&context, 0);

        let map_op: Operation<'_> = MapElementwiseOperation::builder(&context, location)
            .result(&[src_ty])
            .srcs(&[src])
            .scalar_op(scalar_region)
            .pack(pack_attr)
            .build()
            .into();

        let map_result: Value = map_op.result(0).unwrap().into();
        let fn_ret = ReturnOperation::builder(&context, location).srcs(&[map_result]).build();

        func_block.append_operation(map_op);
        func_block.append_operation(fn_ret.into());
        func_op.body().unwrap().append_block(func_block);
        module.body().append_operation(func_op.into());

        let output = module.as_operation().to_string();

        // Pretty-printed form: "tt.map_elementwise.return %arg0 : f32"
        assert!(
            output.contains("tt.map_elementwise.return"),
            "missing tt.map_elementwise.return mnemonic:\n{output}"
        );
        assert!(
            output.contains(": f32"),
            "missing f32 type annotation in tt.map_elementwise.return:\n{output}"
        );
        assert!(
            output.contains("pack = 0") || output.contains("pack = 0 :"),
            "missing pack attribute:\n{output}"
        );
    }

    /// Verify that `make_tensor_ptr` emits a valid `tt.make_tensor_ptr` op.
    ///
    /// Uses function block arguments for all SSA operands so that the
    /// pretty-printed IR does not mix `arith.constant` (generic format) with
    /// `tt.*` ops (pretty format).
    ///
    /// Expected assembly fragment:
    /// ```text
    /// %0 = tt.make_tensor_ptr %arg0, [%arg1, %arg2], [%arg3, %arg4], [%arg5, %arg6]
    ///         {order = array<i32: 1, 0>} : !tt.ptr<tensor<8x4xf16>>
    /// ```
    #[test]
    fn test_make_tensor_ptr() {
        let context = create_test_context();
        load_triton_dialect(&context);

        let location = Location::unknown(&context);
        let module = Module::new(location);

        let f16_type = Type::float16(&context);
        let i64_type: Type = IntegerType::new(&context, 64).into();
        let i32_type: Type = IntegerType::new(&context, 32).into();

        // base: !tt.ptr<f16>
        let base_ptr_ty = pointer_type(f16_type);
        // result: !tt.ptr<tensor<8x4xf16>>
        let tensor_ty: Type = tensor_type(&[8, 4], f16_type).into();
        let result_ty = pointer_type(tensor_ty);

        // Function arguments: base, s0, s1 (shape), str0, str1 (strides), off0, off1 (offsets)
        let arg_types =
            [base_ptr_ty, i64_type, i64_type, i64_type, i64_type, i32_type, i32_type];

        let func_op = create_func(
            &context,
            location,
            "test_make_tensor_ptr",
            "public",
            &arg_types,
            &[result_ty],
            0,
        )
        .unwrap();

        let block = Block::new(&[
            (base_ptr_ty, location),
            (i64_type, location),
            (i64_type, location),
            (i64_type, location),
            (i64_type, location),
            (i32_type, location),
            (i32_type, location),
        ]);

        let base: Value = block.argument(0).unwrap().into();
        let s0: Value = block.argument(1).unwrap().into();
        let s1: Value = block.argument(2).unwrap().into();
        let str0: Value = block.argument(3).unwrap().into();
        let str1: Value = block.argument(4).unwrap().into();
        let off0: Value = block.argument(5).unwrap().into();
        let off1: Value = block.argument(6).unwrap().into();

        let ptr_op: Operation<'_> = make_tensor_ptr(
            &context,
            location,
            base,
            &[s0, s1],
            &[str0, str1],
            &[off0, off1],
            &[1, 0],
            result_ty,
        )
        .unwrap();

        let ptr_val: Value = ptr_op.result(0).unwrap().into();
        let ret_op = ReturnOperation::builder(&context, location).srcs(&[ptr_val]).build();

        block.append_operation(ptr_op);
        block.append_operation(ret_op.into());
        func_op.body().unwrap().append_block(block);
        module.body().append_operation(func_op.into());

        let output = module.as_operation().to_string();

        assert!(
            output.contains("tt.make_tensor_ptr"),
            "missing op mnemonic:\n{output}"
        );
        assert!(
            output.contains("!tt.ptr<tensor<8x4xf16>>"),
            "missing result tensor-pointer type:\n{output}"
        );
        assert!(
            output.contains("order = array<i32: 1, 0>"),
            "missing order attribute:\n{output}"
        );
    }

    /// Verify that `make_tensor_descriptor` emits the correct
    /// `tt.make_tensor_descriptor` IR.
    ///
    /// Expected IR (schematic):
    /// ```text
    /// %0 = tt.make_tensor_descriptor %arg0, [%arg1, %arg2], [%arg3, %arg4]
    ///         {padding = 1 : i32} : !tt.ptr<f32>, !tt.tensordesc<tensor<8x16xf32>>
    /// ```
    #[test]
    fn test_make_tensor_descriptor() {
        let context = create_test_context();
        load_triton_dialect(&context);

        let location = Location::unknown(&context);
        let module = Module::new(location);

        let f32_type = Type::float32(&context);
        let i32_type: Type = IntegerType::new(&context, 32).into();
        let i64_type: Type = IntegerType::new(&context, 64).into();

        // base pointer: !tt.ptr<f32>
        let base_ptr_ty = pointer_type(f32_type);

        // result type: !tt.tensordesc<tensor<8x16xf32>>
        let result_ty = Type::parse(&context, "!tt.tensordesc<tensor<8x16xf32>>")
            .expect("valid tensordesc type");

        // Function arguments: base, s0, s1 (shape i32), str0, str1 (strides i64)
        let arg_types = [base_ptr_ty, i32_type, i32_type, i64_type, i64_type];

        let func_op = create_func(
            &context,
            location,
            "test_make_tensor_descriptor",
            "public",
            &arg_types,
            &[result_ty],
            0,
        )
        .unwrap();

        let block = Block::new(&[
            (base_ptr_ty, location),
            (i32_type, location),
            (i32_type, location),
            (i64_type, location),
            (i64_type, location),
        ]);

        let base: Value = block.argument(0).unwrap().into();
        let s0: Value = block.argument(1).unwrap().into();
        let s1: Value = block.argument(2).unwrap().into();
        let str0: Value = block.argument(3).unwrap().into();
        let str1: Value = block.argument(4).unwrap().into();

        let desc_op: Operation<'_> = make_tensor_descriptor(
            &context,
            location,
            base,
            &[s0, s1],
            &[str0, str1],
            PaddingOption::PadZero,
            result_ty,
        )
        .unwrap();

        let desc_val: Value = desc_op.result(0).unwrap().into();
        let ret_op = ReturnOperation::builder(&context, location).srcs(&[desc_val]).build();

        block.append_operation(desc_op);
        block.append_operation(ret_op.into());
        func_op.body().unwrap().append_block(block);
        module.body().append_operation(func_op.into());

        let output = module.as_operation().to_string();

        assert!(
            output.contains("tt.make_tensor_descriptor"),
            "missing op mnemonic:\n{output}"
        );
        assert!(
            output.contains("!tt.tensordesc<tensor<8x16xf32>>"),
            "missing tensordesc result type:\n{output}"
        );
        assert!(
            output.contains("!tt.ptr<f32>"),
            "missing base pointer type:\n{output}"
        );
    }

    /// Verify that `load` without mask or other emits a valid `tt.load` op
    /// with only the pointer operand.
    #[test]
    fn test_load_no_mask() {
        let context = create_test_context();
        load_triton_dialect(&context);

        let location = Location::unknown(&context);
        let module = Module::new(location);

        let f32_type = Type::float32(&context);
        // Pointer tensor: tensor<8x!tt.ptr<f32>>; result: tensor<8xf32>.
        let ptr_elem_ty = pointer_type(f32_type);
        let ptr_tensor_ty: Type = tensor_type(&[8], ptr_elem_ty).into();
        let val_tensor_ty: Type = tensor_type(&[8], f32_type).into();

        let func_op = create_func(
            &context,
            location,
            "test_load_no_mask",
            "public",
            &[ptr_tensor_ty],
            &[val_tensor_ty],
            0,
        )
        .unwrap();

        let block = Block::new(&[(ptr_tensor_ty, location)]);
        let ptr_val: melior::ir::Value = block.argument(0).unwrap().into();

        let load_op: Operation<'_> = super::load(
            &context,
            location,
            ptr_val,
            None,
            None,
            val_tensor_ty,
            CacheModifier::None,
            EvictionPolicy::Normal,
            false,
        )
        .unwrap();

        let ret_val: melior::ir::Value = load_op.result(0).unwrap().into();
        let ret_op = ReturnOperation::builder(&context, location).srcs(&[ret_val]).build();

        block.append_operation(load_op);
        block.append_operation(ret_op.into());
        func_op.body().unwrap().append_block(block);
        module.body().append_operation(func_op.into());

        let output = module.as_operation().to_string();

        assert!(output.contains("tt.load"), "missing op mnemonic:\n{output}");
        assert!(
            output.contains("tensor<8x!tt.ptr<f32>>"),
            "missing ptr tensor type:\n{output}"
        );
        assert!(
            output.contains("tensor<8xf32>"),
            "missing result tensor type:\n{output}"
        );
    }

    /// Verify that `load` with a mask emits a `tt.load` op that includes the
    /// mask operand in the printed IR.
    #[test]
    fn test_load_with_mask() {
        let context = create_test_context();
        load_triton_dialect(&context);

        let location = Location::unknown(&context);
        let module = Module::new(location);

        let f32_type = Type::float32(&context);
        let i1_type: Type = melior::ir::r#type::IntegerType::new(&context, 1).into();

        // Tensor types: tensor<16x!tt.ptr<f32>>, tensor<16xf32>, tensor<16xi1>.
        let ptr_elem_ty = pointer_type(f32_type);
        let ptr_tensor_ty: Type = tensor_type(&[16], ptr_elem_ty).into();
        let val_tensor_ty: Type = tensor_type(&[16], f32_type).into();
        let mask_tensor_ty: Type = tensor_type(&[16], i1_type).into();

        let func_op = create_func(
            &context,
            location,
            "test_load_with_mask",
            "public",
            &[ptr_tensor_ty, mask_tensor_ty],
            &[val_tensor_ty],
            0,
        )
        .unwrap();

        let block = Block::new(&[(ptr_tensor_ty, location), (mask_tensor_ty, location)]);
        let ptr_val: melior::ir::Value = block.argument(0).unwrap().into();
        let mask_val: melior::ir::Value = block.argument(1).unwrap().into();

        let load_op: Operation<'_> = super::load(
            &context,
            location,
            ptr_val,
            Some(mask_val),
            None,
            val_tensor_ty,
            CacheModifier::None,
            EvictionPolicy::Normal,
            false,
        )
        .unwrap();

        let ret_val: melior::ir::Value = load_op.result(0).unwrap().into();
        let ret_op = ReturnOperation::builder(&context, location).srcs(&[ret_val]).build();

        block.append_operation(load_op);
        block.append_operation(ret_op.into());
        func_op.body().unwrap().append_block(block);
        module.body().append_operation(func_op.into());

        let output = module.as_operation().to_string();

        assert!(output.contains("tt.load"), "missing op mnemonic:\n{output}");
        assert!(
            output.contains("tensor<16x!tt.ptr<f32>>"),
            "missing ptr tensor type:\n{output}"
        );
        assert!(
            output.contains("tensor<16xf32>"),
            "missing result tensor type:\n{output}"
        );
        assert!(
            output.contains("tensor<16xi1>"),
            "missing mask tensor type:\n{output}"
        );
    }

    /// Verify that `load` with mask and other emits a `tt.load` including both
    /// optional operands.
    #[test]
    fn test_load_with_mask_and_other() {
        let context = create_test_context();
        load_triton_dialect(&context);

        let location = Location::unknown(&context);
        let module = Module::new(location);

        let f32_type = Type::float32(&context);
        let i1_type: Type = melior::ir::r#type::IntegerType::new(&context, 1).into();

        let ptr_elem_ty = pointer_type(f32_type);
        let ptr_tensor_ty: Type = tensor_type(&[8], ptr_elem_ty).into();
        let val_tensor_ty: Type = tensor_type(&[8], f32_type).into();
        let mask_tensor_ty: Type = tensor_type(&[8], i1_type).into();

        let func_op = create_func(
            &context,
            location,
            "test_load_with_mask_and_other",
            "public",
            &[ptr_tensor_ty, mask_tensor_ty, val_tensor_ty],
            &[val_tensor_ty],
            0,
        )
        .unwrap();

        let block = Block::new(&[
            (ptr_tensor_ty, location),
            (mask_tensor_ty, location),
            (val_tensor_ty, location),
        ]);
        let ptr_val: melior::ir::Value = block.argument(0).unwrap().into();
        let mask_val: melior::ir::Value = block.argument(1).unwrap().into();
        let other_val: melior::ir::Value = block.argument(2).unwrap().into();

        let load_op: Operation<'_> = super::load(
            &context,
            location,
            ptr_val,
            Some(mask_val),
            Some(other_val),
            val_tensor_ty,
            CacheModifier::None,
            EvictionPolicy::Normal,
            false,
        )
        .unwrap();

        let ret_val: melior::ir::Value = load_op.result(0).unwrap().into();
        let ret_op = ReturnOperation::builder(&context, location).srcs(&[ret_val]).build();

        block.append_operation(load_op);
        block.append_operation(ret_op.into());
        func_op.body().unwrap().append_block(block);
        module.body().append_operation(func_op.into());

        let output = module.as_operation().to_string();

        assert!(output.contains("tt.load"), "missing op mnemonic:\n{output}");
        assert!(
            output.contains("tensor<8x!tt.ptr<f32>>"),
            "missing ptr tensor type:\n{output}"
        );
        assert!(
            output.contains("tensor<8xf32>"),
            "missing result/fill-value tensor type:\n{output}"
        );
        assert!(
            output.contains("tensor<8xi1>"),
            "missing mask tensor type:\n{output}"
        );
    }

    /// Verify that non-default `cache` and `evict` attributes appear in the
    /// emitted IR for `tt.load`.
    #[test]
    fn test_load_cache_modifier() {
        let context = create_test_context();
        load_triton_dialect(&context);

        let location = Location::unknown(&context);
        let module = Module::new(location);

        let f32_type = Type::float32(&context);
        let ptr_elem_ty = pointer_type(f32_type);
        let ptr_tensor_ty: Type = tensor_type(&[8], ptr_elem_ty).into();
        let val_tensor_ty: Type = tensor_type(&[8], f32_type).into();

        let func_op = create_func(
            &context,
            location,
            "test_load_cache",
            "public",
            &[ptr_tensor_ty],
            &[val_tensor_ty],
            0,
        )
        .unwrap();

        let block = Block::new(&[(ptr_tensor_ty, location)]);
        let ptr_val: melior::ir::Value = block.argument(0).unwrap().into();

        // Non-default: CacheModifier::Ca (= 2) and EvictionPolicy::EvictFirst (= 2).
        let load_op: Operation<'_> = super::load(
            &context,
            location,
            ptr_val,
            None,
            None,
            val_tensor_ty,
            CacheModifier::Ca,
            EvictionPolicy::EvictFirst,
            false,
        )
        .unwrap();

        let ret_val: melior::ir::Value = load_op.result(0).unwrap().into();
        let ret_op = ReturnOperation::builder(&context, location).srcs(&[ret_val]).build();

        block.append_operation(load_op);
        block.append_operation(ret_op.into());
        func_op.body().unwrap().append_block(block);
        module.body().append_operation(func_op.into());

        let output = module.as_operation().to_string();

        assert!(output.contains("tt.load"), "missing op mnemonic:\n{output}");
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

    /// Verify that `tt.histogram` without a mask emits the correct op.
    ///
    /// Uses a function block-argument as the source to get a typed tensor value
    /// without relying on a constant-folding pass.
    ///
    /// Expected pretty-printed form:
    /// `%0 = tt.histogram %arg0 : tensor<16xi32> -> tensor<16xi32>`
    #[test]
    fn test_histogram_no_mask() {
        let context = create_test_context();
        load_triton_dialect(&context);

        let location = Location::unknown(&context);
        let module = Module::new(location);

        let i32_type: Type = IntegerType::new(&context, 32).into();
        let src_ty: Type = tensor_type(&[16], i32_type).into();
        let result_ty: Type = tensor_type(&[16], i32_type).into();

        let func_op = create_func(
            &context,
            location,
            "test_histogram_no_mask",
            "public",
            &[src_ty],
            &[result_ty],
            0,
        )
        .unwrap();

        let block = Block::new(&[(src_ty, location)]);
        let src: Value = block.argument(0).unwrap().into();

        let hist_op: Operation<'_> =
            super::histogram(&context, location, src, None, result_ty).unwrap();
        let ret_val: Value = hist_op.result(0).unwrap().into();
        let ret_op = ReturnOperation::builder(&context, location).srcs(&[ret_val]).build();

        block.append_operation(hist_op);
        block.append_operation(ret_op.into());
        func_op.body().unwrap().append_block(block);
        module.body().append_operation(func_op.into());

        let output = module.as_operation().to_string();

        assert!(output.contains("tt.histogram"), "missing op mnemonic:\n{output}");
        assert!(output.contains("tensor<16xi32>"), "missing tensor type:\n{output}");
    }

    /// Verify that `tt.histogram` with a mask emits the correct op.
    ///
    /// Expected pretty-printed form:
    /// `%0 = tt.histogram %arg0, %arg1 : tensor<16xi32> -> tensor<16xi32>`
    #[test]
    fn test_histogram_with_mask() {
        let context = create_test_context();
        load_triton_dialect(&context);

        let location = Location::unknown(&context);
        let module = Module::new(location);

        let i32_type: Type = IntegerType::new(&context, 32).into();
        let i1_type: Type = IntegerType::new(&context, 1).into();
        let src_ty: Type = tensor_type(&[16], i32_type).into();
        let mask_ty: Type = tensor_type(&[16], i1_type).into();
        let result_ty: Type = tensor_type(&[16], i32_type).into();

        let func_op = create_func(
            &context,
            location,
            "test_histogram_with_mask",
            "public",
            &[src_ty, mask_ty],
            &[result_ty],
            0,
        )
        .unwrap();

        let block = Block::new(&[(src_ty, location), (mask_ty, location)]);
        let src: Value = block.argument(0).unwrap().into();
        let mask_val: Value = block.argument(1).unwrap().into();

        let hist_op: Operation<'_> =
            super::histogram(&context, location, src, Some(mask_val), result_ty).unwrap();
        let ret_val: Value = hist_op.result(0).unwrap().into();
        let ret_op = ReturnOperation::builder(&context, location).srcs(&[ret_val]).build();

        block.append_operation(hist_op);
        block.append_operation(ret_op.into());
        func_op.body().unwrap().append_block(block);
        module.body().append_operation(func_op.into());

        let output = module.as_operation().to_string();

        assert!(output.contains("tt.histogram"), "missing op mnemonic:\n{output}");
        assert!(output.contains("tensor<16xi32>"), "missing src/result tensor type:\n{output}");
        assert!(output.contains("tensor<16xi1>"), "missing mask tensor type:\n{output}");
    }

    /// Verify that `gather` emits the correct `tt.gather` IR without `efficient_layout`.
    ///
    /// Expected form:
    /// ```text
    /// %result = tt.gather %src[%indices] {axis = 0 : i32}
    ///     : (tensor<16xf32>, tensor<8xi32>) -> tensor<8xf32>
    /// ```
    #[test]
    fn test_gather() {
        let context = create_test_context();
        load_triton_dialect(&context);

        let location = Location::unknown(&context);
        let module = Module::new(location);

        let f32_type: Type = melior::ir::Type::float32(&context);
        let i32_type: Type = IntegerType::new(&context, 32).into();

        // src: tensor<16xf32>, indices: tensor<8xi32>, result: tensor<8xf32>
        let src_ty: Type = tensor_type(&[16], f32_type).into();
        let idx_ty: Type = tensor_type(&[8], i32_type).into();
        let result_ty: Type = tensor_type(&[8], f32_type).into();

        let func_op = create_func(
            &context,
            location,
            "test_gather",
            "public",
            &[src_ty, idx_ty],
            &[result_ty],
            0,
        )
        .unwrap();

        let block = Block::new(&[(src_ty, location), (idx_ty, location)]);
        let src: Value = block.argument(0).unwrap().into();
        let indices: Value = block.argument(1).unwrap().into();

        let gather_op: Operation<'_> =
            super::gather(&context, location, src, indices, 0, false, result_ty).unwrap();
        let ret_val: Value = gather_op.result(0).unwrap().into();
        let ret_op = ReturnOperation::builder(&context, location).srcs(&[ret_val]).build();

        block.append_operation(gather_op);
        block.append_operation(ret_op.into());
        func_op.body().unwrap().append_block(block);
        module.body().append_operation(func_op.into());

        let output = module.as_operation().to_string();

        assert!(output.contains("tt.gather"), "missing op mnemonic:\n{output}");
        assert!(output.contains("axis = 0"), "missing axis attribute:\n{output}");
        assert!(output.contains("tensor<16xf32>"), "missing src tensor type:\n{output}");
        assert!(output.contains("tensor<8xi32>"), "missing indices tensor type:\n{output}");
        assert!(output.contains("tensor<8xf32>"), "missing result tensor type:\n{output}");
        assert!(!output.contains("efficient_layout"), "unexpected efficient_layout in output:\n{output}");
    }

    /// Verify that `gather` with `efficient_layout = true` includes the unit attr.
    #[test]
    fn test_gather_efficient_layout() {
        let context = create_test_context();
        load_triton_dialect(&context);

        let location = Location::unknown(&context);
        let module = Module::new(location);

        let f32_type: Type = melior::ir::Type::float32(&context);
        let i32_type: Type = IntegerType::new(&context, 32).into();

        let src_ty: Type = tensor_type(&[16], f32_type).into();
        let idx_ty: Type = tensor_type(&[8], i32_type).into();
        let result_ty: Type = tensor_type(&[8], f32_type).into();

        let func_op = create_func(
            &context,
            location,
            "test_gather_efficient_layout",
            "public",
            &[src_ty, idx_ty],
            &[result_ty],
            0,
        )
        .unwrap();

        let block = Block::new(&[(src_ty, location), (idx_ty, location)]);
        let src: Value = block.argument(0).unwrap().into();
        let indices: Value = block.argument(1).unwrap().into();

        let gather_op: Operation<'_> =
            super::gather(&context, location, src, indices, 0, true, result_ty).unwrap();
        let ret_val: Value = gather_op.result(0).unwrap().into();
        let ret_op = ReturnOperation::builder(&context, location).srcs(&[ret_val]).build();

        block.append_operation(gather_op);
        block.append_operation(ret_op.into());
        func_op.body().unwrap().append_block(block);
        module.body().append_operation(func_op.into());

        let output = module.as_operation().to_string();

        assert!(output.contains("tt.gather"), "missing op mnemonic:\n{output}");
        assert!(output.contains("efficient_layout"), "missing efficient_layout attr:\n{output}");
    }

    /// Verify that `fp_to_fp` emits the correct `tt.fp_to_fp` IR without the
    /// optional `rounding` attribute.
    ///
    /// Expected assembly fragment (inside a `tt.func`):
    /// ```text
    /// %0 = tt.fp_to_fp %arg0 : f32 -> f16
    /// ```
    #[test]
    fn test_fp_to_fp_no_rounding() {
        let context = create_test_context();
        load_triton_dialect(&context);

        let location = Location::unknown(&context);
        let module = Module::new(location);

        let f32_type = melior::ir::Type::float32(&context);
        let f16_type = melior::ir::Type::float16(&context);

        let func_op = create_func(
            &context,
            location,
            "test_fp_to_fp_no_rounding",
            "public",
            &[f32_type],
            &[f16_type],
            0,
        )
        .unwrap();

        let block = Block::new(&[(f32_type, location)]);
        let src: Value = block.argument(0).unwrap().into();

        let cast_op: Operation<'_> =
            super::fp_to_fp(&context, location, src, f16_type, None).unwrap();
        let result_val: Value = cast_op.result(0).unwrap().into();
        let ret_op = ReturnOperation::builder(&context, location).srcs(&[result_val]).build();

        block.append_operation(cast_op);
        block.append_operation(ret_op.into());
        func_op.body().unwrap().append_block(block);
        module.body().append_operation(func_op.into());

        let output = module.as_operation().to_string();

        assert!(output.contains("tt.fp_to_fp"), "missing op mnemonic:\n{output}");
        assert!(output.contains("f32"), "missing src type:\n{output}");
        assert!(output.contains("f16"), "missing result type:\n{output}");
        // No rounding attribute should appear in the output (match attribute syntax,
        // not the function name which also contains the word "rounding").
        assert!(!output.contains("rounding ="), "unexpected rounding attr:\n{output}");
    }

    /// Verify that `fp_to_fp` emits the correct `tt.fp_to_fp` IR with the
    /// optional `rounding = rtz` attribute present.
    ///
    /// Expected assembly fragment (inside a `tt.func`):
    /// ```text
    /// %0 = tt.fp_to_fp %arg0, rounding = rtz : f32 -> f16
    /// ```
    #[test]
    fn test_fp_to_fp_with_rounding() {
        let context = create_test_context();
        load_triton_dialect(&context);

        let location = Location::unknown(&context);
        let module = Module::new(location);

        let f32_type = melior::ir::Type::float32(&context);
        let f16_type = melior::ir::Type::float16(&context);

        let func_op = create_func(
            &context,
            location,
            "test_fp_to_fp_rtz",
            "public",
            &[f32_type],
            &[f16_type],
            0,
        )
        .unwrap();

        let block = Block::new(&[(f32_type, location)]);
        let src: Value = block.argument(0).unwrap().into();

        let cast_op: Operation<'_> =
            super::fp_to_fp(&context, location, src, f16_type, Some(super::RoundingMode::RTZ))
                .unwrap();
        let result_val: Value = cast_op.result(0).unwrap().into();
        let ret_op = ReturnOperation::builder(&context, location).srcs(&[result_val]).build();

        block.append_operation(cast_op);
        block.append_operation(ret_op.into());
        func_op.body().unwrap().append_block(block);
        module.body().append_operation(func_op.into());

        let output = module.as_operation().to_string();

        assert!(output.contains("tt.fp_to_fp"), "missing op mnemonic:\n{output}");
        assert!(output.contains("f32"), "missing src type:\n{output}");
        assert!(output.contains("f16"), "missing result type:\n{output}");
        // rtz = 0; the attr appears as either the pretty "rounding = rtz" or
        // the generic "rounding = 0 : i32".
        assert!(
            output.contains("rounding"),
            "missing rounding attr:\n{output}"
        );
    }

    /// Verify that `fp_to_fp` works with tensor operands:
    /// `tensor<8xf32> -> tensor<8xf16>`, no rounding.
    #[test]
    fn test_fp_to_fp_tensor() {
        let context = create_test_context();
        load_triton_dialect(&context);

        let location = Location::unknown(&context);
        let module = Module::new(location);

        let f32_type = melior::ir::Type::float32(&context);
        let f16_type = melior::ir::Type::float16(&context);
        let src_ty: Type = tensor_type(&[8], f32_type).into();
        let res_ty: Type = tensor_type(&[8], f16_type).into();

        let func_op = create_func(
            &context,
            location,
            "test_fp_to_fp_tensor",
            "public",
            &[src_ty],
            &[res_ty],
            0,
        )
        .unwrap();

        let block = Block::new(&[(src_ty, location)]);
        let src: Value = block.argument(0).unwrap().into();

        let cast_op: Operation<'_> =
            super::fp_to_fp(&context, location, src, res_ty, None).unwrap();
        let result_val: Value = cast_op.result(0).unwrap().into();
        let ret_op = ReturnOperation::builder(&context, location).srcs(&[result_val]).build();

        block.append_operation(cast_op);
        block.append_operation(ret_op.into());
        func_op.body().unwrap().append_block(block);
        module.body().append_operation(func_op.into());

        let output = module.as_operation().to_string();

        assert!(output.contains("tt.fp_to_fp"), "missing op mnemonic:\n{output}");
        assert!(output.contains("tensor<8xf32>"), "missing src tensor type:\n{output}");
        assert!(output.contains("tensor<8xf16>"), "missing result tensor type:\n{output}");
    }

    /// Verify that `extern_elementwise` builds successfully and emits the
    /// correct `tt.extern_elementwise` mnemonic with all four required
    /// attributes.
    ///
    /// A `tensor<8xf32>` function argument is passed as the sole source
    /// operand.  The result type is also `tensor<8xf32>`.  We check that the
    /// pretty-printed IR contains the op mnemonic, the `libname`, `libpath`,
    /// `symbol`, and `pure` attributes.
    #[test]
    fn test_extern_elementwise() {
        let context = create_test_context();
        load_triton_dialect(&context);

        let location = Location::unknown(&context);
        let module = Module::new(location);

        let f32_type = melior::ir::Type::float32(&context);
        let tensor_ty: Type = tensor_type(&[8], f32_type).into();

        let func_op = create_func(
            &context,
            location,
            "test_extern_elementwise",
            "public",
            &[tensor_ty],
            &[tensor_ty],
            0,
        )
        .unwrap();

        let block = Block::new(&[(tensor_ty, location)]);
        let src: Value = block.argument(0).unwrap().into();

        let op: Operation<'_> = super::extern_elementwise(
            &context,
            location,
            &[src],
            "libdevice",
            "/usr/local/cuda/nvvm/libdevice",
            "__nv_sinf",
            true,
            tensor_ty,
        )
        .unwrap();

        let result_val: Value = op.result(0).unwrap().into();
        let ret_op = ReturnOperation::builder(&context, location).srcs(&[result_val]).build();

        block.append_operation(op);
        block.append_operation(ret_op.into());
        func_op.body().unwrap().append_block(block);
        module.body().append_operation(func_op.into());

        let output = module.as_operation().to_string();

        assert!(
            output.contains("tt.extern_elementwise"),
            "missing op mnemonic:\n{output}"
        );
        assert!(
            output.contains("libdevice"),
            "missing libname attribute:\n{output}"
        );
        assert!(
            output.contains("__nv_sinf"),
            "missing symbol attribute:\n{output}"
        );
        assert!(
            output.contains("tensor<8xf32>"),
            "missing tensor type:\n{output}"
        );
    }

    /// Verify that `extern_elementwise` with multiple source operands emits
    /// correctly — the `functional-type` format should list all operand types.
    #[test]
    fn test_extern_elementwise_multi_src() {
        let context = create_test_context();
        load_triton_dialect(&context);

        let location = Location::unknown(&context);
        let module = Module::new(location);

        let f32_type = melior::ir::Type::float32(&context);
        let tensor_ty: Type = tensor_type(&[4], f32_type).into();

        let func_op = create_func(
            &context,
            location,
            "test_extern_elementwise_multi",
            "public",
            &[tensor_ty, tensor_ty],
            &[tensor_ty],
            0,
        )
        .unwrap();

        let block = Block::new(&[(tensor_ty, location), (tensor_ty, location)]);
        let src0: Value = block.argument(0).unwrap().into();
        let src1: Value = block.argument(1).unwrap().into();

        let op: Operation<'_> = super::extern_elementwise(
            &context,
            location,
            &[src0, src1],
            "libdevice",
            "/path/to/lib",
            "__nv_fmaf",
            false,
            tensor_ty,
        )
        .unwrap();

        let result_val: Value = op.result(0).unwrap().into();
        let ret_op = ReturnOperation::builder(&context, location).srcs(&[result_val]).build();

        block.append_operation(op);
        block.append_operation(ret_op.into());
        func_op.body().unwrap().append_block(block);
        module.body().append_operation(func_op.into());

        let output = module.as_operation().to_string();

        assert!(
            output.contains("tt.extern_elementwise"),
            "missing op mnemonic:\n{output}"
        );
        assert!(
            output.contains("__nv_fmaf"),
            "missing symbol attribute:\n{output}"
        );
        assert!(
            output.contains("tensor<4xf32>"),
            "missing tensor type:\n{output}"
        );
    }

    /// Verify that `expand_dims` emits the correct `tt.expand_dims` IR.
    ///
    /// Expands a 1-D `tensor<8xf32>` to a 2-D `tensor<8x1xf32>` by inserting a
    /// size-1 dimension at axis 1.
    ///
    /// Expected assembly fragment (inside a `tt.func`):
    /// ```text
    /// %0 = tt.expand_dims %arg0 {axis = 1 : i32} : tensor<8xf32> -> tensor<8x1xf32>
    /// ```
    #[test]
    fn test_expand_dims() {
        let context = create_test_context();
        load_triton_dialect(&context);

        let location = Location::unknown(&context);
        let module = Module::new(location);

        let f32_type = melior::ir::Type::float32(&context);
        let src_ty: Type = tensor_type(&[8], f32_type).into();
        let result_ty: Type = tensor_type(&[8, 1], f32_type).into();

        let func_op = create_func(
            &context,
            location,
            "test_expand_dims",
            "public",
            &[src_ty],
            &[result_ty],
            0,
        )
        .unwrap();

        let block = Block::new(&[(src_ty, location)]);
        let src: Value = block.argument(0).unwrap().into();

        let expand_op: Operation<'_> =
            super::expand_dims(&context, location, src, 1, result_ty).unwrap();
        let ret_val: Value = expand_op.result(0).unwrap().into();
        let ret_op = ReturnOperation::builder(&context, location).srcs(&[ret_val]).build();

        block.append_operation(expand_op);
        block.append_operation(ret_op.into());
        func_op.body().unwrap().append_block(block);
        module.body().append_operation(func_op.into());

        let output = module.as_operation().to_string();

        assert!(output.contains("tt.expand_dims"), "missing op mnemonic:\n{output}");
        assert!(output.contains("axis = 1"), "missing axis attribute:\n{output}");
        assert!(output.contains("tensor<8xf32>"), "missing src tensor type:\n{output}");
        assert!(output.contains("tensor<8x1xf32>"), "missing result tensor type:\n{output}");
    }

    /// Verify that `expand_dims` at axis 0 inserts the new dimension at the front.
    ///
    /// Expands a 1-D `tensor<4xf32>` to `tensor<1x4xf32>` with `axis = 0`.
    ///
    /// Expected assembly fragment:
    /// ```text
    /// %0 = tt.expand_dims %arg0 {axis = 0 : i32} : tensor<4xf32> -> tensor<1x4xf32>
    /// ```
    #[test]
    fn test_expand_dims_axis0() {
        let context = create_test_context();
        load_triton_dialect(&context);

        let location = Location::unknown(&context);
        let module = Module::new(location);

        let f32_type = melior::ir::Type::float32(&context);
        let src_ty: Type = tensor_type(&[4], f32_type).into();
        let result_ty: Type = tensor_type(&[1, 4], f32_type).into();

        let func_op = create_func(
            &context,
            location,
            "test_expand_dims_axis0",
            "public",
            &[src_ty],
            &[result_ty],
            0,
        )
        .unwrap();

        let block = Block::new(&[(src_ty, location)]);
        let src: Value = block.argument(0).unwrap().into();

        let expand_op: Operation<'_> =
            super::expand_dims(&context, location, src, 0, result_ty).unwrap();
        let ret_val: Value = expand_op.result(0).unwrap().into();
        let ret_op = ReturnOperation::builder(&context, location).srcs(&[ret_val]).build();

        block.append_operation(expand_op);
        block.append_operation(ret_op.into());
        func_op.body().unwrap().append_block(block);
        module.body().append_operation(func_op.into());

        let output = module.as_operation().to_string();

        assert!(output.contains("tt.expand_dims"), "missing op mnemonic:\n{output}");
        assert!(output.contains("axis = 0"), "missing axis attribute:\n{output}");
        assert!(output.contains("tensor<4xf32>"), "missing src tensor type:\n{output}");
        assert!(output.contains("tensor<1x4xf32>"), "missing result tensor type:\n{output}");
    }

    /// Verify that `elementwise_inline_asm` with a single input emits the correct
    /// `tt.elementwise_inline_asm` IR.
    ///
    /// Uses a function block-argument as the operand to avoid dependency on
    /// constant-folding.  The expected IR contains the op mnemonic, asm string,
    /// constraint string, `pure` attribute, `packed_element` attribute, and the
    /// operand / result types.
    #[test]
    fn test_elementwise_inline_asm() {
        let context = create_test_context();
        load_triton_dialect(&context);

        let location = Location::unknown(&context);
        let module = Module::new(location);

        let f32_type = melior::ir::Type::float32(&context);
        let tensor_ty: Type = tensor_type(&[8], f32_type).into();

        let func_op = create_func(
            &context,
            location,
            "test_elementwise_inline_asm",
            "public",
            &[tensor_ty],
            &[tensor_ty],
            0,
        )
        .unwrap();

        let block = Block::new(&[(tensor_ty, location)]);
        let arg: Value = block.argument(0).unwrap().into();

        let op: Operation<'_> = super::elementwise_inline_asm(
            &context,
            location,
            "cvt.rn.f32.f32 $0, $1;",
            "=r,r",
            true,
            1,
            &[arg],
            &[tensor_ty],
        )
        .unwrap();

        let result_val: Value = op.result(0).unwrap().into();
        let ret_op = ReturnOperation::builder(&context, location).srcs(&[result_val]).build();

        block.append_operation(op);
        block.append_operation(ret_op.into());
        func_op.body().unwrap().append_block(block);
        module.body().append_operation(func_op.into());

        let output = module.as_operation().to_string();

        assert!(
            output.contains("tt.elementwise_inline_asm"),
            "missing op mnemonic:\n{output}"
        );
        assert!(
            output.contains("cvt.rn.f32.f32 $0, $1;"),
            "missing asm_string attribute:\n{output}"
        );
        assert!(output.contains("=r,r"), "missing constraints attribute:\n{output}");
        assert!(output.contains("packed_element"), "missing packed_element attribute:\n{output}");
        assert!(output.contains("tensor<8xf32>"), "missing tensor type:\n{output}");
    }

    /// Verify that `elementwise_inline_asm` with multiple inputs emits correctly.
    ///
    /// Two `tensor<4xf32>` inputs are passed to a two-operand inline asm block
    /// that produces a single `tensor<4xf32>` result.
    #[test]
    fn test_elementwise_inline_asm_multi_arg() {
        let context = create_test_context();
        load_triton_dialect(&context);

        let location = Location::unknown(&context);
        let module = Module::new(location);

        let f32_type = melior::ir::Type::float32(&context);
        let tensor_ty: Type = tensor_type(&[4], f32_type).into();

        let func_op = create_func(
            &context,
            location,
            "test_elementwise_inline_asm_multi",
            "public",
            &[tensor_ty, tensor_ty],
            &[tensor_ty],
            0,
        )
        .unwrap();

        let block = Block::new(&[(tensor_ty, location), (tensor_ty, location)]);
        let arg0: Value = block.argument(0).unwrap().into();
        let arg1: Value = block.argument(1).unwrap().into();

        let op: Operation<'_> = super::elementwise_inline_asm(
            &context,
            location,
            "add.f32 $0, $1, $2;",
            "=r,r,r",
            true,
            1,
            &[arg0, arg1],
            &[tensor_ty],
        )
        .unwrap();

        let result_val: Value = op.result(0).unwrap().into();
        let ret_op = ReturnOperation::builder(&context, location).srcs(&[result_val]).build();

        block.append_operation(op);
        block.append_operation(ret_op.into());
        func_op.body().unwrap().append_block(block);
        module.body().append_operation(func_op.into());

        let output = module.as_operation().to_string();

        assert!(
            output.contains("tt.elementwise_inline_asm"),
            "missing op mnemonic:\n{output}"
        );
        assert!(
            output.contains("add.f32 $0, $1, $2;"),
            "missing asm_string attribute:\n{output}"
        );
        assert!(output.contains("=r,r,r"), "missing constraints attribute:\n{output}");
        assert!(output.contains("tensor<4xf32>"), "missing tensor type:\n{output}");
    }

    /// Verify that `descriptor_store` emits the correct `tt.descriptor_store` IR.
    ///
    /// Uses function block-arguments to supply typed values without needing a
    /// real TMA descriptor at compile time.
    #[test]
    fn test_descriptor_store() {
        let context = create_test_context();
        load_triton_dialect(&context);

        let location = Location::unknown(&context);
        let module = Module::new(location);

        // Source tensor type: tensor<16x16xf32>.
        let f32_type = melior::ir::Type::float32(&context);
        let src_tensor_ty: Type = tensor_type(&[16, 16], f32_type).into();

        // Descriptor type: !tt.tensordesc<tensor<16x16xf32>>.
        let desc_ty = Type::parse(&context, "!tt.tensordesc<tensor<16x16xf32>>")
            .expect("valid tensordesc type");

        // Two i32 indices for a 2-D descriptor.
        let i32_type: Type = IntegerType::new(&context, 32).into();

        // Build a wrapper function so operands have proper block-argument types.
        let func_op = create_func(
            &context,
            location,
            "test_descriptor_store",
            "public",
            &[desc_ty, src_tensor_ty, i32_type, i32_type],
            &[],
            0,
        )
        .unwrap();

        let block = Block::new(&[
            (desc_ty, location),
            (src_tensor_ty, location),
            (i32_type, location),
            (i32_type, location),
        ]);

        let desc: Value = block.argument(0).unwrap().into();
        let src: Value = block.argument(1).unwrap().into();
        let idx0: Value = block.argument(2).unwrap().into();
        let idx1: Value = block.argument(3).unwrap().into();

        let store_op: Operation<'_> =
            descriptor_store(&context, location, desc, src, &[idx0, idx1]).unwrap();

        // tt.descriptor_store has no results; emit a void return.
        let ret_op = ReturnOperation::builder(&context, location).srcs(&[]).build();

        block.append_operation(store_op);
        block.append_operation(ret_op.into());
        func_op.body().unwrap().append_block(block);
        module.body().append_operation(func_op.into());

        let output = module.as_operation().to_string();

        assert!(
            output.contains("tt.descriptor_store"),
            "missing op mnemonic:\n{output}"
        );
        assert!(
            output.contains("tensordesc"),
            "missing tensordesc operand type:\n{output}"
        );
        assert!(
            output.contains("tensor<16x16xf32>"),
            "missing src tensor type:\n{output}"
        );
    }

    /// Verify that `clampf` emits a correct `tt.clampf` op with
    /// `propagateNan = none` for a scalar f32 operand.
    ///
    /// Expected assembly fragment (inside a `tt.func`):
    /// ```text
    /// %result = tt.clampf %arg0, %arg1, %arg2, propagateNan = none : f32
    /// ```
    #[test]
    fn test_clampf_scalar_none() {
        let context = create_test_context();
        load_triton_dialect(&context);

        let location = Location::unknown(&context);
        let module = Module::new(location);

        let f32_type = melior::ir::Type::float32(&context);

        let func_op = create_func(
            &context,
            location,
            "test_clampf_scalar_none",
            "public",
            &[f32_type, f32_type, f32_type],
            &[f32_type],
            0,
        )
        .unwrap();

        let block = Block::new(&[(f32_type, location), (f32_type, location), (f32_type, location)]);
        let x: Value = block.argument(0).unwrap().into();
        let min: Value = block.argument(1).unwrap().into();
        let max: Value = block.argument(2).unwrap().into();

        let clamp_op: Operation<'_> =
            super::clampf(&context, location, x, min, max, super::PropagateNan::None, f32_type)
                .unwrap();
        let result_val: Value = clamp_op.result(0).unwrap().into();
        let ret_op = ReturnOperation::builder(&context, location).srcs(&[result_val]).build();

        block.append_operation(clamp_op);
        block.append_operation(ret_op.into());
        func_op.body().unwrap().append_block(block);
        module.body().append_operation(func_op.into());

        let output = module.as_operation().to_string();

        assert!(output.contains("tt.clampf"), "missing op mnemonic:\n{output}");
        assert!(output.contains("propagateNan"), "missing propagateNan attr:\n{output}");
        assert!(output.contains("none"), "missing 'none' propagateNan value:\n{output}");
        assert!(output.contains("f32"), "missing f32 type:\n{output}");
    }

    /// Verify that `clampf` emits a correct `tt.clampf` op with
    /// `propagateNan = all` for a tensor operand.
    ///
    /// Expected assembly fragment (inside a `tt.func`):
    /// ```text
    /// %result = tt.clampf %arg0, %arg1, %arg2, propagateNan = all : tensor<8xf32>
    /// ```
    #[test]
    fn test_clampf_tensor_all() {
        let context = create_test_context();
        load_triton_dialect(&context);

        let location = Location::unknown(&context);
        let module = Module::new(location);

        let f32_type = melior::ir::Type::float32(&context);
        let tensor_ty: Type = tensor_type(&[8], f32_type).into();

        let func_op = create_func(
            &context,
            location,
            "test_clampf_tensor_all",
            "public",
            &[tensor_ty, tensor_ty, tensor_ty],
            &[tensor_ty],
            0,
        )
        .unwrap();

        let block = Block::new(&[
            (tensor_ty, location),
            (tensor_ty, location),
            (tensor_ty, location),
        ]);
        let x: Value = block.argument(0).unwrap().into();
        let min: Value = block.argument(1).unwrap().into();
        let max: Value = block.argument(2).unwrap().into();

        let clamp_op: Operation<'_> =
            super::clampf(&context, location, x, min, max, super::PropagateNan::All, tensor_ty)
                .unwrap();
        let result_val: Value = clamp_op.result(0).unwrap().into();
        let ret_op = ReturnOperation::builder(&context, location).srcs(&[result_val]).build();

        block.append_operation(clamp_op);
        block.append_operation(ret_op.into());
        func_op.body().unwrap().append_block(block);
        module.body().append_operation(func_op.into());

        let output = module.as_operation().to_string();

        assert!(output.contains("tt.clampf"), "missing op mnemonic:\n{output}");
        assert!(output.contains("propagateNan"), "missing propagateNan attr:\n{output}");
        assert!(output.contains("all"), "missing 'all' propagateNan value:\n{output}");
        assert!(output.contains("tensor<8xf32>"), "missing tensor<8xf32> type:\n{output}");
    }

    /// Verify that `broadcast` emits the correct `tt.broadcast` IR.
    ///
    /// Broadcasts a `tensor<1x32xf32>` to `tensor<4x32xf32>` by expanding
    /// the size-1 leading dimension.
    ///
    /// Expected assembly fragment (inside a `tt.func`):
    /// ```text
    /// %0 = tt.broadcast %arg0 : tensor<1x32xf32> -> tensor<4x32xf32>
    /// ```
    #[test]
    fn test_broadcast() {
        let context = create_test_context();
        load_triton_dialect(&context);

        let location = Location::unknown(&context);
        let module = Module::new(location);

        let f32_type = melior::ir::Type::float32(&context);
        let src_ty: Type = tensor_type(&[1, 32], f32_type).into();
        let result_ty: Type = tensor_type(&[4, 32], f32_type).into();

        let func_op = create_func(
            &context,
            location,
            "test_broadcast",
            "public",
            &[src_ty],
            &[result_ty],
            0,
        )
        .unwrap();

        let block = Block::new(&[(src_ty, location)]);
        let src: Value = block.argument(0).unwrap().into();

        let broadcast_op: Operation<'_> =
            super::broadcast(&context, location, src, result_ty).unwrap();
        let result_val: Value = broadcast_op.result(0).unwrap().into();
        let ret_op = ReturnOperation::builder(&context, location).srcs(&[result_val]).build();

        block.append_operation(broadcast_op);
        block.append_operation(ret_op.into());
        func_op.body().unwrap().append_block(block);
        module.body().append_operation(func_op.into());

        let output = module.as_operation().to_string();

        assert!(output.contains("tt.broadcast"), "missing op mnemonic:\n{output}");
        assert!(output.contains("tensor<1x32xf32>"), "missing src tensor type:\n{output}");
        assert!(output.contains("tensor<4x32xf32>"), "missing result tensor type:\n{output}");
    }

    /// Verify that `broadcast` works with a 3-D tensor expanding the first and
    /// third dimensions simultaneously.
    ///
    /// Broadcasts `tensor<1x32x1xf32>` to `tensor<2x32x4xf32>`.
    ///
    /// Expected assembly fragment:
    /// ```text
    /// %0 = tt.broadcast %arg0 : tensor<1x32x1xf32> -> tensor<2x32x4xf32>
    /// ```
    #[test]
    fn test_broadcast_multi_dim() {
        let context = create_test_context();
        load_triton_dialect(&context);

        let location = Location::unknown(&context);
        let module = Module::new(location);

        let f32_type = melior::ir::Type::float32(&context);
        let src_ty: Type = tensor_type(&[1, 32, 1], f32_type).into();
        let result_ty: Type = tensor_type(&[2, 32, 4], f32_type).into();

        let func_op = create_func(
            &context,
            location,
            "test_broadcast_multi_dim",
            "public",
            &[src_ty],
            &[result_ty],
            0,
        )
        .unwrap();

        let block = Block::new(&[(src_ty, location)]);
        let src: Value = block.argument(0).unwrap().into();

        let broadcast_op: Operation<'_> =
            super::broadcast(&context, location, src, result_ty).unwrap();
        let result_val: Value = broadcast_op.result(0).unwrap().into();
        let ret_op = ReturnOperation::builder(&context, location).srcs(&[result_val]).build();

        block.append_operation(broadcast_op);
        block.append_operation(ret_op.into());
        func_op.body().unwrap().append_block(block);
        module.body().append_operation(func_op.into());

        let output = module.as_operation().to_string();

        assert!(output.contains("tt.broadcast"), "missing op mnemonic:\n{output}");
        assert!(output.contains("tensor<1x32x1xf32>"), "missing src tensor type:\n{output}");
        assert!(output.contains("tensor<2x32x4xf32>"), "missing result tensor type:\n{output}");
    }

    /// Verify that `bitcast` emits the correct `tt.bitcast` IR for a scalar cast.
    ///
    /// Casts `f32` to `i32` (same bitwidth, scalar).
    ///
    /// Expected assembly fragment:
    /// ```text
    /// %0 = tt.bitcast %arg0 : f32 -> i32
    /// ```
    #[test]
    fn test_bitcast_scalar() {
        let context = create_test_context();
        load_triton_dialect(&context);

        let location = Location::unknown(&context);
        let module = Module::new(location);

        let f32_type = melior::ir::Type::float32(&context);
        let i32_type: Type = IntegerType::new(&context, 32).into();

        let func_op = create_func(
            &context,
            location,
            "test_bitcast_scalar",
            "public",
            &[f32_type],
            &[i32_type],
            0,
        )
        .unwrap();

        let block = Block::new(&[(f32_type, location)]);
        let src: Value = block.argument(0).unwrap().into();

        let bitcast_op: Operation<'_> =
            super::bitcast(&context, location, src, i32_type).unwrap();
        let result_val: Value = bitcast_op.result(0).unwrap().into();
        let ret_op = ReturnOperation::builder(&context, location).srcs(&[result_val]).build();

        block.append_operation(bitcast_op);
        block.append_operation(ret_op.into());
        func_op.body().unwrap().append_block(block);
        module.body().append_operation(func_op.into());

        let output = module.as_operation().to_string();

        assert!(output.contains("tt.bitcast"), "missing op mnemonic:\n{output}");
        assert!(output.contains("f32"), "missing src type:\n{output}");
        assert!(output.contains("i32"), "missing result type:\n{output}");
    }

    /// Verify that `bitcast` emits the correct `tt.bitcast` IR for a tensor cast.
    ///
    /// Reinterprets `tensor<8xf32>` as `tensor<8xi32>`.
    ///
    /// Expected assembly fragment:
    /// ```text
    /// %0 = tt.bitcast %arg0 : tensor<8xf32> -> tensor<8xi32>
    /// ```
    #[test]
    fn test_bitcast_tensor() {
        let context = create_test_context();
        load_triton_dialect(&context);

        let location = Location::unknown(&context);
        let module = Module::new(location);

        let f32_type = melior::ir::Type::float32(&context);
        let i32_type: Type = IntegerType::new(&context, 32).into();
        let src_ty: Type = tensor_type(&[8], f32_type).into();
        let result_ty: Type = tensor_type(&[8], i32_type).into();

        let func_op = create_func(
            &context,
            location,
            "test_bitcast_tensor",
            "public",
            &[src_ty],
            &[result_ty],
            0,
        )
        .unwrap();

        let block = Block::new(&[(src_ty, location)]);
        let src: Value = block.argument(0).unwrap().into();

        let bitcast_op: Operation<'_> =
            super::bitcast(&context, location, src, result_ty).unwrap();
        let result_val: Value = bitcast_op.result(0).unwrap().into();
        let ret_op = ReturnOperation::builder(&context, location).srcs(&[result_val]).build();

        block.append_operation(bitcast_op);
        block.append_operation(ret_op.into());
        func_op.body().unwrap().append_block(block);
        module.body().append_operation(func_op.into());

        let output = module.as_operation().to_string();

        assert!(output.contains("tt.bitcast"), "missing op mnemonic:\n{output}");
        assert!(output.contains("tensor<8xf32>"), "missing src tensor type:\n{output}");
        assert!(output.contains("tensor<8xi32>"), "missing result tensor type:\n{output}");
    }
}
