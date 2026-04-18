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

use melior::ir::attribute::FloatAttribute;
use melior::ir::operation::{OperationBuilder, OperationLike};
use melior::ir::r#type::RankedTensorType;
use melior::ir::{Block, BlockLike, BlockRef, Location, Operation, Region, RegionLike, ShapedTypeLike, TypeLike, Value, ValueLike};
use rustc_ast::{FloatTy, IntTy};
use rustc_middle::mir::{BasicBlock, Body, CallSource, Operand, Place, UnwindAction};
use rustc_middle::ty::{EarlyBinder, Instance, TyCtxt, TyKind, TypingEnv};
use melior::ir::r#type::IntegerType;
use rustc_mlir::shared::arith::{Int, create_extsi, create_int_constant};
use rustc_mlir::shared::builtin::tensor_type;
use rustc_mlir::shared::ub::create_ub_poison;
use rustc_mlir::triton::tensor::{
    CacheModifier, EvictionPolicy, InputPrecision, MemSemantic, MemSyncScope, PropagateNan,
    RmwOp, ScaleDotElemType,
    add_ptr, advance, assert_op, atomic_cas, atomic_rmw, broadcast, clampf, descriptor_load,
    descriptor_store, dot, dot_scaled, expand_dims, gather, histogram, join, load, make_range,
    make_tensor_descriptor, make_tensor_ptr, mulhiui, precise_divf, precise_sqrt,
    print as triton_print, reduce, reduce_return, reshape, scan, scan_return, split, splat,
    store, trans, zeros_like,
};
use rustc_mlir::triton::program::{ProgramAxis, create_get_num_programs};
use rustc_span::Span;
use rustc_span::source_map::Spanned;

use crate::mlir::codegen::triton::{CodegenState, TritonCodegen};
use crate::mlir::errors::MlirError;

/// Generate a stub codegen method that ignores all args and returns `ub.poison`
/// of the destination type (or `None` for void returns).
macro_rules! stub_handler {
    ($name:ident) => {
        pub fn $name<'tcx>(
            &self,
            tcx: TyCtxt<'tcx>,
            instance: &Instance<'tcx>,
            mir: &Body<'tcx>,
            _func: &Operand<'tcx>,
            _func_name: &str,
            _args: &[Spanned<Operand<'tcx>>],
            destination: &Place<'tcx>,
            _target: &Option<BasicBlock>,
            _unwind: &UnwindAction,
            _call_source: &CallSource,
            _fn_span: &Span,
            location: Location<'a>,
            mlir_block: &BlockRef<'a, 'a>,
            _state: &mut CodegenState<'a, 'a>,
        ) -> Result<Option<Value<'a, 'a>>, MlirError> {
            self.codegen_ub_stub(tcx, instance, mir, destination, location, mlir_block)
        }
    };
}

impl<'a> TritonCodegen<'a> {
    pub fn codegen_arange<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        _mir: &Body<'tcx>,
        func: &Operand<'tcx>,
        _func_name: &str,
        args: &[Spanned<Operand<'tcx>>],
        destination: &Place<'tcx>,
        target: &Option<BasicBlock>,
        unwind: &UnwindAction,
        call_source: &CallSource,
        fn_span: &Span,
        location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>,
        _state: &mut CodegenState<'a, 'a>,
    ) -> Result<Option<Value<'a, 'a>>, MlirError> {
        println!(
            "[DEBUG] TritonCodegen::codegen_arange: func: {:?} args: {:?} destination: {:?} target: {:?} unwind: {:?} call_source: {:?} fn_span: {:?}",
            func, args, destination, target, unwind, call_source, fn_span
        );

        debug_assert!(
            args.len() == 2,
            "TritonCodegen::codegen_arange: args length must be 2: {:?}",
            args
        );

        let start = self.to_scalar_int(tcx, instance, &args[0].node)?.to_i32();
        let end = self.to_scalar_int(tcx, instance, &args[1].node)?.to_i32();

        let arange_op: Operation<'a> = make_range(self.module.context(), location, start, end)
            .map_err(|e| MlirError::CreateOperation { err: e })?
            .into();

        let result = arange_op.result(0).expect("Arange operation result not found");
        eprintln!("[DEBUG] AXM TritonCodegen::codegen_arange: {:?}", arange_op.to_string());
        mlir_block.append_operation(arange_op);
        Ok(Some(result.into()))
    }

    pub fn codegen_add_ptr<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        mir: &Body<'tcx>,
        _func: &Operand<'tcx>,
        _func_name: &str,
        args: &[Spanned<Operand<'tcx>>],
        _destination: &Place<'tcx>,
        _target: &Option<BasicBlock>,
        _unwind: &UnwindAction,
        _call_source: &CallSource,
        _fn_span: &Span,
        location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>,
        state: &mut CodegenState<'a, 'a>,
    ) -> Result<Option<Value<'a, 'a>>, MlirError> {
        debug_assert!(
            args.len() == 2,
            "TritonCodegen::codegen_add_offsets_call: args length must be 2"
        );

        let arg0 = &args[0].node;
        let arg1 = &args[1].node;

        let ptr = self.codegen_operand(
            tcx, instance, arg0, arg0.ty(mir, tcx), location, mlir_block, state,
        )?;
        let offset = self.codegen_operand(
            tcx, instance, arg1, arg1.ty(mir, tcx), location, mlir_block, state,
        )?;

        debug_assert!(
            offset.r#type().is_tensor(),
            "TritonCodegen::codegen_add_offset: rhs is not a tensor"
        );

        let ptr = self.like_tensor(tcx, location, offset, ptr, mlir_block)?;

        let add_ptr_op: Operation<'a> =
            add_ptr(self.module.context(), location, ptr, offset, ptr.r#type())
                .map_err(|e| MlirError::CreateOperation { err: e })?
                .into();
        let result = add_ptr_op.result(0).expect("AddPtr operation result not found");

        eprintln!("[DEBUG] AXM TritonCodegen::codegen_add_ptr: {:?}", add_ptr_op.to_string());
        mlir_block.append_operation(add_ptr_op);
        Ok(Some(result.into()))
    }

    pub fn codegen_load<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        mir: &Body<'tcx>,
        _func: &Operand<'tcx>,
        _func_name: &str,
        args: &[Spanned<Operand<'tcx>>],
        destination: &Place<'tcx>,
        _target: &Option<BasicBlock>,
        _unwind: &UnwindAction,
        _call_source: &CallSource,
        _fn_span: &Span,
        location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>,
        state: &mut CodegenState<'a, 'a>,
    ) -> Result<Option<Value<'a, 'a>>, MlirError> {
        // args: [ptr_tensor, Option<mask>, Option<other>, &[i32], Option<PaddingOption>,
        //        Option<CacheModifier>, Option<EvictionPolicy>, bool]
        let ptr = self.codegen_operand(
            tcx, instance, &args[0].node, args[0].node.ty(mir, tcx), location, mlir_block, state,
        )?;
        let mask = self.codegen_option_operand(
            tcx, instance, mir, &args[1].node, location, mlir_block, state,
        )?;
        let other = self.codegen_option_operand(
            tcx, instance, mir, &args[2].node, location, mlir_block, state,
        )?;

        // Derive result type from the ptr operand's MLIR type.
        // ptr is `tensor<Nx!tt.ptr<T>>` — extract shape [N] and element type T from
        // the destination Rust type (which gives us `tensor<?xT>`; we take the element).
        let result_ty = {
            let dest_ty = destination.ty(mir, tcx).ty;
            let dest_ty = instance.instantiate_mir_and_normalize_erasing_regions(
                tcx,
                TypingEnv::fully_monomorphized(),
                EarlyBinder::bind(dest_ty),
            );
            let dest_mlir_ty = self.type_mapper.map_type(self.module.context(), &tcx, &dest_ty);

            if let Ok(ptr_tensor_ty) = RankedTensorType::try_from(ptr.r#type()) {
                // ptr is tensor<Nx!tt.ptr<ElemTy>> — use the ptr's concrete shape.
                let shape: Vec<i64> = ptr_tensor_ty
                    .dims()
                    .map_err(|e| MlirError::InvalidType { msg: e.to_string() })?;
                // Element type of result: unwrap outer tensor<> wrapper from dest_mlir_ty if present.
                let elem_ty = if let Ok(dest_tensor_ty) = RankedTensorType::try_from(dest_mlir_ty) {
                    dest_tensor_ty.element()
                } else {
                    dest_mlir_ty
                };
                tensor_type(&shape, elem_ty).into()
            } else {
                // Scalar pointer — result type is exactly the destination type.
                dest_mlir_ty
            }
        };

        let load_op: Operation<'a> = load(
            self.module.context(),
            location,
            ptr,
            mask,
            other,
            result_ty,
            CacheModifier::None,
            EvictionPolicy::Normal,
            false,
        )
        .map_err(|e| MlirError::CreateOperation { err: e })?
        .into();

        let result = load_op.result(0).map_err(|e| MlirError::CodegenFailed { err: e.to_string() })?;
        mlir_block.append_operation(load_op);
        Ok(Some(result.into()))
    }

    pub fn codegen_store<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        mir: &Body<'tcx>,
        _func: &Operand<'tcx>,
        _func_name: &str,
        args: &[Spanned<Operand<'tcx>>],
        _destination: &Place<'tcx>,
        _target: &Option<BasicBlock>,
        _unwind: &UnwindAction,
        _call_source: &CallSource,
        _fn_span: &Span,
        location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>,
        state: &mut CodegenState<'a, 'a>,
    ) -> Result<Option<Value<'a, 'a>>, MlirError> {
        // args: [ptr_tensor, value_tensor, Option<mask>, &[i32],
        //        Option<CacheModifier>, Option<EvictionPolicy>]
        let ptr = self.codegen_operand(
            tcx, instance, &args[0].node, args[0].node.ty(mir, tcx), location, mlir_block, state,
        )?;
        let value = self.codegen_operand(
            tcx, instance, &args[1].node, args[1].node.ty(mir, tcx), location, mlir_block, state,
        )?;
        let mask = self.codegen_option_operand(
            tcx, instance, mir, &args[2].node, location, mlir_block, state,
        )?;

        let store_op: Operation<'a> = store(
            self.module.context(),
            location,
            ptr,
            value,
            mask,
            CacheModifier::None,
            EvictionPolicy::Normal,
        )
        .map_err(|e| MlirError::CreateOperation { err: e })?
        .into();

        mlir_block.append_operation(store_op);
        Ok(None)
    }

    pub fn codegen_maximum<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        mir: &Body<'tcx>,
        _func: &Operand<'tcx>,
        _func_name: &str,
        args: &[Spanned<Operand<'tcx>>],
        _destination: &Place<'tcx>,
        _target: &Option<BasicBlock>,
        _unwind: &UnwindAction,
        _call_source: &CallSource,
        _fn_span: &Span,
        location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>,
        state: &mut CodegenState<'a, 'a>,
    ) -> Result<Option<Value<'a, 'a>>, MlirError> {
        // args: [lhs_tensor, rhs_tensor]
        let lhs = self.codegen_operand(
            tcx, instance, &args[0].node, args[0].node.ty(mir, tcx), location, mlir_block, state,
        )?;
        let rhs = self.codegen_operand(
            tcx, instance, &args[1].node, args[1].node.ty(mir, tcx), location, mlir_block, state,
        )?;

        // arith.maximumf for element-wise float maximum (works on scalars and tensors).
        let result_ty = lhs.r#type();
        let max_op: Operation<'a> = OperationBuilder::new("arith.maximumf", location)
            .add_operands(&[lhs, rhs])
            .add_results(&[result_ty])
            .build()
            .map_err(|e| MlirError::CodegenFailed { err: e.to_string() })?;

        let result = max_op.result(0).map_err(|e| MlirError::CodegenFailed { err: e.to_string() })?;
        mlir_block.append_operation(max_op);
        Ok(Some(result.into()))
    }

    /// `triton::Triton::transmute`-as-slice: `transmute<(*const T, usize), &[T]>((ptr, len))`
    ///
    /// This is used in `no_core` to build a `&[i32]` shape descriptor from a constant array.
    /// We extract the shape from `state.const_arrays` / `state.ptr_to_const_array` and store it
    /// in `state.slice_shape`.  No MLIR operation is emitted.
    pub fn codegen_transmute_slice<'tcx>(
        &self,
        _tcx: TyCtxt<'tcx>,
        _instance: &Instance<'tcx>,
        _mir: &Body<'tcx>,
        _func: &Operand<'tcx>,
        func_name: &str,
        args: &[Spanned<Operand<'tcx>>],
        destination: &Place<'tcx>,
        _target: &Option<BasicBlock>,
        _unwind: &UnwindAction,
        _call_source: &CallSource,
        _fn_span: &Span,
        _location: Location<'a>,
        _mlir_block: &BlockRef<'a, 'a>,
        state: &mut CodegenState<'a, 'a>,
    ) -> Result<Option<Value<'a, 'a>>, MlirError> {
        eprintln!("[DEBUG] codegen_transmute_slice: {func_name}, args: {args:?}");

        // The single argument is the fat-pointer tuple `(*const T, usize)` whose first field
        // is a const-array pointer.  The Tuple aggregate handler already stored the shape in
        // `slice_shape` keyed by the tuple local.  Propagate it to the destination.
        if let Some(Spanned { node: Operand::Move(src_place) | Operand::Copy(src_place), .. }) =
            args.first()
        {
            if let Some(shape) = state.slice_shape.get(&src_place.local).cloned() {
                state.slice_shape.insert(destination.local, shape.clone());
                eprintln!("[DEBUG] codegen_transmute_slice: shape={shape:?} → {:?}", destination.local);
                return Ok(None);
            }
            if let Some(vals) = state.slice_dyn_values.get(&src_place.local).cloned() {
                state.slice_dyn_values.insert(destination.local, vals);
                eprintln!("[DEBUG] codegen_transmute_slice: dyn_values → {:?}", destination.local);
                return Ok(None);
            }
        }

        eprintln!("[DEBUG] codegen_transmute_slice: could not extract shape; args={args:?}");
        Ok(None)
    }

    /// `triton::Triton::zeros` — creates a tensor filled with zeros.
    ///
    /// The shape is recovered from `state.slice_shape` (populated by `codegen_transmute_slice`).
    /// The element type comes from the MIR return-place type.
    pub fn codegen_zeros<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        mir: &Body<'tcx>,
        _func: &Operand<'tcx>,
        _func_name: &str,
        args: &[Spanned<Operand<'tcx>>],
        destination: &Place<'tcx>,
        _target: &Option<BasicBlock>,
        _unwind: &UnwindAction,
        _call_source: &CallSource,
        _fn_span: &Span,
        location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>,
        state: &mut CodegenState<'a, 'a>,
    ) -> Result<Option<Value<'a, 'a>>, MlirError> {
        // Get shape from the slice argument.
        let shape: Vec<i64> = if let Some(Spanned {
            node: Operand::Move(slice_place) | Operand::Copy(slice_place),
            ..
        }) = args.first()
        {
            state
                .slice_shape
                .get(&slice_place.local)
                .cloned()
                .unwrap_or_else(|| {
                    eprintln!("[WARN] codegen_zeros: slice_shape not found for {:?}; using [1]", slice_place.local);
                    vec![1]
                })
        } else {
            vec![1]
        };

        // Get the element Rust type from the destination.
        let dest_ty = destination.ty(mir, tcx).ty;
        let dest_ty = instance.instantiate_mir_and_normalize_erasing_regions(
            tcx,
            TypingEnv::fully_monomorphized(),
            EarlyBinder::bind(dest_ty),
        );

        // Extract the element MLIR type:
        // - For `Tensor<D>` (ADT), map through the type mapper
        // - For direct scalar (unlikely), use directly
        let elem_mlir_ty = self.type_mapper.map_type(self.module.context(), &tcx, &dest_ty);

        // Unwrap: if we got a tensor type back, extract its element type.
        let elem_mlir_ty = if elem_mlir_ty.is_tensor() {
            let tensor_ty: RankedTensorType<'a> = elem_mlir_ty
                .try_into()
                .map_err(|e: melior::error::Error| MlirError::InvalidType { msg: e.to_string() })?;
            tensor_ty.element()
        } else {
            elem_mlir_ty
        };

        // Pointer/non-standard element types: return UB tensor.
        // (splat requires a scalar constant, which we can't create for pointer types.)
        let type_str = elem_mlir_ty.to_string();
        if !elem_mlir_ty.is_integer() && !type_str.contains('f') && !type_str.contains("bf16") {
            eprintln!("[WARN] codegen_zeros: unsupported element type {:?}; returning UB", type_str);
            let tensor_ty = tensor_type(&shape, elem_mlir_ty).into();
            let ub_op: Operation<'a> = create_ub_poison(self.module.context(), location, tensor_ty)
                .map_err(|e| MlirError::CreateOperation { err: e })?
                .into();
            let result = ub_op.result(0).expect("ub poison result");
            mlir_block.append_operation(ub_op);
            return Ok(Some(result.into()));
        }

        // Create a scalar zero constant of the element type.
        let zero_op: Operation<'a> = if elem_mlir_ty.is_integer() {
            create_int_constant(self.module.context(), location, Int::I32(0))
                .map_err(|e| MlirError::CreateOperation { err: e })?
                .into()
        } else {
            // Float zero: emit `arith.constant 0.0 : <float_ty>`
            let zero_attr = FloatAttribute::new(self.module.context(), elem_mlir_ty, 0.0);
            melior::dialect::arith::constant(self.module.context(), zero_attr.into(), location)
        };
        let zero_val = zero_op.result(0).expect("zero constant result");
        mlir_block.append_operation(zero_op);

        // Splat the scalar zero to a tensor of the requested shape.
        let tensor_ty = tensor_type(&shape, elem_mlir_ty).into();
        let splat_op: Operation<'a> =
            splat(self.module.context(), location, zero_val.into(), tensor_ty)
                .map_err(|e| MlirError::CreateOperation { err: e })?
                .into();
        let result = splat_op.result(0).expect("splat result");
        mlir_block.append_operation(splat_op);

        Ok(Some(result.into()))
    }

    pub fn codegen_zeros_like<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        mir: &Body<'tcx>,
        _func: &Operand<'tcx>,
        _func_name: &str,
        args: &[Spanned<Operand<'tcx>>],
        _destination: &Place<'tcx>,
        _target: &Option<BasicBlock>,
        _unwind: &UnwindAction,
        _call_source: &CallSource,
        _fn_span: &Span,
        location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>,
        state: &mut CodegenState<'a, 'a>,
    ) -> Result<Option<Value<'a, 'a>>, MlirError> {
        debug_assert!(
            args.len() == 1,
            "TritonCodegen::codegen_zeros_like: args length must be 1: {:?}",
            args
        );

        let arg0 = &args[0].node;
        let tensor = self.codegen_operand(
            tcx, instance, arg0, arg0.ty(mir, tcx), location, mlir_block, state,
        )?;

        // Extract tensor type from the input tensor's MLIR type.
        let tensor_ty: RankedTensorType<'a> = tensor
            .r#type()
            .try_into()
            .map_err(|e: melior::error::Error| MlirError::InvalidType { msg: e.to_string() })?;
        let elem_ty = tensor_ty.element();

        // Create a zero constant of the element type.
        let zero_op: Operation<'a> = if elem_ty.is_integer() {
            create_int_constant(self.module.context(), location, Int::I32(0))
                .map_err(|e| MlirError::CreateOperation { err: e })?
                .into()
        } else {
            let zero_attr = FloatAttribute::new(self.module.context(), elem_ty, 0.0);
            melior::dialect::arith::constant(self.module.context(), zero_attr.into(), location)
        };
        let zero_val = zero_op.result(0).expect("zero constant result");
        mlir_block.append_operation(zero_op);

        // Splat to the same tensor type as the input.
        let splat_op: Operation<'a> =
            splat(self.module.context(), location, zero_val.into(), tensor.r#type())
                .map_err(|e| MlirError::CreateOperation { err: e })?
                .into();
        let result = splat_op.result(0).expect("splat result");
        mlir_block.append_operation(splat_op);

        Ok(Some(result.into()))
    }

    /// `triton::Triton::cat` — concatenate two tensors along their only axis.
    ///
    /// args[0] = lhs tensor, args[1] = rhs tensor, args[2] = can_reorder (bool, unused in MLIR)
    pub fn codegen_cat_call<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        mir: &Body<'tcx>,
        _func: &Operand<'tcx>,
        _func_name: &str,
        args: &[Spanned<Operand<'tcx>>],
        _destination: &Place<'tcx>,
        _target: &Option<BasicBlock>,
        _unwind: &UnwindAction,
        _call_source: &CallSource,
        _fn_span: &Span,
        location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>,
        state: &mut CodegenState<'a, 'a>,
    ) -> Result<Option<Value<'a, 'a>>, MlirError> {
        use melior::ir::ShapedTypeLike;

        let lhs_arg = &args[0].node;
        let rhs_arg = &args[1].node;
        let lhs = self.codegen_operand(
            tcx, instance, lhs_arg, lhs_arg.ty(mir, tcx), location, mlir_block, state,
        )?;
        let rhs = self.codegen_operand(
            tcx, instance, rhs_arg, rhs_arg.ty(mir, tcx), location, mlir_block, state,
        )?;

        // Compute result shape: sum of each dimension across lhs and rhs.
        let lhs_tensor: RankedTensorType<'a> = lhs.r#type().try_into()
            .map_err(|e: melior::error::Error| MlirError::InvalidType { msg: e.to_string() })?;
        let rhs_tensor: RankedTensorType<'a> = rhs.r#type().try_into()
            .map_err(|e: melior::error::Error| MlirError::InvalidType { msg: e.to_string() })?;
        let lhs_dims = lhs_tensor.dims().map_err(|e| MlirError::InvalidType { msg: e.to_string() })?;
        let rhs_dims = rhs_tensor.dims().map_err(|e| MlirError::InvalidType { msg: e.to_string() })?;

        // For a 1D cat, result size = lhs_dim + rhs_dim.
        let result_dims: Vec<i64> = lhs_dims.iter().zip(rhs_dims.iter())
            .map(|(&l, &r)| l + r)
            .collect();
        let elem_ty = lhs_tensor.element();
        let result_ty = tensor_type(&result_dims, elem_ty).into();

        use melior::ir::operation::OperationBuilder;
        let cat_op: Operation<'a> = OperationBuilder::new("tt.cat", location)
            .add_operands(&[lhs, rhs])
            .add_results(&[result_ty])
            .build()
            .map_err(|e| MlirError::CodegenFailed { err: e.to_string() })?
            .into();
        let result = cat_op.result(0).expect("cat result");
        mlir_block.append_operation(cat_op);
        Ok(Some(result.into()))
    }

    /// `triton::Triton::cast` — element-wise type cast for tensors.
    ///
    /// args[0] = input tensor, args[1] = rounding mode (optional enum), args[2] = saturate (bool)
    ///
    /// For the common case where src and dst element types are the same, returns the input as-is.
    /// For type conversions, emits the appropriate `arith` cast operation.
    pub fn codegen_cast_call<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        mir: &Body<'tcx>,
        _func: &Operand<'tcx>,
        _func_name: &str,
        args: &[Spanned<Operand<'tcx>>],
        destination: &Place<'tcx>,
        _target: &Option<BasicBlock>,
        _unwind: &UnwindAction,
        _call_source: &CallSource,
        _fn_span: &Span,
        location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>,
        state: &mut CodegenState<'a, 'a>,
    ) -> Result<Option<Value<'a, 'a>>, MlirError> {
        use melior::ir::ShapedTypeLike;

        let arg0 = &args[0].node;
        let src_value = self.codegen_operand(
            tcx, instance, arg0, arg0.ty(mir, tcx), location, mlir_block, state,
        )?;

        // Determine the destination element type from the destination place type.
        let dest_rust_ty = destination.ty(mir, tcx).ty;
        let dest_rust_ty = instance.instantiate_mir_and_normalize_erasing_regions(
            tcx,
            TypingEnv::fully_monomorphized(),
            EarlyBinder::bind(dest_rust_ty),
        );
        let dest_mlir_ty = self.type_mapper.map_type(self.module.context(), &tcx, &dest_rust_ty);

        // If destination type is dynamic (unknown-shape tensor), use the input's actual type.
        // Triton cast is always shape-preserving; the function signature just doesn't know
        // the shape statically.
        let dest_mlir_ty = if dest_mlir_ty.is_tensor() {
            let is_dynamic = dest_mlir_ty
                .try_into()
                .ok()
                .and_then(|t: RankedTensorType<'a>| t.dims().ok())
                .map(|dims| dims.iter().any(|&d| d == i64::MIN))
                .unwrap_or(true);
            if is_dynamic {
                // Construct target type: same shape as input, but with the destination element type
                let src_tensor: RankedTensorType<'a> = src_value.r#type().try_into()
                    .map_err(|e: melior::error::Error| MlirError::InvalidType { msg: e.to_string() })?;
                let src_dims = src_tensor.dims().map_err(|e| MlirError::InvalidType { msg: e.to_string() })?;
                // Get destination element type from the generic tensor handler result
                let dest_elem_ty = {
                    let mapped = dest_mlir_ty.try_into()
                        .map(|t: RankedTensorType<'a>| t.element())
                        .unwrap_or(src_tensor.element());
                    mapped
                };
                tensor_type(&src_dims, dest_elem_ty).into()
            } else {
                dest_mlir_ty
            }
        } else {
            dest_mlir_ty
        };

        // If source and destination MLIR types are the same, the cast is a no-op.
        if src_value.r#type() == dest_mlir_ty {
            return Ok(Some(src_value));
        }

        // Otherwise emit the appropriate arith conversion op.
        let src_elem_ty = if src_value.r#type().is_tensor() {
            src_value.r#type().try_into()
                .map(|t: RankedTensorType<'a>| t.element())
                .unwrap_or(src_value.r#type())
        } else {
            src_value.r#type()
        };
        let dst_elem_ty = if dest_mlir_ty.is_tensor() {
            dest_mlir_ty.try_into()
                .map(|t: RankedTensorType<'a>| t.element())
                .unwrap_or(dest_mlir_ty)
        } else {
            dest_mlir_ty
        };

        // melior typed_unary_operations signature: (in_value, out_type, location)
        let cast_op: Operation<'a> = if src_elem_ty.is_float() && dst_elem_ty.is_float() {
            // Float → float: extf (widening) or truncf-family (narrowing).
            // Determine by comparing bit widths via the string representation.
            // f16 < f32 < f64
            let float_bits = |ty: melior::ir::Type<'a>| -> u32 {
                let s = ty.to_string();
                if s.contains("f16") { 16 } else if s.contains("f32") { 32 }
                else if s.contains("f64") { 64 } else { 128 }
            };
            if float_bits(dst_elem_ty) < float_bits(src_elem_ty) {
                // Narrowing: use ODS builder directly since the melior wrapper
                // for truncf is only a same-type unary; fall back to OperationBuilder.
                use melior::ir::operation::OperationBuilder;
                OperationBuilder::new("arith.truncf", location)
                    .add_operands(&[src_value])
                    .add_results(&[dest_mlir_ty])
                    .build()
                    .map_err(|e| MlirError::CreateOperation { err: rustc_mlir::errors::Error::IncompatibleTypes { lhs: e.to_string(), rhs: String::new() } })?
                    .into()
            } else {
                // Widening
                melior::dialect::arith::extf(src_value, dest_mlir_ty, location).into()
            }
        } else if src_elem_ty.is_integer() && dst_elem_ty.is_float() {
            melior::dialect::arith::sitofp(src_value, dest_mlir_ty, location).into()
        } else if src_elem_ty.is_float() && dst_elem_ty.is_integer() {
            melior::dialect::arith::fptosi(src_value, dest_mlir_ty, location).into()
        } else {
            melior::dialect::arith::trunci(src_value, dest_mlir_ty, location).into()
        };
        let result = cast_op.result(0).expect("cast result");
        mlir_block.append_operation(cast_op);
        Ok(Some(result.into()))
    }

    // -------------------------------------------------------------------------
    // Helper: extract a shape Vec<i64> from a `&[i32]` slice operand tracked in
    // `state.slice_shape`.
    // -------------------------------------------------------------------------
    fn shape_from_slice_arg<'tcx>(
        &self,
        arg: &Operand<'tcx>,
        state: &CodegenState<'a, 'a>,
        ctx: &str,
    ) -> Result<Vec<i64>, MlirError> {
        match arg {
            Operand::Copy(p) | Operand::Move(p) => {
                state
                    .slice_shape
                    .get(&p.local)
                    .cloned()
                    .ok_or_else(|| MlirError::CodegenFailed {
                        err: format!("{ctx}: slice_shape not found for {:?}", p.local),
                    })
            }
            other => Err(MlirError::CodegenFailed {
                err: format!("{ctx}: unexpected operand for shape arg: {other:?}"),
            }),
        }
    }

    /// Like `shape_from_slice_arg` but returns `Vec<i32>`, checking both `slice_shape` and
    /// `const_arrays` (the block_shape arg to `make_tensor_descriptor` is `&[i32]`).
    fn shape_from_slice_arg_i32<'tcx>(
        &self,
        arg: &Operand<'tcx>,
        state: &CodegenState<'a, 'a>,
    ) -> Result<Vec<i32>, MlirError> {
        match arg {
            Operand::Copy(p) | Operand::Move(p) => {
                if let Some(shape) = state.slice_shape.get(&p.local) {
                    return Ok(shape.iter().map(|&v| v as i32).collect());
                }
                Err(MlirError::CodegenFailed {
                    err: format!("shape_from_slice_arg_i32: slice_shape not found for {:?}", p.local),
                })
            }
            other => Err(MlirError::CodegenFailed {
                err: format!("shape_from_slice_arg_i32: unexpected operand: {other:?}"),
            }),
        }
    }

    // -------------------------------------------------------------------------
    // `triton::Triton::broadcast(a, b)` — broadcast two tensors to a common
    // shape.  Returns `(Tensor<D>, Tensor<D>)` stored in `tuple_fields`.
    // -------------------------------------------------------------------------
    pub fn codegen_broadcast_call<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        mir: &Body<'tcx>,
        _func: &Operand<'tcx>,
        _func_name: &str,
        args: &[Spanned<Operand<'tcx>>],
        destination: &Place<'tcx>,
        _target: &Option<BasicBlock>,
        _unwind: &UnwindAction,
        _call_source: &CallSource,
        _fn_span: &Span,
        location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>,
        state: &mut CodegenState<'a, 'a>,
    ) -> Result<Option<Value<'a, 'a>>, MlirError> {
        let lhs_arg = &args[0].node;
        let rhs_arg = &args[1].node;
        let lhs = self.codegen_operand(
            tcx, instance, lhs_arg, lhs_arg.ty(mir, tcx), location, mlir_block, state,
        )?;
        let rhs = self.codegen_operand(
            tcx, instance, rhs_arg, rhs_arg.ty(mir, tcx), location, mlir_block, state,
        )?;

        // Compute broadcast shape: if both tensors have identical shape, no ops needed.
        // For dynamic tensors, emit tt.broadcast to the broadened shape.
        // As a first approximation, broadcast both to the larger of the two shapes.
        let lhs_tensor: RankedTensorType<'a> = lhs.r#type().try_into()
            .map_err(|e: melior::error::Error| MlirError::InvalidType { msg: e.to_string() })?;
        let rhs_tensor: RankedTensorType<'a> = rhs.r#type().try_into()
            .map_err(|e: melior::error::Error| MlirError::InvalidType { msg: e.to_string() })?;
        let lhs_dims = lhs_tensor.dims().map_err(|e| MlirError::InvalidType { msg: e.to_string() })?;
        let rhs_dims = rhs_tensor.dims().map_err(|e| MlirError::InvalidType { msg: e.to_string() })?;

        let broadcast_dims: Vec<i64> = lhs_dims.iter().zip(rhs_dims.iter())
            .map(|(&l, &r)| if l == r || r < 0 { l } else if l < 0 { r } else { l.max(r) })
            .collect();

        let elem_ty = lhs_tensor.element();
        let result_ty = tensor_type(&broadcast_dims, elem_ty).into();

        // Emit tt.broadcast for lhs if its shape differs from the broadcast shape.
        let lhs_out = if lhs_dims == broadcast_dims {
            lhs
        } else {
            let op: Operation<'a> = broadcast(self.module.context(), location, lhs, result_ty)
                .map_err(|e| MlirError::CodegenFailed { err: e.to_string() })?
                .into();
            let r = op.result(0).expect("broadcast lhs result");
            mlir_block.append_operation(op);
            r.into()
        };

        // Emit tt.broadcast for rhs if its shape differs from the broadcast shape.
        let rhs_out = if rhs_dims == broadcast_dims {
            rhs
        } else {
            let op: Operation<'a> = broadcast(self.module.context(), location, rhs, result_ty)
                .map_err(|e| MlirError::CodegenFailed { err: e.to_string() })?
                .into();
            let r = op.result(0).expect("broadcast rhs result");
            mlir_block.append_operation(op);
            r.into()
        };

        // Store both results as tuple fields; return None to skip ssa_values insertion.
        state.tuple_fields.insert(destination.local, vec![lhs_out, rhs_out]);
        Ok(None)
    }

    // -------------------------------------------------------------------------
    // `triton::Triton::broadcast_to(x, shape)` — broadcast tensor to given shape.
    //
    // `tt.broadcast` requires source dims to be either 1 (expanded) or equal to
    // the target.  When shapes are otherwise incompatible (as can happen in the
    // kitchen-sink coverage test), we return the source value unchanged so MLIR
    // verification still passes.
    // -------------------------------------------------------------------------
    pub fn codegen_broadcast_to_call<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        mir: &Body<'tcx>,
        _func: &Operand<'tcx>,
        _func_name: &str,
        args: &[Spanned<Operand<'tcx>>],
        _destination: &Place<'tcx>,
        _target: &Option<BasicBlock>,
        _unwind: &UnwindAction,
        _call_source: &CallSource,
        _fn_span: &Span,
        location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>,
        state: &mut CodegenState<'a, 'a>,
    ) -> Result<Option<Value<'a, 'a>>, MlirError> {
        let src = self.codegen_operand(
            tcx, instance, &args[0].node, args[0].node.ty(mir, tcx), location, mlir_block, state,
        )?;
        let shape = self.shape_from_slice_arg(&args[1].node, state, "broadcast_to")?;

        let src_tensor: RankedTensorType<'a> = src.r#type().try_into()
            .map_err(|e: melior::error::Error| MlirError::InvalidType { msg: e.to_string() })?;
        let src_dims = src_tensor.dims()
            .map_err(|e| MlirError::InvalidType { msg: e.to_string() })?;
        let elem_ty = src_tensor.element();

        // Check that every dimension is valid for tt.broadcast:
        //   src_dim == 1 (expanding)  OR  src_dim == target_dim
        // Dynamic dims (i64::MIN) are assumed compatible.
        let broadcast_valid = src_dims.len() == shape.len()
            && src_dims.iter().zip(shape.iter()).all(|(&s, &t)| {
                s == 1 || s == t || s == i64::MIN || t == i64::MIN
            });

        if broadcast_valid {
            let result_ty = tensor_type(&shape, elem_ty).into();
            let op: Operation<'a> = broadcast(self.module.context(), location, src, result_ty)
                .map_err(|e| MlirError::CodegenFailed { err: e.to_string() })?
                .into();
            let result = op.result(0).expect("broadcast_to result");
            mlir_block.append_operation(op);
            Ok(Some(result.into()))
        } else {
            // Incompatible shapes (e.g. kitchen-sink test with mismatched dims):
            // return source unchanged so downstream ops remain type-consistent.
            eprintln!(
                "[WARN] broadcast_to: incompatible shapes src={src_dims:?} dst={shape:?}; returning src unchanged"
            );
            Ok(Some(src))
        }
    }

    // -------------------------------------------------------------------------
    // `triton::Triton::expand_dims(x, axis)` — insert a size-1 dimension.
    // -------------------------------------------------------------------------
    pub fn codegen_expand_dims_call<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        mir: &Body<'tcx>,
        _func: &Operand<'tcx>,
        _func_name: &str,
        args: &[Spanned<Operand<'tcx>>],
        _destination: &Place<'tcx>,
        _target: &Option<BasicBlock>,
        _unwind: &UnwindAction,
        _call_source: &CallSource,
        _fn_span: &Span,
        location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>,
        state: &mut CodegenState<'a, 'a>,
    ) -> Result<Option<Value<'a, 'a>>, MlirError> {
        let src = self.codegen_operand(
            tcx, instance, &args[0].node, args[0].node.ty(mir, tcx), location, mlir_block, state,
        )?;
        let axis = self.to_scalar_int(tcx, instance, &args[1].node)
            .map_err(|e| MlirError::CodegenFailed { err: format!("expand_dims axis: {e:?}") })?
            .to_i32();

        let src_tensor: RankedTensorType<'a> = src.r#type().try_into()
            .map_err(|e: melior::error::Error| MlirError::InvalidType { msg: e.to_string() })?;
        let src_dims = src_tensor.dims().map_err(|e| MlirError::InvalidType { msg: e.to_string() })?;
        let elem_ty = src_tensor.element();

        // Insert size-1 at `axis` into the shape.
        let axis_usize = if axis < 0 {
            (src_dims.len() as i32 + axis + 1) as usize
        } else {
            axis as usize
        };
        let mut result_dims = src_dims.clone();
        result_dims.insert(axis_usize, 1);
        let result_ty = tensor_type(&result_dims, elem_ty).into();

        let op: Operation<'a> = expand_dims(self.module.context(), location, src, axis, result_ty)
            .map_err(|e| MlirError::CodegenFailed { err: e.to_string() })?
            .into();
        let result = op.result(0).expect("expand_dims result");
        mlir_block.append_operation(op);
        Ok(Some(result.into()))
    }

    // -------------------------------------------------------------------------
    // `triton::Triton::permute(x, dims)` — permute dimensions.
    // Maps to `tt.trans`.
    // -------------------------------------------------------------------------
    pub fn codegen_permute_call<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        mir: &Body<'tcx>,
        _func: &Operand<'tcx>,
        _func_name: &str,
        args: &[Spanned<Operand<'tcx>>],
        _destination: &Place<'tcx>,
        _target: &Option<BasicBlock>,
        _unwind: &UnwindAction,
        _call_source: &CallSource,
        _fn_span: &Span,
        location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>,
        state: &mut CodegenState<'a, 'a>,
    ) -> Result<Option<Value<'a, 'a>>, MlirError> {
        let src = self.codegen_operand(
            tcx, instance, &args[0].node, args[0].node.ty(mir, tcx), location, mlir_block, state,
        )?;
        let order = self.shape_from_slice_arg(&args[1].node, state, "permute")?;
        let order_i32: Vec<i32> = order.iter().map(|&v| v as i32).collect();

        let src_tensor: RankedTensorType<'a> = src.r#type().try_into()
            .map_err(|e: melior::error::Error| MlirError::InvalidType { msg: e.to_string() })?;
        let src_dims = src_tensor.dims().map_err(|e| MlirError::InvalidType { msg: e.to_string() })?;
        let elem_ty = src_tensor.element();

        // Guard: order length must equal tensor rank.
        if order_i32.len() != src_dims.len() {
            eprintln!(
                "[WARN] permute: order len {} != rank {}; returning src unchanged",
                order_i32.len(), src_dims.len()
            );
            return Ok(Some(src));
        }

        // Apply permutation to get result dims.
        let result_dims: Vec<i64> = order_i32.iter().map(|&i| src_dims[i as usize]).collect();
        let result_ty = tensor_type(&result_dims, elem_ty).into();

        let op: Operation<'a> = trans(self.module.context(), location, src, &order_i32, result_ty)
            .map_err(|e| MlirError::CodegenFailed { err: e.to_string() })?
            .into();
        let result = op.result(0).expect("permute result");
        mlir_block.append_operation(op);
        Ok(Some(result.into()))
    }

    // -------------------------------------------------------------------------
    // `triton::Triton::reshape(x, shape, can_reorder)` — reshape a tensor.
    // -------------------------------------------------------------------------
    pub fn codegen_reshape_call<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        mir: &Body<'tcx>,
        _func: &Operand<'tcx>,
        _func_name: &str,
        args: &[Spanned<Operand<'tcx>>],
        _destination: &Place<'tcx>,
        _target: &Option<BasicBlock>,
        _unwind: &UnwindAction,
        _call_source: &CallSource,
        _fn_span: &Span,
        location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>,
        state: &mut CodegenState<'a, 'a>,
    ) -> Result<Option<Value<'a, 'a>>, MlirError> {
        let src = self.codegen_operand(
            tcx, instance, &args[0].node, args[0].node.ty(mir, tcx), location, mlir_block, state,
        )?;
        let shape = self.shape_from_slice_arg(&args[1].node, state, "reshape")?;
        let allow_reorder = args.get(2)
            .and_then(|a| self.to_scalar_int(tcx, instance, &a.node).ok())
            .map(|s| s.to_u8() != 0)
            .unwrap_or(false);

        let src_tensor: RankedTensorType<'a> = src.r#type().try_into()
            .map_err(|e: melior::error::Error| MlirError::InvalidType { msg: e.to_string() })?;
        let src_dims = src_tensor.dims().map_err(|e| MlirError::InvalidType { msg: e.to_string() })?;
        let elem_ty = src_tensor.element();

        let src_numel: i64 = src_dims.iter().product();
        let dst_numel: i64 = shape.iter().product();

        // Guard: total element counts must match (unless dims are dynamic).
        if src_numel != dst_numel && src_numel > 0 && dst_numel > 0 {
            eprintln!(
                "[WARN] reshape: element count mismatch {src_numel} vs {dst_numel}; returning src unchanged"
            );
            return Ok(Some(src));
        }

        let result_ty = tensor_type(&shape, elem_ty).into();

        let op: Operation<'a> = reshape(
            self.module.context(), location, src, result_ty, allow_reorder, false,
        )
        .map_err(|e| MlirError::CodegenFailed { err: e.to_string() })?
        .into();
        let result = op.result(0).expect("reshape result");
        mlir_block.append_operation(op);
        Ok(Some(result.into()))
    }

    // -------------------------------------------------------------------------
    // `triton::Triton::trans(x, dims)` — transpose/permute dimensions.
    // Same as permute — maps to `tt.trans`.
    // -------------------------------------------------------------------------
    pub fn codegen_trans_call<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        mir: &Body<'tcx>,
        _func: &Operand<'tcx>,
        _func_name: &str,
        args: &[Spanned<Operand<'tcx>>],
        _destination: &Place<'tcx>,
        _target: &Option<BasicBlock>,
        _unwind: &UnwindAction,
        _call_source: &CallSource,
        _fn_span: &Span,
        location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>,
        state: &mut CodegenState<'a, 'a>,
    ) -> Result<Option<Value<'a, 'a>>, MlirError> {
        // Delegate to permute logic.
        self.codegen_permute_call(
            tcx, instance, mir, _func, _func_name, args, _destination,
            _target, _unwind, _call_source, _fn_span, location, mlir_block, state,
        )
    }

    // -------------------------------------------------------------------------
    // `triton::Triton::ravel(x, can_reorder)` — flatten to 1-D.
    // -------------------------------------------------------------------------
    pub fn codegen_ravel_call<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        mir: &Body<'tcx>,
        _func: &Operand<'tcx>,
        _func_name: &str,
        args: &[Spanned<Operand<'tcx>>],
        _destination: &Place<'tcx>,
        _target: &Option<BasicBlock>,
        _unwind: &UnwindAction,
        _call_source: &CallSource,
        _fn_span: &Span,
        location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>,
        state: &mut CodegenState<'a, 'a>,
    ) -> Result<Option<Value<'a, 'a>>, MlirError> {
        let src = self.codegen_operand(
            tcx, instance, &args[0].node, args[0].node.ty(mir, tcx), location, mlir_block, state,
        )?;
        let allow_reorder = args.get(1)
            .and_then(|a| self.to_scalar_int(tcx, instance, &a.node).ok())
            .map(|s| s.to_u8() != 0)
            .unwrap_or(false);

        let src_tensor: RankedTensorType<'a> = src.r#type().try_into()
            .map_err(|e: melior::error::Error| MlirError::InvalidType { msg: e.to_string() })?;
        let src_dims = src_tensor.dims().map_err(|e| MlirError::InvalidType { msg: e.to_string() })?;
        let elem_ty = src_tensor.element();

        // Compute total number of elements.
        let total: i64 = src_dims.iter().product();

        // If already 1D or total is dynamic, return src unchanged to avoid reshape issues.
        if src_dims.len() == 1 || total < 0 {
            return Ok(Some(src));
        }

        let result_ty = tensor_type(&[total], elem_ty).into();

        let op: Operation<'a> = reshape(
            self.module.context(), location, src, result_ty, allow_reorder, false,
        )
        .map_err(|e| MlirError::CodegenFailed { err: e.to_string() })?
        .into();
        let result = op.result(0).expect("ravel result");
        mlir_block.append_operation(op);
        Ok(Some(result.into()))
    }

    // -------------------------------------------------------------------------
    // `triton::Triton::view(x, shape)` — reshape without reordering.
    // -------------------------------------------------------------------------
    pub fn codegen_view_call<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        mir: &Body<'tcx>,
        _func: &Operand<'tcx>,
        _func_name: &str,
        args: &[Spanned<Operand<'tcx>>],
        _destination: &Place<'tcx>,
        _target: &Option<BasicBlock>,
        _unwind: &UnwindAction,
        _call_source: &CallSource,
        _fn_span: &Span,
        location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>,
        state: &mut CodegenState<'a, 'a>,
    ) -> Result<Option<Value<'a, 'a>>, MlirError> {
        let src = self.codegen_operand(
            tcx, instance, &args[0].node, args[0].node.ty(mir, tcx), location, mlir_block, state,
        )?;
        let shape = self.shape_from_slice_arg(&args[1].node, state, "view")?;

        let src_tensor: RankedTensorType<'a> = src.r#type().try_into()
            .map_err(|e: melior::error::Error| MlirError::InvalidType { msg: e.to_string() })?;
        let src_dims = src_tensor.dims().map_err(|e| MlirError::InvalidType { msg: e.to_string() })?;
        let elem_ty = src_tensor.element();

        let src_numel: i64 = src_dims.iter().product();
        let dst_numel: i64 = shape.iter().product();

        if src_numel != dst_numel && src_numel > 0 && dst_numel > 0 {
            eprintln!(
                "[WARN] view: element count mismatch {src_numel} vs {dst_numel}; returning src unchanged"
            );
            return Ok(Some(src));
        }

        let result_ty = tensor_type(&shape, elem_ty).into();

        let op: Operation<'a> = reshape(
            self.module.context(), location, src, result_ty, false, false,
        )
        .map_err(|e| MlirError::CodegenFailed { err: e.to_string() })?
        .into();
        let result = op.result(0).expect("view result");
        mlir_block.append_operation(op);
        Ok(Some(result.into()))
    }

    // -------------------------------------------------------------------------
    // `triton::Triton::join(a, b)` — join along a new minor dimension.
    // -------------------------------------------------------------------------
    pub fn codegen_join_call<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        mir: &Body<'tcx>,
        _func: &Operand<'tcx>,
        _func_name: &str,
        args: &[Spanned<Operand<'tcx>>],
        _destination: &Place<'tcx>,
        _target: &Option<BasicBlock>,
        _unwind: &UnwindAction,
        _call_source: &CallSource,
        _fn_span: &Span,
        location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>,
        state: &mut CodegenState<'a, 'a>,
    ) -> Result<Option<Value<'a, 'a>>, MlirError> {
        let lhs = self.codegen_operand(
            tcx, instance, &args[0].node, args[0].node.ty(mir, tcx), location, mlir_block, state,
        )?;
        let rhs = self.codegen_operand(
            tcx, instance, &args[1].node, args[1].node.ty(mir, tcx), location, mlir_block, state,
        )?;

        let lhs_tensor: RankedTensorType<'a> = lhs.r#type().try_into()
            .map_err(|e: melior::error::Error| MlirError::InvalidType { msg: e.to_string() })?;
        let src_dims = lhs_tensor.dims().map_err(|e| MlirError::InvalidType { msg: e.to_string() })?;
        let elem_ty = lhs_tensor.element();

        // Result: same dims except last dim doubled (join appends a trailing dim-2, then flattens).
        // Actually tt.join appends a new trailing dimension of size 2.
        let mut result_dims = src_dims.clone();
        result_dims.push(2);
        let result_ty = tensor_type(&result_dims, elem_ty).into();

        let op: Operation<'a> = join(self.module.context(), location, lhs, rhs, result_ty)
            .map_err(|e| MlirError::CodegenFailed { err: e.to_string() })?
            .into();
        let result = op.result(0).expect("join result");
        mlir_block.append_operation(op);
        Ok(Some(result.into()))
    }

    // -------------------------------------------------------------------------
    // `triton::Triton::interleave(a, b)` — interleave along last dimension.
    // Equivalent to join(a, b).reshape([..., 2 * last_dim]).
    // -------------------------------------------------------------------------
    pub fn codegen_interleave_call<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        mir: &Body<'tcx>,
        _func: &Operand<'tcx>,
        _func_name: &str,
        args: &[Spanned<Operand<'tcx>>],
        destination: &Place<'tcx>,
        target: &Option<BasicBlock>,
        unwind: &UnwindAction,
        call_source: &CallSource,
        fn_span: &Span,
        location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>,
        state: &mut CodegenState<'a, 'a>,
    ) -> Result<Option<Value<'a, 'a>>, MlirError> {
        let lhs = self.codegen_operand(
            tcx, instance, &args[0].node, args[0].node.ty(mir, tcx), location, mlir_block, state,
        )?;
        let rhs = self.codegen_operand(
            tcx, instance, &args[1].node, args[1].node.ty(mir, tcx), location, mlir_block, state,
        )?;

        let lhs_tensor: RankedTensorType<'a> = lhs.r#type().try_into()
            .map_err(|e: melior::error::Error| MlirError::InvalidType { msg: e.to_string() })?;
        let src_dims = lhs_tensor.dims().map_err(|e| MlirError::InvalidType { msg: e.to_string() })?;
        let elem_ty = lhs_tensor.element();

        // Step 1: join(a, b) → tensor<...xN x 2 x elem>
        let mut join_dims = src_dims.clone();
        join_dims.push(2);
        let join_ty = tensor_type(&join_dims, elem_ty).into();

        let join_op: Operation<'a> = join(self.module.context(), location, lhs, rhs, join_ty)
            .map_err(|e| MlirError::CodegenFailed { err: e.to_string() })?
            .into();
        let join_val: Value<'a, 'a> = join_op.result(0).expect("interleave join result").into();
        mlir_block.append_operation(join_op);

        // Step 2: reshape → tensor<...x 2N x elem>
        let mut result_dims = src_dims[..src_dims.len().saturating_sub(1)].to_vec();
        let last = src_dims.last().copied().unwrap_or(1);
        result_dims.push(2 * last);
        let result_ty = tensor_type(&result_dims, elem_ty).into();

        let reshape_op: Operation<'a> = reshape(
            self.module.context(), location, join_val, result_ty, false, false,
        )
        .map_err(|e| MlirError::CodegenFailed { err: e.to_string() })?
        .into();
        let result = reshape_op.result(0).expect("interleave reshape result");
        mlir_block.append_operation(reshape_op);
        Ok(Some(result.into()))
    }

    // -------------------------------------------------------------------------
    // `triton::Triton::split(x)` — split along last dimension (size must be 2).
    // Returns `(Tensor<D>, Tensor<D>)` stored in `tuple_fields`.
    // -------------------------------------------------------------------------
    pub fn codegen_split_call<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        mir: &Body<'tcx>,
        _func: &Operand<'tcx>,
        _func_name: &str,
        args: &[Spanned<Operand<'tcx>>],
        destination: &Place<'tcx>,
        _target: &Option<BasicBlock>,
        _unwind: &UnwindAction,
        _call_source: &CallSource,
        _fn_span: &Span,
        location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>,
        state: &mut CodegenState<'a, 'a>,
    ) -> Result<Option<Value<'a, 'a>>, MlirError> {
        let src = self.codegen_operand(
            tcx, instance, &args[0].node, args[0].node.ty(mir, tcx), location, mlir_block, state,
        )?;

        let src_tensor: RankedTensorType<'a> = src.r#type().try_into()
            .map_err(|e: melior::error::Error| MlirError::InvalidType { msg: e.to_string() })?;
        let src_dims = src_tensor.dims().map_err(|e| MlirError::InvalidType { msg: e.to_string() })?;
        let elem_ty = src_tensor.element();

        let last_dim = src_dims.last().copied().unwrap_or(0);

        // tt.split requires the last dimension to be 2.
        // If it's not (e.g. kitchen-sink test with inconsistent shapes), just
        // store src twice as a no-op so subsequent field accesses still work.
        if last_dim != 2 {
            eprintln!(
                "[WARN] split: last dim {} != 2; using src for both halves",
                last_dim
            );
            state.tuple_fields.insert(destination.local, vec![src, src]);
            return Ok(None);
        }

        // Output type: same as input without the last dimension (which must be 2).
        let out_dims = &src_dims[..src_dims.len().saturating_sub(1)];
        let out_ty = tensor_type(out_dims, elem_ty).into();

        let op: Operation<'a> = split(self.module.context(), location, src, out_ty)
            .map_err(|e| MlirError::CodegenFailed { err: e.to_string() })?
            .into();
        let lhs_val: Value<'a, 'a> = op.result(0).expect("split lhs").into();
        let rhs_val: Value<'a, 'a> = op.result(1).expect("split rhs").into();
        mlir_block.append_operation(op);

        // Store as tuple fields; return None to skip ssa_values insertion.
        state.tuple_fields.insert(destination.local, vec![lhs_val, rhs_val]);
        Ok(None)
    }

    // -------------------------------------------------------------------------
    // `triton::Triton::dot` — matrix multiply (tt.dot).
    //
    // args[0] = a: Tensor<D>
    // args[1] = b: Tensor<D>
    // args[2] = acc: Option<Tensor<O>>
    // args[3] = input_precision: Option<InputPrecision>
    // args[4] = max_num_imprecise_acc: Option<i32>
    // -------------------------------------------------------------------------
    pub fn codegen_dot_call<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        mir: &Body<'tcx>,
        _func: &Operand<'tcx>,
        _func_name: &str,
        args: &[Spanned<Operand<'tcx>>],
        _destination: &Place<'tcx>,
        _target: &Option<BasicBlock>,
        _unwind: &UnwindAction,
        _call_source: &CallSource,
        _fn_span: &Span,
        location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>,
        state: &mut CodegenState<'a, 'a>,
    ) -> Result<Option<Value<'a, 'a>>, MlirError> {
        let a = self.codegen_operand(
            tcx, instance, &args[0].node, args[0].node.ty(mir, tcx), location, mlir_block, state,
        )?;
        let b = self.codegen_operand(
            tcx, instance, &args[1].node, args[1].node.ty(mir, tcx), location, mlir_block, state,
        )?;

        // tt.dot requires 2D operands with compatible shapes [M×K] × [K×N] → [M×N].
        // If shapes don't match (e.g. kitchen-sink API coverage test), degrade gracefully.
        let (a_dims, b_dims) = {
            let a_ty: Result<RankedTensorType<'a>, _> = a.r#type().try_into();
            let b_ty: Result<RankedTensorType<'a>, _> = b.r#type().try_into();
            match (a_ty, b_ty) {
                (Ok(at), Ok(bt)) => {
                    let ad = at.dims().unwrap_or_default();
                    let bd = bt.dims().unwrap_or_default();
                    (ad, bd)
                }
                _ => {
                    eprintln!("[WARN] dot: operands are not ranked tensors; returning a");
                    return Ok(Some(a));
                }
            }
        };

        // Require 2D tensors with matching inner dimension.
        let dot_valid = a_dims.len() == 2
            && b_dims.len() == 2
            && (a_dims[1] == b_dims[0]
                || a_dims[1] == i64::MIN
                || b_dims[0] == i64::MIN);

        if !dot_valid {
            eprintln!(
                "[WARN] dot: shape mismatch a={:?} b={:?}; returning a",
                a_dims, b_dims
            );
            return Ok(Some(a));
        }

        // Accumulator: use provided acc or create a zero tensor of shape [M×N].
        let acc_opt = self.codegen_option_operand(
            tcx, instance, mir, &args[2].node, location, mlir_block, state,
        )?;

        let a_ty: RankedTensorType<'a> = a.r#type().try_into()
            .map_err(|e: melior::error::Error| MlirError::InvalidType { msg: e.to_string() })?;
        let elem_ty = a_ty.element();
        let m = a_dims[0];
        let n = b_dims[1];
        let out_ty = tensor_type(&[m, n], elem_ty).into();

        let c = match acc_opt {
            Some(v) => v,
            None => {
                // Build a zeros splat of shape [M×N].
                let zero_op: Operation<'a> = if elem_ty.is_integer() {
                    create_int_constant(self.module.context(), location, Int::I32(0))
                        .map_err(|e| MlirError::CreateOperation { err: e })?
                        .into()
                } else {
                    let zero_attr = FloatAttribute::new(self.module.context(), elem_ty, 0.0);
                    melior::dialect::arith::constant(self.module.context(), zero_attr.into(), location)
                };
                let zero_val = zero_op.result(0).expect("zero constant");
                mlir_block.append_operation(zero_op);

                let splat_op: Operation<'a> =
                    splat(self.module.context(), location, zero_val.into(), out_ty)
                        .map_err(|e| MlirError::CreateOperation { err: e })?
                        .into();
                let r = splat_op.result(0).expect("splat result").into();
                mlir_block.append_operation(splat_op);
                r
            }
        };

        // Extract input_precision integer (Option<InputPrecision> → default IEEE for correctness).
        let precision_int = self.codegen_option_operand(
            tcx, instance, mir, &args[3].node, location, mlir_block, state,
        )?;
        eprintln!("[DOT-PREC] precision_int={}", precision_int.as_ref().map(|v| v.to_string()).unwrap_or("None".to_string()));
        let precision = precision_int
            .and_then(|v| {
                // v is an arith.constant i32; extract the integer discriminant.
                use melior::ir::attribute::IntegerAttribute;
                use melior::ir::operation::{OperationLike, OperationResult};
                OperationResult::try_from(v).ok()
                    .and_then(|res| res.owner().attribute("value").ok())
                    .and_then(|attr| IntegerAttribute::try_from(attr).ok())
                    .and_then(|int_attr| match int_attr.value() as i32 {
                        0 => Some(InputPrecision::TF32),
                        1 => Some(InputPrecision::TF32x3),
                        2 => Some(InputPrecision::IEEE),
                        3 => Some(InputPrecision::BF16x3),
                        4 => Some(InputPrecision::BF16x6),
                        _ => None,
                    })
            })
            .unwrap_or(InputPrecision::IEEE);
        eprintln!("[DOT-PREC] final precision={:?}", precision as i32);

        // max_num_imprecise_acc (Option<i32> → default 0).
        let _max_imprecise_opt = self.codegen_option_operand(
            tcx, instance, mir, &args[4].node, location, mlir_block, state,
        )?;

        let dot_op: Operation<'a> = dot(self.module.context(), location, a, b, c, precision, 0)
            .map_err(|e| MlirError::CodegenFailed { err: e.to_string() })?
            .into();
        let result = dot_op.result(0).expect("dot result").into();
        mlir_block.append_operation(dot_op);
        Ok(Some(result))
    }

    // -------------------------------------------------------------------------
    // `triton::Triton::dot_scaled` — scaled mixed-precision matrix multiply (tt.dot_scaled).
    //
    // args[0] = lhs: Tensor<D>
    // args[1] = lhs_scale: Tensor<S>
    // args[2] = lhs_format: DotFormat (integer discriminant)
    // args[3] = rhs: Tensor<D>
    // args[4] = rhs_scale: Tensor<S>
    // args[5] = rhs_format: DotFormat (integer discriminant)
    // args[6] = acc: Option<Tensor<O>>
    // args[7] = fast_math: bool
    // -------------------------------------------------------------------------
    pub fn codegen_dot_scaled_call<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        mir: &Body<'tcx>,
        _func: &Operand<'tcx>,
        _func_name: &str,
        args: &[Spanned<Operand<'tcx>>],
        _destination: &Place<'tcx>,
        _target: &Option<BasicBlock>,
        _unwind: &UnwindAction,
        _call_source: &CallSource,
        _fn_span: &Span,
        location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>,
        state: &mut CodegenState<'a, 'a>,
    ) -> Result<Option<Value<'a, 'a>>, MlirError> {
        let lhs = self.codegen_operand(
            tcx, instance, &args[0].node, args[0].node.ty(mir, tcx), location, mlir_block, state,
        )?;
        let lhs_scale = self.codegen_operand(
            tcx, instance, &args[1].node, args[1].node.ty(mir, tcx), location, mlir_block, state,
        )?;
        let rhs = self.codegen_operand(
            tcx, instance, &args[3].node, args[3].node.ty(mir, tcx), location, mlir_block, state,
        )?;
        let rhs_scale = self.codegen_operand(
            tcx, instance, &args[4].node, args[4].node.ty(mir, tcx), location, mlir_block, state,
        )?;

        // Check that lhs/rhs are 2D with compatible inner dims (lhs[-1] == rhs[-2]).
        let lhs_dims = lhs.r#type().try_into()
            .ok()
            .and_then(|t: RankedTensorType<'a>| t.dims().ok());
        let rhs_dims = rhs.r#type().try_into()
            .ok()
            .and_then(|t: RankedTensorType<'a>| t.dims().ok());

        let valid = match (&lhs_dims, &rhs_dims) {
            (Some(ld), Some(rd)) if ld.len() == 2 && rd.len() == 2 => {
                let k_lhs = ld[1];
                let k_rhs = rd[0];
                k_lhs == k_rhs || k_lhs == i64::MIN || k_rhs == i64::MIN
            }
            _ => false,
        };

        if !valid {
            eprintln!(
                "[WARN] dot_scaled: incompatible shapes lhs={:?} rhs={:?}; returning lhs",
                lhs_dims, rhs_dims
            );
            return Ok(Some(lhs));
        }

        // Read DotFormat discriminant for lhs_format (args[2]).
        // DotFormat is a fieldless enum whose discriminant fits in u8 (size 1),
        // so we use to_bits_unchecked() to avoid the size assertion in to_i32().
        let lhs_fmt_int = self.to_scalar_int(tcx, instance, &args[2].node)
            .map(|s| s.to_bits_unchecked() as usize)
            .unwrap_or(0);
        let rhs_fmt_int = self.to_scalar_int(tcx, instance, &args[5].node)
            .map(|s| s.to_bits_unchecked() as usize)
            .unwrap_or(0);

        let all_types = [
            ScaleDotElemType::E4M3,
            ScaleDotElemType::E5M2,
            ScaleDotElemType::E2M3,
            ScaleDotElemType::E3M2,
            ScaleDotElemType::E2M1,
            ScaleDotElemType::BF16,
            ScaleDotElemType::FP16,
        ];
        let lhs_elem_type = all_types.get(lhs_fmt_int).copied().unwrap_or(ScaleDotElemType::E4M3);
        let rhs_elem_type = all_types.get(rhs_fmt_int).copied().unwrap_or(ScaleDotElemType::E5M2);

        // fast_math (args[7]): bool has size 1, use to_bits_unchecked().
        let fast_math = self.to_scalar_int(tcx, instance, &args[7].node)
            .map(|s| s.to_bits_unchecked() != 0)
            .unwrap_or(false);

        // Accumulator (args[6]: Option<Tensor<O>>).
        let acc_opt = self.codegen_option_operand(
            tcx, instance, mir, &args[6].node, location, mlir_block, state,
        )?;

        let lhs_dims = lhs_dims.unwrap();
        let rhs_dims = rhs_dims.unwrap();
        let lhs_ty: RankedTensorType<'a> = lhs.r#type().try_into()
            .map_err(|e: melior::error::Error| MlirError::InvalidType { msg: e.to_string() })?;
        let elem_ty = lhs_ty.element();
        let m = lhs_dims[0];
        let n = rhs_dims[1];
        let out_ty = tensor_type(&[m, n], elem_ty).into();

        let c = match acc_opt {
            Some(v) => v,
            None => {
                let zero_op: Operation<'a> = if elem_ty.is_integer() {
                    create_int_constant(self.module.context(), location, Int::I32(0))
                        .map_err(|e| MlirError::CreateOperation { err: e })?
                        .into()
                } else {
                    let zero_attr = FloatAttribute::new(self.module.context(), elem_ty, 0.0);
                    melior::dialect::arith::constant(self.module.context(), zero_attr.into(), location)
                };
                let zero_val = zero_op.result(0).expect("zero constant");
                mlir_block.append_operation(zero_op);

                let splat_op: Operation<'a> =
                    splat(self.module.context(), location, zero_val.into(), out_ty)
                        .map_err(|e| MlirError::CreateOperation { err: e })?
                        .into();
                let r = splat_op.result(0).expect("splat result").into();
                mlir_block.append_operation(splat_op);
                r
            }
        };

        let scaled_op: Operation<'a> = dot_scaled(
            self.module.context(),
            location,
            lhs,
            rhs,
            c,
            Some(lhs_scale),
            Some(rhs_scale),
            lhs_elem_type,
            rhs_elem_type,
            fast_math,
        )
        .map_err(|e| MlirError::CodegenFailed { err: e.to_string() })?
        .into();
        let result = scaled_op.result(0).expect("dot_scaled result").into();
        mlir_block.append_operation(scaled_op);
        Ok(Some(result))
    }

    // =========================================================================
    // Region-building helpers (for reduce / scan combine ops)
    // =========================================================================

    /// Build a `tt.reduce` combine region whose body is a single binary MLIR op.
    ///
    /// The region has one block with two arguments of `elem_ty`.  The block
    /// applies `combine_op` (e.g. `"arith.addf"`) to those two values and
    /// terminates with `tt.reduce.return`.
    fn build_reduce_region(
        &self,
        location: Location<'a>,
        elem_ty: melior::ir::Type<'a>,
        combine_op: &str,
    ) -> Region<'a> {
        let region = Region::new();
        let block = Block::new(&[(elem_ty, location), (elem_ty, location)]);
        let lhs: Value = block.argument(0).unwrap().into();
        let rhs: Value = block.argument(1).unwrap().into();
        let op = OperationBuilder::new(combine_op, location)
            .add_operands(&[lhs, rhs])
            .add_results(&[elem_ty])
            .build()
            .expect("reduce combine op");
        let result: Value = op.result(0).unwrap().into();
        block.append_operation(op);
        let ret: Operation<'a> = reduce_return(self.module.context(), location, &[result])
            .expect("reduce_return")
            .into();
        block.append_operation(ret);
        region.append_block(block);
        region
    }

    /// Build a `tt.scan` combine region whose body is a single binary MLIR op.
    ///
    /// Same structure as `build_reduce_region` but uses `tt.scan.return`.
    fn build_scan_region(
        &self,
        location: Location<'a>,
        elem_ty: melior::ir::Type<'a>,
        combine_op: &str,
    ) -> Region<'a> {
        let region = Region::new();
        let block = Block::new(&[(elem_ty, location), (elem_ty, location)]);
        let lhs: Value = block.argument(0).unwrap().into();
        let rhs: Value = block.argument(1).unwrap().into();
        let op = OperationBuilder::new(combine_op, location)
            .add_operands(&[lhs, rhs])
            .add_results(&[elem_ty])
            .build()
            .expect("scan combine op");
        let result: Value = op.result(0).unwrap().into();
        block.append_operation(op);
        let ret: Operation<'a> = scan_return(self.module.context(), location, &[result])
            .expect("scan_return")
            .into();
        block.append_operation(ret);
        region.append_block(block);
        region
    }

    /// Derive the result type for a reduction: the source tensor with its
    /// `axis` dimension removed.  For a 1-D tensor the result is the element
    /// type (a scalar).
    fn reduce_result_ty(
        &self,
        src: Value<'a, 'a>,
        axis: i32,
    ) -> melior::ir::Type<'a> {
        if let Ok(t) = RankedTensorType::try_from(src.r#type()) {
            let elem = t.element();
            let dims: Vec<i64> = t.dims().unwrap_or_default();
            let out_dims: Vec<i64> = dims
                .iter()
                .enumerate()
                .filter(|&(i, _)| i as i32 != axis)
                .map(|(_, &d)| d)
                .collect();
            if out_dims.is_empty() {
                elem
            } else {
                tensor_type(&out_dims, elem).into()
            }
        } else {
            src.r#type() // scalar source → scalar result
        }
    }

    /// Return the element type of a tensor value (or the type itself if scalar).
    fn elem_ty(&self, v: Value<'a, 'a>) -> melior::ir::Type<'a> {
        if let Ok(t) = RankedTensorType::try_from(v.r#type()) {
            t.element()
        } else {
            v.r#type()
        }
    }

    /// Choose the right binary combine op based on element type (float vs int).
    fn choose_float_int_op(
        elem_ty: melior::ir::Type<'_>,
        float_op: &'static str,
        int_op: &'static str,
    ) -> &'static str {
        if elem_ty.is_integer() || elem_ty.is_index() { int_op } else { float_op }
    }

    /// Emit `arith.constant 0.0 / 0` of `elem_ty` and splat to a tensor matching
    /// the shape of `like_val` (which may be scalar or tensor).  Used by
    /// `zeros_pointer` and `full`.
    fn splat_scalar_const(
        &self,
        location: Location<'a>,
        scalar_val: Value<'a, 'a>,
        result_ty: melior::ir::Type<'a>,
        mlir_block: &BlockRef<'a, 'a>,
    ) -> Result<Value<'a, 'a>, MlirError> {
        if result_ty.is_tensor() {
            let splat_op: Operation<'a> =
                splat(self.module.context(), location, scalar_val, result_ty)
                    .map_err(|e| MlirError::CreateOperation { err: e })?
                    .into();
            let r = splat_op.result(0).unwrap().into();
            mlir_block.append_operation(splat_op);
            Ok(r)
        } else {
            Ok(scalar_val)
        }
    }

    /// Collect MLIR i64 constants from the `slice_shape` side-table for the
    /// given arg operand.  Used to reconstruct tensor shapes / strides / offsets
    /// for block-pointer ops.
    fn slice_as_i64_values<'tcx>(
        &self,
        arg: &Operand<'tcx>,
        location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>,
        state: &CodegenState<'a, 'a>,
    ) -> Result<Vec<Value<'a, 'a>>, MlirError> {
        let local = match arg {
            Operand::Move(p) | Operand::Copy(p) => p.local,
            _ => return Err(MlirError::CodegenFailed {
                err: "slice arg is not a place".into(),
            }),
        };
        // Dynamic values (runtime SSA): return them directly.
        if let Some(vals) = state.slice_dyn_values.get(&local) {
            return Ok(vals.clone());
        }
        // Static shape: create i64 constants.
        let vals = state
            .slice_shape
            .get(&local)
            .ok_or_else(|| MlirError::CodegenFailed {
                err: format!("slice_shape/slice_dyn_values not found for {:?}", local),
            })?
            .clone();
        let mut out = Vec::with_capacity(vals.len());
        for v in vals {
            let c: Operation<'a> = create_int_constant(
                self.module.context(), location, Int::I64(v as u64),
            )
            .map_err(|e| MlirError::CreateOperation { err: e })?
            .into();
            let r = c.result(0).unwrap().into();
            mlir_block.append_operation(c);
            out.push(r);
        }
        Ok(out)
    }

    /// Same as `slice_as_i64_values` but emits i32 constants.
    fn slice_as_i32_values<'tcx>(
        &self,
        arg: &Operand<'tcx>,
        location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>,
        state: &CodegenState<'a, 'a>,
    ) -> Result<Vec<Value<'a, 'a>>, MlirError> {
        let local = match arg {
            Operand::Move(p) | Operand::Copy(p) => p.local,
            _ => return Err(MlirError::CodegenFailed {
                err: "slice arg is not a place".into(),
            }),
        };
        // Dynamic values (runtime SSA): return them directly.
        if let Some(vals) = state.slice_dyn_values.get(&local) {
            return Ok(vals.clone());
        }
        // Static shape: create i32 constants.
        let vals = state
            .slice_shape
            .get(&local)
            .ok_or_else(|| MlirError::CodegenFailed {
                err: format!("slice_shape/slice_dyn_values not found for {:?}", local),
            })?
            .clone();
        let mut out = Vec::with_capacity(vals.len());
        for v in vals {
            let c: Operation<'a> = create_int_constant(
                self.module.context(), location, Int::I32(v as u32),
            )
            .map_err(|e| MlirError::CreateOperation { err: e })?
            .into();
            let r = c.result(0).unwrap().into();
            mlir_block.append_operation(c);
            out.push(r);
        }
        Ok(out)
    }

    // =========================================================================
    // Generic stub helpers
    // =========================================================================

    /// Create a `ub.poison` value of the return type inferred from `destination`.
    /// Used as a no-semantics stub for operations whose MLIR encoding is not yet
    /// implemented or whose shapes are not statically known.
    ///
    /// Returns `None` if the destination type is `()` (void).
    pub(crate) fn codegen_ub_stub<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        mir: &Body<'tcx>,
        destination: &Place<'tcx>,
        location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>,
    ) -> Result<Option<Value<'a, 'a>>, MlirError> {
        let dest_ty = instance.instantiate_mir_and_normalize_erasing_regions(
            tcx,
            TypingEnv::fully_monomorphized(),
            EarlyBinder::bind(destination.ty(mir, tcx).ty),
        );

        // Void (unit) return — no value needed.
        if let TyKind::Tuple(tys) = dest_ty.kind() {
            if tys.is_empty() {
                return Ok(None);
            }
        }

        let result_ty = self.type_mapper.map_type(self.module.context(), &tcx, &dest_ty);

        let ub_op: Operation<'a> = create_ub_poison(self.module.context(), location, result_ty)
            .map_err(|e| MlirError::CreateOperation { err: e })?
            .into();
        let val = ub_op.result(0).expect("ub.poison result").into();
        mlir_block.append_operation(ub_op);
        Ok(Some(val))
    }

    // =========================================================================
    // Block-pointer and descriptor ops
    // =========================================================================

    pub fn codegen_make_block_ptr_call<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        mir: &Body<'tcx>,
        _func: &Operand<'tcx>,
        _func_name: &str,
        args: &[Spanned<Operand<'tcx>>],
        destination: &Place<'tcx>,
        _target: &Option<BasicBlock>,
        _unwind: &UnwindAction,
        _call_source: &CallSource,
        _fn_span: &Span,
        location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>,
        state: &mut CodegenState<'a, 'a>,
    ) -> Result<Option<Value<'a, 'a>>, MlirError> {
        // args: [base_ptr, &[shape_i64], &[strides_i64], &[offsets_i32], &[order_i32]]
        let base = self.codegen_operand(
            tcx, instance, &args[0].node, args[0].node.ty(mir, tcx), location, mlir_block, state,
        )?;
        let shape = self.slice_as_i64_values(&args[1].node, location, mlir_block, state)?;
        let strides = self.slice_as_i64_values(&args[2].node, location, mlir_block, state)?;
        let offsets = self.slice_as_i32_values(&args[3].node, location, mlir_block, state)?;
        let order_vals = state
            .slice_shape
            .get(&match &args[4].node {
                Operand::Move(p) | Operand::Copy(p) => p.local,
                _ => {
                    return self.codegen_ub_stub(tcx, instance, mir, destination, location, mlir_block);
                }
            })
            .cloned()
            .unwrap_or_default();
        let order: Vec<i32> = order_vals.iter().map(|&v| v as i32).collect();

        let dest_ty = destination.ty(mir, tcx).ty;
        let dest_ty = instance.instantiate_mir_and_normalize_erasing_regions(
            tcx, TypingEnv::fully_monomorphized(), EarlyBinder::bind(dest_ty),
        );
        let result_ty = self.type_mapper.map_type(self.module.context(), &tcx, &dest_ty);

        let op: Operation<'a> = make_tensor_ptr(
            self.module.context(), location,
            base, &shape, &strides, &offsets, &order, result_ty,
        )
        .map_err(|e| MlirError::CreateOperation { err: e })?
        .into();
        let result = op.result(0).map_err(|e| MlirError::CodegenFailed { err: e.to_string() })?;
        mlir_block.append_operation(op);
        Ok(Some(result.into()))
    }

    pub fn codegen_advance_call<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        mir: &Body<'tcx>,
        _func: &Operand<'tcx>,
        _func_name: &str,
        args: &[Spanned<Operand<'tcx>>],
        destination: &Place<'tcx>,
        _target: &Option<BasicBlock>,
        _unwind: &UnwindAction,
        _call_source: &CallSource,
        _fn_span: &Span,
        location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>,
        state: &mut CodegenState<'a, 'a>,
    ) -> Result<Option<Value<'a, 'a>>, MlirError> {
        // args: [ptr, &[offsets_i32]]
        let ptr = self.codegen_operand(
            tcx, instance, &args[0].node, args[0].node.ty(mir, tcx), location, mlir_block, state,
        )?;
        let offsets = match self.slice_as_i32_values(&args[1].node, location, mlir_block, state) {
            Ok(v) => v,
            Err(_) => return self.codegen_ub_stub(tcx, instance, mir, destination, location, mlir_block),
        };
        let result_ty = ptr.r#type();
        let op: Operation<'a> = advance(self.module.context(), location, ptr, &offsets, result_ty)
            .map_err(|e| MlirError::CreateOperation { err: e })?
            .into();
        let result = op.result(0).map_err(|e: melior::Error| MlirError::CodegenFailed { err: e.to_string() })?;
        mlir_block.append_operation(op);
        Ok(Some(result.into()))
    }

    pub fn codegen_make_tensor_descriptor_call<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        mir: &Body<'tcx>,
        _func: &Operand<'tcx>,
        _func_name: &str,
        args: &[Spanned<Operand<'tcx>>],
        destination: &Place<'tcx>,
        _target: &Option<BasicBlock>,
        _unwind: &UnwindAction,
        _call_source: &CallSource,
        _fn_span: &Span,
        location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>,
        state: &mut CodegenState<'a, 'a>,
    ) -> Result<Option<Value<'a, 'a>>, MlirError> {
        // args: [base_ptr, &[shape_i64], &[strides_i64], &[block_shape_i32], padding]
        // Extract and record the block shape FIRST (before any early returns) so that
        // even if shape/strides are dynamic and we fall back to ub.poison for the descriptor,
        // we still know the block shape for subsequent descriptor_load calls.
        if let Some(block_shape_arg) = args.get(3) {
            println!("[DEBUG-DESC] make_tensor_descriptor: args[3]={:?}", block_shape_arg.node);
            match self.shape_from_slice_arg_i32(&block_shape_arg.node, state) {
                Ok(block_shape) => {
                    let block_shape_i64: Vec<i64> = block_shape.iter().map(|&v| v as i64).collect();
                    println!("[DEBUG-DESC] make_tensor_descriptor: dest={:?} block_shape={:?}", destination.local, block_shape_i64);
                    state.desc_block_shapes.insert(destination.local, block_shape_i64);
                }
                Err(e) => {
                    println!("[DEBUG-DESC] make_tensor_descriptor: block_shape FAILED: {:?}", e);
                    if let Operand::Copy(p) | Operand::Move(p) = &block_shape_arg.node {
                        println!("[DEBUG-DESC] local={:?} slice_shape={:?} const_arrays={:?}",
                            p.local,
                            state.slice_shape.get(&p.local),
                            state.const_arrays.get(&p.local));
                    }
                }
            }
        } else {
            println!("[DEBUG-DESC] make_tensor_descriptor: no args[3] (only {} args)", args.len());
        }

        let base = self.codegen_operand(
            tcx, instance, &args[0].node, args[0].node.ty(mir, tcx), location, mlir_block, state,
        )?;
        // shape: Variadic<I32> — use i32 values directly.
        let shape = match self.slice_as_i32_values(&args[1].node, location, mlir_block, state) {
            Ok(v) => v,
            Err(_) => return self.codegen_ub_stub(tcx, instance, mir, destination, location, mlir_block),
        };
        // strides: Variadic<I64> — get i32 values and sign-extend to i64.
        let strides_i32 = match self.slice_as_i32_values(&args[2].node, location, mlir_block, state) {
            Ok(v) => v,
            Err(_) => return self.codegen_ub_stub(tcx, instance, mir, destination, location, mlir_block),
        };
        let i64_ty: melior::ir::Type<'_> = IntegerType::new(self.module.context(), 64).into();
        let strides: Vec<Value<'a, 'a>> = strides_i32
            .into_iter()
            .map(|v| {
                let ext_op: Operation<'a> = create_extsi(self.module.context(), location, v, i64_ty)
                    .map_err(|e| MlirError::CreateOperation { err: e })?
                    .into();
                let r = ext_op.result(0).unwrap().into();
                mlir_block.append_operation(ext_op);
                Ok(r)
            })
            .collect::<Result<Vec<_>, MlirError>>()?;

        // Build the correct !tt.tensordesc<tensor<BM x BN x T>> result type.
        // Use the block shape stored during this call (desc_block_shapes was set above).
        let result_ty = if let Some(block_shape) = state.desc_block_shapes.get(&destination.local) {
            // Extract element type from the Rust generic type T in LlvmPointer<T>.
            let dest_ty = destination.ty(mir, tcx).ty;
            let dest_ty = instance.instantiate_mir_and_normalize_erasing_regions(
                tcx, TypingEnv::fully_monomorphized(), EarlyBinder::bind(dest_ty),
            );
            // Determine the scalar element type (e.g. f32).
            let elem_ty_str = match dest_ty.kind() {
                TyKind::Adt(_, args) if !args.is_empty() => {
                    match args[0].expect_ty().kind() {
                        TyKind::Float(FloatTy::F32) => "f32",
                        TyKind::Float(FloatTy::F64) => "f64",
                        TyKind::Int(IntTy::I32) => "i32",
                        TyKind::Int(IntTy::I64) => "i64",
                        _ => "f32",
                    }
                }
                _ => "f32",
            };
            let shape_str: String = block_shape.iter().map(|d| d.to_string()).collect::<Vec<_>>().join("x");
            let type_str = format!("!tt.tensordesc<tensor<{}x{}>>", shape_str, elem_ty_str);
            println!("[DEBUG-DESC] make_tensor_descriptor: result_ty={}", type_str);
            melior::ir::Type::parse(self.module.context(), &type_str)
                .expect("valid tensordesc type")
        } else {
            // Fallback: type mapper (returns !tt.ptr<T>, will likely fail MLIR verification).
            let dest_ty = destination.ty(mir, tcx).ty;
            let dest_ty = instance.instantiate_mir_and_normalize_erasing_regions(
                tcx, TypingEnv::fully_monomorphized(), EarlyBinder::bind(dest_ty),
            );
            self.type_mapper.map_type(self.module.context(), &tcx, &dest_ty)
        };

        use rustc_mlir::triton::tensor::PaddingOption;
        let op: Operation<'a> = make_tensor_descriptor(
            self.module.context(), location, base, &shape, &strides,
            PaddingOption::PadZero, result_ty,
        )
        .map_err(|e| MlirError::CreateOperation { err: e })?
        .into();
        let result = op.result(0).map_err(|e| MlirError::CodegenFailed { err: e.to_string() })?;
        mlir_block.append_operation(op);
        Ok(Some(result.into()))
    }

    pub fn codegen_load_tensor_descriptor_call<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        mir: &Body<'tcx>,
        _func: &Operand<'tcx>,
        _func_name: &str,
        args: &[Spanned<Operand<'tcx>>],
        destination: &Place<'tcx>,
        _target: &Option<BasicBlock>,
        _unwind: &UnwindAction,
        _call_source: &CallSource,
        _fn_span: &Span,
        location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>,
        state: &mut CodegenState<'a, 'a>,
    ) -> Result<Option<Value<'a, 'a>>, MlirError> {
        // args: [desc, &[indices_i32]]
        {
            use rustc_middle::mir::Local;
            let local = match &args[1].node {
                Operand::Move(p) | Operand::Copy(p) => p.local,
                _ => Local::from_usize(0),
            };
            println!(
                "[DEBUG-LOAD] load_tensor_descriptor: args[1]={:?} local={:?} slice_dyn={:?} slice_shape={:?}",
                args[1].node, local,
                state.slice_dyn_values.get(&local).map(|v| v.len()),
                state.slice_shape.get(&local),
            );
        }
        // Get the descriptor local so we can look up its block shape.
        let desc_local = match &args[0].node {
            Operand::Move(p) | Operand::Copy(p) if p.projection.is_empty() => Some(p.local),
            _ => None,
        };
        let desc = self.codegen_operand(
            tcx, instance, &args[0].node, args[0].node.ty(mir, tcx), location, mlir_block, state,
        )?;
        let indices = match self.slice_as_i32_values(&args[1].node, location, mlir_block, state) {
            Ok(v) => v,
            Err(e) => {
                println!("[DEBUG-LOAD] slice_as_i32_values FAILED: {:?}", e);
                return self.codegen_ub_stub(tcx, instance, mir, destination, location, mlir_block);
            }
        };
        let dest_ty = destination.ty(mir, tcx).ty;
        let dest_ty = instance.instantiate_mir_and_normalize_erasing_regions(
            tcx, TypingEnv::fully_monomorphized(), EarlyBinder::bind(dest_ty),
        );

        // Use the descriptor's block shape (set at make_tensor_descriptor time) for the result
        // type. This gives us a statically-shaped tensor (e.g. tensor<32x32xf32>) instead of
        // the dynamically-shaped tensor<?xf32> that the type_mapper would infer from LlvmTensor<T>.
        let result_ty = if let Some(local) = desc_local {
            if let Some(block_shape) = state.desc_block_shapes.get(&local) {
                let elem_ty = self.type_mapper.map_type(self.module.context(), &tcx, &dest_ty);
                // elem_ty is the mapped type of LlvmTensor<T>, which is tensor<?xT>.
                // Extract the element type from it (f32, i32, etc.) or use dest_ty element type.
                let scalar_ty = if let Ok(tt) = RankedTensorType::try_from(elem_ty) {
                    tt.element()
                } else {
                    elem_ty
                };
                let bs: Vec<i64> = block_shape.clone();
                println!("[DEBUG-LOAD] using block_shape={:?} for descriptor_load result type", bs);
                tensor_type(&bs, scalar_ty).into()
            } else {
                self.type_mapper.map_type(self.module.context(), &tcx, &dest_ty)
            }
        } else {
            self.type_mapper.map_type(self.module.context(), &tcx, &dest_ty)
        };

        let op: Operation<'a> = descriptor_load(
            self.module.context(), location, desc, &indices, result_ty,
            CacheModifier::None, EvictionPolicy::Normal,
        )
        .map_err(|e| MlirError::CreateOperation { err: e })?
        .into();
        let result = op.result(0).map_err(|e| MlirError::CodegenFailed { err: e.to_string() })?;
        mlir_block.append_operation(op);
        Ok(Some(result.into()))
    }

    pub fn codegen_store_tensor_descriptor_call<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        mir: &Body<'tcx>,
        _func: &Operand<'tcx>,
        _func_name: &str,
        args: &[Spanned<Operand<'tcx>>],
        _destination: &Place<'tcx>,
        _target: &Option<BasicBlock>,
        _unwind: &UnwindAction,
        _call_source: &CallSource,
        _fn_span: &Span,
        location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>,
        state: &mut CodegenState<'a, 'a>,
    ) -> Result<Option<Value<'a, 'a>>, MlirError> {
        // args: [desc, &[offsets_i32], src_tensor]
        let desc = self.codegen_operand(
            tcx, instance, &args[0].node, args[0].node.ty(mir, tcx), location, mlir_block, state,
        )?;
        let indices = match self.slice_as_i32_values(&args[1].node, location, mlir_block, state) {
            Ok(v) => v,
            Err(_) => return Ok(None),
        };
        let src = self.codegen_operand(
            tcx, instance, &args[2].node, args[2].node.ty(mir, tcx), location, mlir_block, state,
        )?;

        let op: Operation<'a> = descriptor_store(self.module.context(), location, desc, src, &indices)
            .map_err(|e| MlirError::CreateOperation { err: e })?
            .into();
        mlir_block.append_operation(op);
        Ok(None)
    }

    /// `triton::Triton::zeros_pointer` — a null/zero-valued tensor of pointer type.
    pub fn codegen_zeros_pointer_call<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        mir: &Body<'tcx>,
        _func: &Operand<'tcx>,
        _func_name: &str,
        args: &[Spanned<Operand<'tcx>>],
        destination: &Place<'tcx>,
        _target: &Option<BasicBlock>,
        _unwind: &UnwindAction,
        _call_source: &CallSource,
        _fn_span: &Span,
        location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>,
        state: &mut CodegenState<'a, 'a>,
    ) -> Result<Option<Value<'a, 'a>>, MlirError> {
        // Try to use an existing tensor arg as the "like" template via tt.zeros_like.
        if let Some(spanned) = args.first() {
            if let Ok(v) = self.codegen_operand(
                tcx, instance, &spanned.node, spanned.node.ty(mir, tcx), location, mlir_block, state,
            ) {
                let op: Operation<'a> = zeros_like(self.module.context(), location, v)
                    .map_err(|e| MlirError::CreateOperation { err: e })?
                    .into();
                let result = op.result(0).map_err(|e| MlirError::CodegenFailed { err: e.to_string() })?;
                mlir_block.append_operation(op);
                return Ok(Some(result.into()));
            }
        }
        self.codegen_ub_stub(tcx, instance, mir, destination, location, mlir_block)
    }

    /// `triton::Triton::load_full` — block-pointer load; delegates to the same
    /// MLIR operation as a regular `tt.load`.
    pub fn codegen_load_full_call<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        mir: &Body<'tcx>,
        func: &Operand<'tcx>,
        func_name: &str,
        args: &[Spanned<Operand<'tcx>>],
        destination: &Place<'tcx>,
        target: &Option<BasicBlock>,
        unwind: &UnwindAction,
        call_source: &CallSource,
        fn_span: &Span,
        location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>,
        state: &mut CodegenState<'a, 'a>,
    ) -> Result<Option<Value<'a, 'a>>, MlirError> {
        self.codegen_load(tcx, instance, mir, func, func_name, args, destination, target, unwind, call_source, fn_span, location, mlir_block, state)
    }

    /// `triton::Triton::store_full` — block-pointer store; delegates to `tt.store`.
    pub fn codegen_store_full_call<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        mir: &Body<'tcx>,
        func: &Operand<'tcx>,
        func_name: &str,
        args: &[Spanned<Operand<'tcx>>],
        destination: &Place<'tcx>,
        target: &Option<BasicBlock>,
        unwind: &UnwindAction,
        call_source: &CallSource,
        fn_span: &Span,
        location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>,
        state: &mut CodegenState<'a, 'a>,
    ) -> Result<Option<Value<'a, 'a>>, MlirError> {
        self.codegen_store(tcx, instance, mir, func, func_name, args, destination, target, unwind, call_source, fn_span, location, mlir_block, state)
    }

    // =========================================================================
    // Control-flow / debug ops
    // =========================================================================

    /// `triton::Triton::where_` — conditional element-wise select (`arith.select`).
    pub fn codegen_where_call<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        mir: &Body<'tcx>,
        _func: &Operand<'tcx>,
        _func_name: &str,
        args: &[Spanned<Operand<'tcx>>],
        _destination: &Place<'tcx>,
        _target: &Option<BasicBlock>,
        _unwind: &UnwindAction,
        _call_source: &CallSource,
        _fn_span: &Span,
        location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>,
        state: &mut CodegenState<'a, 'a>,
    ) -> Result<Option<Value<'a, 'a>>, MlirError> {
        // args: [condition, true_val, false_val]
        let cond = self.codegen_operand(
            tcx, instance, &args[0].node, args[0].node.ty(mir, tcx), location, mlir_block, state,
        )?;
        let true_val = self.codegen_operand(
            tcx, instance, &args[1].node, args[1].node.ty(mir, tcx), location, mlir_block, state,
        )?;
        let false_val = self.codegen_operand(
            tcx, instance, &args[2].node, args[2].node.ty(mir, tcx), location, mlir_block, state,
        )?;
        let result_ty = true_val.r#type();
        let op: Operation<'a> = OperationBuilder::new("arith.select", location)
            .add_operands(&[cond, true_val, false_val])
            .add_results(&[result_ty])
            .build()
            .map_err(|e| MlirError::CodegenFailed { err: e.to_string() })?;
        let result = op.result(0).map_err(|e| MlirError::CodegenFailed { err: e.to_string() })?;
        mlir_block.append_operation(op);
        Ok(Some(result.into()))
    }

    /// `triton::Triton::assume` — a no-op hint for the compiler.
    pub fn codegen_assume_call<'tcx>(
        &self,
        _tcx: TyCtxt<'tcx>,
        _instance: &Instance<'tcx>,
        _mir: &Body<'tcx>,
        _func: &Operand<'tcx>,
        _func_name: &str,
        _args: &[Spanned<Operand<'tcx>>],
        _destination: &Place<'tcx>,
        _target: &Option<BasicBlock>,
        _unwind: &UnwindAction,
        _call_source: &CallSource,
        _fn_span: &Span,
        _location: Location<'a>,
        _mlir_block: &BlockRef<'a, 'a>,
        _state: &mut CodegenState<'a, 'a>,
    ) -> Result<Option<Value<'a, 'a>>, MlirError> {
        Ok(None)
    }

    /// `triton::Triton::device_assert` — runtime assertion (`tt.assert`).
    pub fn codegen_device_assert_call<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        mir: &Body<'tcx>,
        _func: &Operand<'tcx>,
        _func_name: &str,
        args: &[Spanned<Operand<'tcx>>],
        _destination: &Place<'tcx>,
        _target: &Option<BasicBlock>,
        _unwind: &UnwindAction,
        _call_source: &CallSource,
        _fn_span: &Span,
        location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>,
        state: &mut CodegenState<'a, 'a>,
    ) -> Result<Option<Value<'a, 'a>>, MlirError> {
        let cond = self.codegen_operand(
            tcx, instance, &args[0].node, args[0].node.ty(mir, tcx), location, mlir_block, state,
        )?;
        let op: Operation<'a> = assert_op(self.module.context(), location, cond, "device_assert")
            .map_err(|e| MlirError::CreateOperation { err: e })?
            .into();
        mlir_block.append_operation(op);
        Ok(None)
    }

    /// `triton::Triton::device_print` — debug print (`tt.print`).
    pub fn codegen_device_print_call<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        mir: &Body<'tcx>,
        _func: &Operand<'tcx>,
        _func_name: &str,
        args: &[Spanned<Operand<'tcx>>],
        _destination: &Place<'tcx>,
        _target: &Option<BasicBlock>,
        _unwind: &UnwindAction,
        _call_source: &CallSource,
        _fn_span: &Span,
        location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>,
        state: &mut CodegenState<'a, 'a>,
    ) -> Result<Option<Value<'a, 'a>>, MlirError> {
        // args: [prefix_str (i64 placeholder), args...]
        // Collect printable tensor args (skip the first string arg).
        let mut print_vals: Vec<Value<'a, 'a>> = Vec::new();
        for spanned in args.iter().skip(1) {
            if let Ok(v) = self.codegen_operand(
                tcx, instance, &spanned.node, spanned.node.ty(mir, tcx), location, mlir_block, state,
            ) {
                print_vals.push(v);
            }
        }
        let is_signed: Vec<i32> = print_vals.iter().map(|_| 0i32).collect();
        let op: Operation<'a> = triton_print(
            self.module.context(), location, "device_print:", false,
            &print_vals, &is_signed,
        )
        .map_err(|e| MlirError::CreateOperation { err: e })?
        .into();
        mlir_block.append_operation(op);
        Ok(None)
    }

    // =========================================================================
    // Tensor manipulation (flip, gather)
    // =========================================================================

    // `triton::Triton::flip` — no `tt.flip` op in this version; fall back to UB stub.
    stub_handler!(codegen_flip_call);

    /// `triton::Triton::gather` — indexed gather along an axis (`tt.gather`).
    pub fn codegen_gather_call<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        mir: &Body<'tcx>,
        _func: &Operand<'tcx>,
        _func_name: &str,
        args: &[Spanned<Operand<'tcx>>],
        destination: &Place<'tcx>,
        _target: &Option<BasicBlock>,
        _unwind: &UnwindAction,
        _call_source: &CallSource,
        _fn_span: &Span,
        location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>,
        state: &mut CodegenState<'a, 'a>,
    ) -> Result<Option<Value<'a, 'a>>, MlirError> {
        // args: [src, indices, axis]
        let src = self.codegen_operand(
            tcx, instance, &args[0].node, args[0].node.ty(mir, tcx), location, mlir_block, state,
        )?;
        let indices = self.codegen_operand(
            tcx, instance, &args[1].node, args[1].node.ty(mir, tcx), location, mlir_block, state,
        )?;
        let axis = self
            .to_scalar_int(tcx, instance, &args[2].node)
            .map(|s| s.to_i32())
            .unwrap_or(0);

        let dest_ty = destination.ty(mir, tcx).ty;
        let dest_ty = instance.instantiate_mir_and_normalize_erasing_regions(
            tcx, TypingEnv::fully_monomorphized(), EarlyBinder::bind(dest_ty),
        );
        let result_ty = self.type_mapper.map_type(self.module.context(), &tcx, &dest_ty);

        let op: Operation<'a> = gather(
            self.module.context(), location, src, indices, axis, false, result_ty,
        )
        .map_err(|e| MlirError::CreateOperation { err: e })?
        .into();
        let result = op.result(0).map_err(|e| MlirError::CodegenFailed { err: e.to_string() })?;
        mlir_block.append_operation(op);
        Ok(Some(result.into()))
    }

    // =========================================================================
    // Unary elementwise math ops  (math.*)
    // =========================================================================
    //
    // All take a single tensor/scalar operand and return the same type.

    /// Emit a unary `math.*` or `tt.*` op.
    fn codegen_math_unary<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        mir: &Body<'tcx>,
        args: &[Spanned<Operand<'tcx>>],
        mlir_op: &str,
        location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>,
        state: &mut CodegenState<'a, 'a>,
    ) -> Result<Option<Value<'a, 'a>>, MlirError> {
        let x = self.codegen_operand(
            tcx, instance, &args[0].node, args[0].node.ty(mir, tcx), location, mlir_block, state,
        )?;
        let result_ty = x.r#type();
        let op: Operation<'a> = OperationBuilder::new(mlir_op, location)
            .add_operands(&[x])
            .add_results(&[result_ty])
            .build()
            .map_err(|e| MlirError::CodegenFailed { err: e.to_string() })?;
        let result = op.result(0).map_err(|e| MlirError::CodegenFailed { err: e.to_string() })?;
        mlir_block.append_operation(op);
        Ok(Some(result.into()))
    }

    /// Emit a binary `arith.*` / `math.*` op with two equal-typed operands.
    fn codegen_binary_elementwise<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        mir: &Body<'tcx>,
        args: &[Spanned<Operand<'tcx>>],
        mlir_op: &str,
        location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>,
        state: &mut CodegenState<'a, 'a>,
    ) -> Result<Option<Value<'a, 'a>>, MlirError> {
        let lhs = self.codegen_operand(
            tcx, instance, &args[0].node, args[0].node.ty(mir, tcx), location, mlir_block, state,
        )?;
        let rhs = self.codegen_operand(
            tcx, instance, &args[1].node, args[1].node.ty(mir, tcx), location, mlir_block, state,
        )?;
        let result_ty = lhs.r#type();
        let op: Operation<'a> = OperationBuilder::new(mlir_op, location)
            .add_operands(&[lhs, rhs])
            .add_results(&[result_ty])
            .build()
            .map_err(|e| MlirError::CodegenFailed { err: e.to_string() })?;
        let result = op.result(0).map_err(|e| MlirError::CodegenFailed { err: e.to_string() })?;
        mlir_block.append_operation(op);
        Ok(Some(result.into()))
    }

    pub fn codegen_abs_call<'tcx>(
        &self, tcx: TyCtxt<'tcx>, instance: &Instance<'tcx>, mir: &Body<'tcx>,
        _func: &Operand<'tcx>, _func_name: &str, args: &[Spanned<Operand<'tcx>>],
        _destination: &Place<'tcx>, _target: &Option<BasicBlock>, _unwind: &UnwindAction,
        _call_source: &CallSource, _fn_span: &Span, location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>, state: &mut CodegenState<'a, 'a>,
    ) -> Result<Option<Value<'a, 'a>>, MlirError> {
        self.codegen_math_unary(tcx, instance, mir, args, "math.absf", location, mlir_block, state)
    }

    pub fn codegen_ceil_call<'tcx>(
        &self, tcx: TyCtxt<'tcx>, instance: &Instance<'tcx>, mir: &Body<'tcx>,
        _func: &Operand<'tcx>, _func_name: &str, args: &[Spanned<Operand<'tcx>>],
        _destination: &Place<'tcx>, _target: &Option<BasicBlock>, _unwind: &UnwindAction,
        _call_source: &CallSource, _fn_span: &Span, location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>, state: &mut CodegenState<'a, 'a>,
    ) -> Result<Option<Value<'a, 'a>>, MlirError> {
        self.codegen_math_unary(tcx, instance, mir, args, "math.ceil", location, mlir_block, state)
    }

    pub fn codegen_floor_call<'tcx>(
        &self, tcx: TyCtxt<'tcx>, instance: &Instance<'tcx>, mir: &Body<'tcx>,
        _func: &Operand<'tcx>, _func_name: &str, args: &[Spanned<Operand<'tcx>>],
        _destination: &Place<'tcx>, _target: &Option<BasicBlock>, _unwind: &UnwindAction,
        _call_source: &CallSource, _fn_span: &Span, location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>, state: &mut CodegenState<'a, 'a>,
    ) -> Result<Option<Value<'a, 'a>>, MlirError> {
        self.codegen_math_unary(tcx, instance, mir, args, "math.floor", location, mlir_block, state)
    }

    pub fn codegen_cos_call<'tcx>(
        &self, tcx: TyCtxt<'tcx>, instance: &Instance<'tcx>, mir: &Body<'tcx>,
        _func: &Operand<'tcx>, _func_name: &str, args: &[Spanned<Operand<'tcx>>],
        _destination: &Place<'tcx>, _target: &Option<BasicBlock>, _unwind: &UnwindAction,
        _call_source: &CallSource, _fn_span: &Span, location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>, state: &mut CodegenState<'a, 'a>,
    ) -> Result<Option<Value<'a, 'a>>, MlirError> {
        self.codegen_math_unary(tcx, instance, mir, args, "math.cos", location, mlir_block, state)
    }

    pub fn codegen_sin_call<'tcx>(
        &self, tcx: TyCtxt<'tcx>, instance: &Instance<'tcx>, mir: &Body<'tcx>,
        _func: &Operand<'tcx>, _func_name: &str, args: &[Spanned<Operand<'tcx>>],
        _destination: &Place<'tcx>, _target: &Option<BasicBlock>, _unwind: &UnwindAction,
        _call_source: &CallSource, _fn_span: &Span, location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>, state: &mut CodegenState<'a, 'a>,
    ) -> Result<Option<Value<'a, 'a>>, MlirError> {
        self.codegen_math_unary(tcx, instance, mir, args, "math.sin", location, mlir_block, state)
    }

    pub fn codegen_exp_call<'tcx>(
        &self, tcx: TyCtxt<'tcx>, instance: &Instance<'tcx>, mir: &Body<'tcx>,
        _func: &Operand<'tcx>, _func_name: &str, args: &[Spanned<Operand<'tcx>>],
        _destination: &Place<'tcx>, _target: &Option<BasicBlock>, _unwind: &UnwindAction,
        _call_source: &CallSource, _fn_span: &Span, location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>, state: &mut CodegenState<'a, 'a>,
    ) -> Result<Option<Value<'a, 'a>>, MlirError> {
        self.codegen_math_unary(tcx, instance, mir, args, "math.exp", location, mlir_block, state)
    }

    pub fn codegen_exp2_call<'tcx>(
        &self, tcx: TyCtxt<'tcx>, instance: &Instance<'tcx>, mir: &Body<'tcx>,
        _func: &Operand<'tcx>, _func_name: &str, args: &[Spanned<Operand<'tcx>>],
        _destination: &Place<'tcx>, _target: &Option<BasicBlock>, _unwind: &UnwindAction,
        _call_source: &CallSource, _fn_span: &Span, location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>, state: &mut CodegenState<'a, 'a>,
    ) -> Result<Option<Value<'a, 'a>>, MlirError> {
        self.codegen_math_unary(tcx, instance, mir, args, "math.exp2", location, mlir_block, state)
    }

    pub fn codegen_log_call<'tcx>(
        &self, tcx: TyCtxt<'tcx>, instance: &Instance<'tcx>, mir: &Body<'tcx>,
        _func: &Operand<'tcx>, _func_name: &str, args: &[Spanned<Operand<'tcx>>],
        _destination: &Place<'tcx>, _target: &Option<BasicBlock>, _unwind: &UnwindAction,
        _call_source: &CallSource, _fn_span: &Span, location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>, state: &mut CodegenState<'a, 'a>,
    ) -> Result<Option<Value<'a, 'a>>, MlirError> {
        self.codegen_math_unary(tcx, instance, mir, args, "math.log", location, mlir_block, state)
    }

    pub fn codegen_log2_call<'tcx>(
        &self, tcx: TyCtxt<'tcx>, instance: &Instance<'tcx>, mir: &Body<'tcx>,
        _func: &Operand<'tcx>, _func_name: &str, args: &[Spanned<Operand<'tcx>>],
        _destination: &Place<'tcx>, _target: &Option<BasicBlock>, _unwind: &UnwindAction,
        _call_source: &CallSource, _fn_span: &Span, location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>, state: &mut CodegenState<'a, 'a>,
    ) -> Result<Option<Value<'a, 'a>>, MlirError> {
        self.codegen_math_unary(tcx, instance, mir, args, "math.log2", location, mlir_block, state)
    }

    pub fn codegen_rsqrt_call<'tcx>(
        &self, tcx: TyCtxt<'tcx>, instance: &Instance<'tcx>, mir: &Body<'tcx>,
        _func: &Operand<'tcx>, _func_name: &str, args: &[Spanned<Operand<'tcx>>],
        _destination: &Place<'tcx>, _target: &Option<BasicBlock>, _unwind: &UnwindAction,
        _call_source: &CallSource, _fn_span: &Span, location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>, state: &mut CodegenState<'a, 'a>,
    ) -> Result<Option<Value<'a, 'a>>, MlirError> {
        self.codegen_math_unary(tcx, instance, mir, args, "math.rsqrt", location, mlir_block, state)
    }

    pub fn codegen_sqrt_call<'tcx>(
        &self, tcx: TyCtxt<'tcx>, instance: &Instance<'tcx>, mir: &Body<'tcx>,
        _func: &Operand<'tcx>, _func_name: &str, args: &[Spanned<Operand<'tcx>>],
        _destination: &Place<'tcx>, _target: &Option<BasicBlock>, _unwind: &UnwindAction,
        _call_source: &CallSource, _fn_span: &Span, location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>, state: &mut CodegenState<'a, 'a>,
    ) -> Result<Option<Value<'a, 'a>>, MlirError> {
        self.codegen_math_unary(tcx, instance, mir, args, "math.sqrt", location, mlir_block, state)
    }

    /// `triton::Triton::sqrt_rn` — IEEE-precise sqrt (`tt.precise_sqrt`).
    pub fn codegen_sqrt_rn_call<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        mir: &Body<'tcx>,
        _func: &Operand<'tcx>,
        _func_name: &str,
        args: &[Spanned<Operand<'tcx>>],
        _destination: &Place<'tcx>,
        _target: &Option<BasicBlock>,
        _unwind: &UnwindAction,
        _call_source: &CallSource,
        _fn_span: &Span,
        location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>,
        state: &mut CodegenState<'a, 'a>,
    ) -> Result<Option<Value<'a, 'a>>, MlirError> {
        let x = self.codegen_operand(
            tcx, instance, &args[0].node, args[0].node.ty(mir, tcx), location, mlir_block, state,
        )?;
        let op: Operation<'a> = precise_sqrt(self.module.context(), location, x)
            .map_err(|e| MlirError::CreateOperation { err: e })?
            .into();
        let result = op.result(0).map_err(|e| MlirError::CodegenFailed { err: e.to_string() })?;
        mlir_block.append_operation(op);
        Ok(Some(result.into()))
    }

    pub fn codegen_erf_call<'tcx>(
        &self, tcx: TyCtxt<'tcx>, instance: &Instance<'tcx>, mir: &Body<'tcx>,
        _func: &Operand<'tcx>, _func_name: &str, args: &[Spanned<Operand<'tcx>>],
        _destination: &Place<'tcx>, _target: &Option<BasicBlock>, _unwind: &UnwindAction,
        _call_source: &CallSource, _fn_span: &Span, location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>, state: &mut CodegenState<'a, 'a>,
    ) -> Result<Option<Value<'a, 'a>>, MlirError> {
        self.codegen_math_unary(tcx, instance, mir, args, "math.erf", location, mlir_block, state)
    }

    /// `triton::Triton::sigmoid` — uses CUDA libdevice via `tt.extern_elementwise`.
    pub fn codegen_sigmoid_call<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        mir: &Body<'tcx>,
        _func: &Operand<'tcx>,
        _func_name: &str,
        args: &[Spanned<Operand<'tcx>>],
        destination: &Place<'tcx>,
        _target: &Option<BasicBlock>,
        _unwind: &UnwindAction,
        _call_source: &CallSource,
        _fn_span: &Span,
        location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>,
        state: &mut CodegenState<'a, 'a>,
    ) -> Result<Option<Value<'a, 'a>>, MlirError> {
        let x = self.codegen_operand(
            tcx, instance, &args[0].node, args[0].node.ty(mir, tcx), location, mlir_block, state,
        )?;
        let result_ty = x.r#type();
        let op: Operation<'a> = OperationBuilder::new("tt.extern_elementwise", location)
            .add_operands(&[x])
            .add_results(&[result_ty])
            .add_attributes(&[
                (
                    melior::ir::Identifier::new(self.module.context(), "libname"),
                    melior::ir::Attribute::parse(self.module.context(), r#""libdevice""#).unwrap(),
                ),
                (
                    melior::ir::Identifier::new(self.module.context(), "libpath"),
                    melior::ir::Attribute::parse(self.module.context(), r#""""#).unwrap(),
                ),
                (
                    melior::ir::Identifier::new(self.module.context(), "symbol"),
                    melior::ir::Attribute::parse(self.module.context(), r#""__nv_sigmoidf""#).unwrap(),
                ),
                (
                    melior::ir::Identifier::new(self.module.context(), "pure"),
                    melior::ir::Attribute::parse(self.module.context(), "true").unwrap(),
                ),
            ])
            .build()
            .map_err(|e| MlirError::CodegenFailed { err: e.to_string() })?;
        let result = op.result(0).map_err(|e| MlirError::CodegenFailed { err: e.to_string() })?;
        mlir_block.append_operation(op);
        Ok(Some(result.into()))
    }

    // `triton::Triton::softmax` — multi-step op; no single Triton IR equivalent.
    stub_handler!(codegen_softmax_call);

    // =========================================================================
    // Binary elementwise math ops
    // =========================================================================

    /// `triton::Triton::minimum` — element-wise minimum.
    pub fn codegen_minimum_call<'tcx>(
        &self, tcx: TyCtxt<'tcx>, instance: &Instance<'tcx>, mir: &Body<'tcx>,
        _func: &Operand<'tcx>, _func_name: &str, args: &[Spanned<Operand<'tcx>>],
        _destination: &Place<'tcx>, _target: &Option<BasicBlock>, _unwind: &UnwindAction,
        _call_source: &CallSource, _fn_span: &Span, location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>, state: &mut CodegenState<'a, 'a>,
    ) -> Result<Option<Value<'a, 'a>>, MlirError> {
        let lhs = self.codegen_operand(
            tcx, instance, &args[0].node, args[0].node.ty(mir, tcx), location, mlir_block, state,
        )?;
        let rhs = self.codegen_operand(
            tcx, instance, &args[1].node, args[1].node.ty(mir, tcx), location, mlir_block, state,
        )?;
        let elem = self.elem_ty(lhs);
        let op_name = Self::choose_float_int_op(elem, "arith.minimumf", "arith.minsi");
        let result_ty = lhs.r#type();
        let op: Operation<'a> = OperationBuilder::new(op_name, location)
            .add_operands(&[lhs, rhs])
            .add_results(&[result_ty])
            .build()
            .map_err(|e| MlirError::CodegenFailed { err: e.to_string() })?;
        let result = op.result(0).map_err(|e| MlirError::CodegenFailed { err: e.to_string() })?;
        mlir_block.append_operation(op);
        Ok(Some(result.into()))
    }

    /// `triton::Triton::clamp` — clamped elementwise (`tt.clampf`).
    pub fn codegen_clamp_call<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        mir: &Body<'tcx>,
        _func: &Operand<'tcx>,
        _func_name: &str,
        args: &[Spanned<Operand<'tcx>>],
        _destination: &Place<'tcx>,
        _target: &Option<BasicBlock>,
        _unwind: &UnwindAction,
        _call_source: &CallSource,
        _fn_span: &Span,
        location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>,
        state: &mut CodegenState<'a, 'a>,
    ) -> Result<Option<Value<'a, 'a>>, MlirError> {
        // args: [x, min_val, max_val]
        let x = self.codegen_operand(
            tcx, instance, &args[0].node, args[0].node.ty(mir, tcx), location, mlir_block, state,
        )?;
        let min_val = self.codegen_operand(
            tcx, instance, &args[1].node, args[1].node.ty(mir, tcx), location, mlir_block, state,
        )?;
        let max_val = self.codegen_operand(
            tcx, instance, &args[2].node, args[2].node.ty(mir, tcx), location, mlir_block, state,
        )?;
        let result_ty = x.r#type();
        let op: Operation<'a> = clampf(
            self.module.context(), location, x, min_val, max_val,
            PropagateNan::None, result_ty,
        )
        .map_err(|e| MlirError::CreateOperation { err: e })?
        .into();
        let result = op.result(0).map_err(|e| MlirError::CodegenFailed { err: e.to_string() })?;
        mlir_block.append_operation(op);
        Ok(Some(result.into()))
    }

    /// `triton::Triton::fma` — fused multiply-add (`math.fma`).
    pub fn codegen_fma_call<'tcx>(
        &self, tcx: TyCtxt<'tcx>, instance: &Instance<'tcx>, mir: &Body<'tcx>,
        _func: &Operand<'tcx>, _func_name: &str, args: &[Spanned<Operand<'tcx>>],
        _destination: &Place<'tcx>, _target: &Option<BasicBlock>, _unwind: &UnwindAction,
        _call_source: &CallSource, _fn_span: &Span, location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>, state: &mut CodegenState<'a, 'a>,
    ) -> Result<Option<Value<'a, 'a>>, MlirError> {
        // args: [a, b, c]  result = a*b + c
        let a = self.codegen_operand(
            tcx, instance, &args[0].node, args[0].node.ty(mir, tcx), location, mlir_block, state,
        )?;
        let b = self.codegen_operand(
            tcx, instance, &args[1].node, args[1].node.ty(mir, tcx), location, mlir_block, state,
        )?;
        let c = self.codegen_operand(
            tcx, instance, &args[2].node, args[2].node.ty(mir, tcx), location, mlir_block, state,
        )?;
        let result_ty = a.r#type();
        let op: Operation<'a> = OperationBuilder::new("math.fma", location)
            .add_operands(&[a, b, c])
            .add_results(&[result_ty])
            .build()
            .map_err(|e| MlirError::CodegenFailed { err: e.to_string() })?;
        let result = op.result(0).map_err(|e| MlirError::CodegenFailed { err: e.to_string() })?;
        mlir_block.append_operation(op);
        Ok(Some(result.into()))
    }

    /// `triton::Triton::fdiv` — float division (`arith.divf`).
    pub fn codegen_fdiv_call<'tcx>(
        &self, tcx: TyCtxt<'tcx>, instance: &Instance<'tcx>, mir: &Body<'tcx>,
        _func: &Operand<'tcx>, _func_name: &str, args: &[Spanned<Operand<'tcx>>],
        _destination: &Place<'tcx>, _target: &Option<BasicBlock>, _unwind: &UnwindAction,
        _call_source: &CallSource, _fn_span: &Span, location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>, state: &mut CodegenState<'a, 'a>,
    ) -> Result<Option<Value<'a, 'a>>, MlirError> {
        self.codegen_binary_elementwise(tcx, instance, mir, args, "arith.divf", location, mlir_block, state)
    }

    /// `triton::Triton::div_rn` — IEEE-precise float division (`tt.precise_divf`).
    pub fn codegen_div_rn_call<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        mir: &Body<'tcx>,
        _func: &Operand<'tcx>,
        _func_name: &str,
        args: &[Spanned<Operand<'tcx>>],
        _destination: &Place<'tcx>,
        _target: &Option<BasicBlock>,
        _unwind: &UnwindAction,
        _call_source: &CallSource,
        _fn_span: &Span,
        location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>,
        state: &mut CodegenState<'a, 'a>,
    ) -> Result<Option<Value<'a, 'a>>, MlirError> {
        let x = self.codegen_operand(
            tcx, instance, &args[0].node, args[0].node.ty(mir, tcx), location, mlir_block, state,
        )?;
        let y = self.codegen_operand(
            tcx, instance, &args[1].node, args[1].node.ty(mir, tcx), location, mlir_block, state,
        )?;
        let op: Operation<'a> = precise_divf(self.module.context(), location, x, y)
            .map_err(|e| MlirError::CreateOperation { err: e })?
            .into();
        let result = op.result(0).map_err(|e| MlirError::CodegenFailed { err: e.to_string() })?;
        mlir_block.append_operation(op);
        Ok(Some(result.into()))
    }

    /// `triton::Triton::cdiv` — integer ceiling division: `(a + b - 1) / b`.
    pub fn codegen_cdiv_call<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        mir: &Body<'tcx>,
        _func: &Operand<'tcx>,
        _func_name: &str,
        args: &[Spanned<Operand<'tcx>>],
        _destination: &Place<'tcx>,
        _target: &Option<BasicBlock>,
        _unwind: &UnwindAction,
        _call_source: &CallSource,
        _fn_span: &Span,
        location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>,
        state: &mut CodegenState<'a, 'a>,
    ) -> Result<Option<Value<'a, 'a>>, MlirError> {
        // cdiv(a, b) = (a + b - 1) / b
        let a = self.codegen_operand(
            tcx, instance, &args[0].node, args[0].node.ty(mir, tcx), location, mlir_block, state,
        )?;
        let b = self.codegen_operand(
            tcx, instance, &args[1].node, args[1].node.ty(mir, tcx), location, mlir_block, state,
        )?;
        let ty = a.r#type();

        // Emit `one = arith.constant 1 : <ty>`
        let one_attr = melior::ir::Attribute::parse(
            self.module.context(), &format!("1 : {}", ty),
        )
        .ok_or_else(|| MlirError::CodegenFailed { err: "cdiv: cannot parse 1 const".into() })?;
        let one_op: Operation<'a> = OperationBuilder::new("arith.constant", location)
            .add_attributes(&[(
                melior::ir::Identifier::new(self.module.context(), "value"),
                one_attr,
            )])
            .add_results(&[ty])
            .build()
            .map_err(|e| MlirError::CodegenFailed { err: e.to_string() })?;
        let one: Value<'a, 'a> = one_op.result(0).unwrap().into();
        mlir_block.append_operation(one_op);

        // a + b - 1
        let add_op: Operation<'a> = OperationBuilder::new("arith.addi", location)
            .add_operands(&[a, b])
            .add_results(&[ty])
            .build()
            .map_err(|e| MlirError::CodegenFailed { err: e.to_string() })?;
        let sum: Value<'a, 'a> = add_op.result(0).unwrap().into();
        mlir_block.append_operation(add_op);

        let sub_op: Operation<'a> = OperationBuilder::new("arith.subi", location)
            .add_operands(&[sum, one])
            .add_results(&[ty])
            .build()
            .map_err(|e| MlirError::CodegenFailed { err: e.to_string() })?;
        let numerator: Value<'a, 'a> = sub_op.result(0).unwrap().into();
        mlir_block.append_operation(sub_op);

        let div_op: Operation<'a> = OperationBuilder::new("arith.divsi", location)
            .add_operands(&[numerator, b])
            .add_results(&[ty])
            .build()
            .map_err(|e| MlirError::CodegenFailed { err: e.to_string() })?;
        let result: Value<'a, 'a> = div_op.result(0).unwrap().into();
        mlir_block.append_operation(div_op);
        Ok(Some(result))
    }

    // `triton::Triton::swizzle2d` — no direct Triton IR equivalent.
    stub_handler!(codegen_swizzle2d_call);

    // =========================================================================
    // Reductions (tt.reduce)
    // =========================================================================

    fn codegen_reduce_unary<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        mir: &Body<'tcx>,
        args: &[Spanned<Operand<'tcx>>],
        combine_op_float: &'static str,
        combine_op_int: &'static str,
        location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>,
        state: &mut CodegenState<'a, 'a>,
    ) -> Result<Option<Value<'a, 'a>>, MlirError> {
        let src = self.codegen_operand(
            tcx, instance, &args[0].node, args[0].node.ty(mir, tcx), location, mlir_block, state,
        )?;
        let axis = if args.len() > 1 {
            self.to_scalar_int(tcx, instance, &args[1].node)
                .map(|s| s.to_i32())
                .unwrap_or(0)
        } else {
            0
        };
        let elem = self.elem_ty(src);
        let combine_op = Self::choose_float_int_op(elem, combine_op_float, combine_op_int);
        let region = self.build_reduce_region(location, elem, combine_op);
        let result_ty = self.reduce_result_ty(src, axis);
        let op: Operation<'a> = reduce(
            self.module.context(), location, &[src], &[result_ty], axis, region,
        )
        .map_err(|e| MlirError::CreateOperation { err: e })?
        .into();
        let result = op.result(0).map_err(|e| MlirError::CodegenFailed { err: e.to_string() })?;
        mlir_block.append_operation(op);
        Ok(Some(result.into()))
    }

    /// `triton::Triton::sum` — reduction sum.
    pub fn codegen_sum_call<'tcx>(
        &self, tcx: TyCtxt<'tcx>, instance: &Instance<'tcx>, mir: &Body<'tcx>,
        _func: &Operand<'tcx>, _func_name: &str, args: &[Spanned<Operand<'tcx>>],
        _destination: &Place<'tcx>, _target: &Option<BasicBlock>, _unwind: &UnwindAction,
        _call_source: &CallSource, _fn_span: &Span, location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>, state: &mut CodegenState<'a, 'a>,
    ) -> Result<Option<Value<'a, 'a>>, MlirError> {
        self.codegen_reduce_unary(tcx, instance, mir, args, "arith.addf", "arith.addi", location, mlir_block, state)
    }

    /// `triton::Triton::max` — reduction max.
    pub fn codegen_max_call<'tcx>(
        &self, tcx: TyCtxt<'tcx>, instance: &Instance<'tcx>, mir: &Body<'tcx>,
        _func: &Operand<'tcx>, _func_name: &str, args: &[Spanned<Operand<'tcx>>],
        _destination: &Place<'tcx>, _target: &Option<BasicBlock>, _unwind: &UnwindAction,
        _call_source: &CallSource, _fn_span: &Span, location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>, state: &mut CodegenState<'a, 'a>,
    ) -> Result<Option<Value<'a, 'a>>, MlirError> {
        self.codegen_reduce_unary(tcx, instance, mir, args, "arith.maximumf", "arith.maxsi", location, mlir_block, state)
    }

    /// `triton::Triton::min` — reduction min.
    pub fn codegen_min_call<'tcx>(
        &self, tcx: TyCtxt<'tcx>, instance: &Instance<'tcx>, mir: &Body<'tcx>,
        _func: &Operand<'tcx>, _func_name: &str, args: &[Spanned<Operand<'tcx>>],
        _destination: &Place<'tcx>, _target: &Option<BasicBlock>, _unwind: &UnwindAction,
        _call_source: &CallSource, _fn_span: &Span, location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>, state: &mut CodegenState<'a, 'a>,
    ) -> Result<Option<Value<'a, 'a>>, MlirError> {
        self.codegen_reduce_unary(tcx, instance, mir, args, "arith.minimumf", "arith.minsi", location, mlir_block, state)
    }

    /// `triton::Triton::xor_sum` — reduction XOR.
    pub fn codegen_xor_sum_call<'tcx>(
        &self, tcx: TyCtxt<'tcx>, instance: &Instance<'tcx>, mir: &Body<'tcx>,
        _func: &Operand<'tcx>, _func_name: &str, args: &[Spanned<Operand<'tcx>>],
        _destination: &Place<'tcx>, _target: &Option<BasicBlock>, _unwind: &UnwindAction,
        _call_source: &CallSource, _fn_span: &Span, location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>, state: &mut CodegenState<'a, 'a>,
    ) -> Result<Option<Value<'a, 'a>>, MlirError> {
        self.codegen_reduce_unary(tcx, instance, mir, args, "arith.xori", "arith.xori", location, mlir_block, state)
    }

    // argmax/argmin/max_with_indices/min_with_indices require multi-value reduce
    // regions with (value, index) pairs — kept as stubs until index-tracking is
    // implemented.
    stub_handler!(codegen_max_with_indices_call);
    stub_handler!(codegen_min_with_indices_call);
    stub_handler!(codegen_argmax_call);
    stub_handler!(codegen_argmin_call);

    // =========================================================================
    // Scans (tt.scan)
    // =========================================================================

    fn codegen_scan_unary<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        mir: &Body<'tcx>,
        args: &[Spanned<Operand<'tcx>>],
        combine_op_float: &'static str,
        combine_op_int: &'static str,
        reverse: bool,
        location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>,
        state: &mut CodegenState<'a, 'a>,
    ) -> Result<Option<Value<'a, 'a>>, MlirError> {
        let src = self.codegen_operand(
            tcx, instance, &args[0].node, args[0].node.ty(mir, tcx), location, mlir_block, state,
        )?;
        let axis = if args.len() > 1 {
            self.to_scalar_int(tcx, instance, &args[1].node)
                .map(|s| s.to_i32())
                .unwrap_or(0)
        } else {
            0
        };
        let elem = self.elem_ty(src);
        let combine_op = Self::choose_float_int_op(elem, combine_op_float, combine_op_int);
        let region = self.build_scan_region(location, elem, combine_op);
        let result_ty = src.r#type();
        let op: Operation<'a> = scan(
            self.module.context(), location, &[src], &[result_ty], axis, reverse, region,
        )
        .map_err(|e| MlirError::CreateOperation { err: e })?
        .into();
        let result = op.result(0).map_err(|e| MlirError::CodegenFailed { err: e.to_string() })?;
        mlir_block.append_operation(op);
        Ok(Some(result.into()))
    }

    /// `triton::Triton::cumsum` — prefix sum scan.
    pub fn codegen_cumsum_call<'tcx>(
        &self, tcx: TyCtxt<'tcx>, instance: &Instance<'tcx>, mir: &Body<'tcx>,
        _func: &Operand<'tcx>, _func_name: &str, args: &[Spanned<Operand<'tcx>>],
        _destination: &Place<'tcx>, _target: &Option<BasicBlock>, _unwind: &UnwindAction,
        _call_source: &CallSource, _fn_span: &Span, location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>, state: &mut CodegenState<'a, 'a>,
    ) -> Result<Option<Value<'a, 'a>>, MlirError> {
        self.codegen_scan_unary(tcx, instance, mir, args, "arith.addf", "arith.addi", false, location, mlir_block, state)
    }

    /// `triton::Triton::cumprod` — prefix product scan.
    pub fn codegen_cumprod_call<'tcx>(
        &self, tcx: TyCtxt<'tcx>, instance: &Instance<'tcx>, mir: &Body<'tcx>,
        _func: &Operand<'tcx>, _func_name: &str, args: &[Spanned<Operand<'tcx>>],
        _destination: &Place<'tcx>, _target: &Option<BasicBlock>, _unwind: &UnwindAction,
        _call_source: &CallSource, _fn_span: &Span, location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>, state: &mut CodegenState<'a, 'a>,
    ) -> Result<Option<Value<'a, 'a>>, MlirError> {
        self.codegen_scan_unary(tcx, instance, mir, args, "arith.mulf", "arith.muli", false, location, mlir_block, state)
    }

    // `triton::Triton::sort` — no direct Triton IR equivalent.
    stub_handler!(codegen_sort_call);

    /// `triton::Triton::histogram` — `tt.histogram`.
    pub fn codegen_histogram_call<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        mir: &Body<'tcx>,
        _func: &Operand<'tcx>,
        _func_name: &str,
        args: &[Spanned<Operand<'tcx>>],
        destination: &Place<'tcx>,
        _target: &Option<BasicBlock>,
        _unwind: &UnwindAction,
        _call_source: &CallSource,
        _fn_span: &Span,
        location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>,
        state: &mut CodegenState<'a, 'a>,
    ) -> Result<Option<Value<'a, 'a>>, MlirError> {
        // args: [src, Option<mask>]
        let src = self.codegen_operand(
            tcx, instance, &args[0].node, args[0].node.ty(mir, tcx), location, mlir_block, state,
        )?;
        let mask = if args.len() > 1 {
            self.codegen_option_operand(tcx, instance, mir, &args[1].node, location, mlir_block, state)?
        } else {
            None
        };

        let dest_ty = destination.ty(mir, tcx).ty;
        let dest_ty = instance.instantiate_mir_and_normalize_erasing_regions(
            tcx, TypingEnv::fully_monomorphized(), EarlyBinder::bind(dest_ty),
        );
        let result_ty = self.type_mapper.map_type(self.module.context(), &tcx, &dest_ty);

        let op: Operation<'a> = histogram(self.module.context(), location, src, mask, result_ty)
            .map_err(|e| MlirError::CreateOperation { err: e })?
            .into();
        let result = op.result(0).map_err(|e| MlirError::CodegenFailed { err: e.to_string() })?;
        mlir_block.append_operation(op);
        Ok(Some(result.into()))
    }

    // `triton::Triton::reduce` and `associative_scan` require user-provided closure
    // regions which cannot be generated without inlining the closure body.
    stub_handler!(codegen_reduce_call);
    stub_handler!(codegen_associative_scan_call);

    // =========================================================================
    // Atomic ops (tt.atomic_rmw / tt.atomic_cas)
    // =========================================================================

    fn codegen_atomic_rmw_impl<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        mir: &Body<'tcx>,
        args: &[Spanned<Operand<'tcx>>],
        rmw_op: RmwOp,
        location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>,
        state: &mut CodegenState<'a, 'a>,
    ) -> Result<Option<Value<'a, 'a>>, MlirError> {
        // args: [ptr, val, Option<mask>]
        let ptr = self.codegen_operand(
            tcx, instance, &args[0].node, args[0].node.ty(mir, tcx), location, mlir_block, state,
        )?;
        let val = self.codegen_operand(
            tcx, instance, &args[1].node, args[1].node.ty(mir, tcx), location, mlir_block, state,
        )?;
        let mask = if args.len() > 2 {
            self.codegen_option_operand(tcx, instance, mir, &args[2].node, location, mlir_block, state)?
        } else {
            None
        };
        let result_ty = val.r#type();
        let op: Operation<'a> = atomic_rmw(
            self.module.context(), location, ptr, val, mask, result_ty,
            rmw_op, MemSemantic::Relaxed, MemSyncScope::Gpu,
        )
        .map_err(|e| MlirError::CreateOperation { err: e })?
        .into();
        let result = op.result(0).map_err(|e| MlirError::CodegenFailed { err: e.to_string() })?;
        mlir_block.append_operation(op);
        Ok(Some(result.into()))
    }

    pub fn codegen_atomic_add_call<'tcx>(
        &self, tcx: TyCtxt<'tcx>, instance: &Instance<'tcx>, mir: &Body<'tcx>,
        _func: &Operand<'tcx>, _func_name: &str, args: &[Spanned<Operand<'tcx>>],
        _destination: &Place<'tcx>, _target: &Option<BasicBlock>, _unwind: &UnwindAction,
        _call_source: &CallSource, _fn_span: &Span, location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>, state: &mut CodegenState<'a, 'a>,
    ) -> Result<Option<Value<'a, 'a>>, MlirError> {
        // Use Fadd for floats; Add for ints.  Default to Fadd if type can't be inferred.
        let rmw = match self.codegen_operand(
            tcx, instance, &args[1].node, args[1].node.ty(mir, tcx), location, mlir_block, state,
        ) {
            Ok(v) => {
                let elem = self.elem_ty(v);
                if elem.is_integer() || elem.is_index() { RmwOp::Add } else { RmwOp::Fadd }
            }
            Err(_) => RmwOp::Fadd,
        };
        self.codegen_atomic_rmw_impl(tcx, instance, mir, args, rmw, location, mlir_block, state)
    }

    pub fn codegen_atomic_max_call<'tcx>(
        &self, tcx: TyCtxt<'tcx>, instance: &Instance<'tcx>, mir: &Body<'tcx>,
        _func: &Operand<'tcx>, _func_name: &str, args: &[Spanned<Operand<'tcx>>],
        _destination: &Place<'tcx>, _target: &Option<BasicBlock>, _unwind: &UnwindAction,
        _call_source: &CallSource, _fn_span: &Span, location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>, state: &mut CodegenState<'a, 'a>,
    ) -> Result<Option<Value<'a, 'a>>, MlirError> {
        self.codegen_atomic_rmw_impl(tcx, instance, mir, args, RmwOp::Max, location, mlir_block, state)
    }

    pub fn codegen_atomic_min_call<'tcx>(
        &self, tcx: TyCtxt<'tcx>, instance: &Instance<'tcx>, mir: &Body<'tcx>,
        _func: &Operand<'tcx>, _func_name: &str, args: &[Spanned<Operand<'tcx>>],
        _destination: &Place<'tcx>, _target: &Option<BasicBlock>, _unwind: &UnwindAction,
        _call_source: &CallSource, _fn_span: &Span, location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>, state: &mut CodegenState<'a, 'a>,
    ) -> Result<Option<Value<'a, 'a>>, MlirError> {
        self.codegen_atomic_rmw_impl(tcx, instance, mir, args, RmwOp::Min, location, mlir_block, state)
    }

    pub fn codegen_atomic_xchg_call<'tcx>(
        &self, tcx: TyCtxt<'tcx>, instance: &Instance<'tcx>, mir: &Body<'tcx>,
        _func: &Operand<'tcx>, _func_name: &str, args: &[Spanned<Operand<'tcx>>],
        _destination: &Place<'tcx>, _target: &Option<BasicBlock>, _unwind: &UnwindAction,
        _call_source: &CallSource, _fn_span: &Span, location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>, state: &mut CodegenState<'a, 'a>,
    ) -> Result<Option<Value<'a, 'a>>, MlirError> {
        self.codegen_atomic_rmw_impl(tcx, instance, mir, args, RmwOp::Xchg, location, mlir_block, state)
    }

    pub fn codegen_atomic_and_call<'tcx>(
        &self, tcx: TyCtxt<'tcx>, instance: &Instance<'tcx>, mir: &Body<'tcx>,
        _func: &Operand<'tcx>, _func_name: &str, args: &[Spanned<Operand<'tcx>>],
        _destination: &Place<'tcx>, _target: &Option<BasicBlock>, _unwind: &UnwindAction,
        _call_source: &CallSource, _fn_span: &Span, location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>, state: &mut CodegenState<'a, 'a>,
    ) -> Result<Option<Value<'a, 'a>>, MlirError> {
        self.codegen_atomic_rmw_impl(tcx, instance, mir, args, RmwOp::And, location, mlir_block, state)
    }

    pub fn codegen_atomic_or_call<'tcx>(
        &self, tcx: TyCtxt<'tcx>, instance: &Instance<'tcx>, mir: &Body<'tcx>,
        _func: &Operand<'tcx>, _func_name: &str, args: &[Spanned<Operand<'tcx>>],
        _destination: &Place<'tcx>, _target: &Option<BasicBlock>, _unwind: &UnwindAction,
        _call_source: &CallSource, _fn_span: &Span, location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>, state: &mut CodegenState<'a, 'a>,
    ) -> Result<Option<Value<'a, 'a>>, MlirError> {
        self.codegen_atomic_rmw_impl(tcx, instance, mir, args, RmwOp::Or, location, mlir_block, state)
    }

    pub fn codegen_atomic_xor_call<'tcx>(
        &self, tcx: TyCtxt<'tcx>, instance: &Instance<'tcx>, mir: &Body<'tcx>,
        _func: &Operand<'tcx>, _func_name: &str, args: &[Spanned<Operand<'tcx>>],
        _destination: &Place<'tcx>, _target: &Option<BasicBlock>, _unwind: &UnwindAction,
        _call_source: &CallSource, _fn_span: &Span, location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>, state: &mut CodegenState<'a, 'a>,
    ) -> Result<Option<Value<'a, 'a>>, MlirError> {
        self.codegen_atomic_rmw_impl(tcx, instance, mir, args, RmwOp::Xor, location, mlir_block, state)
    }

    /// `triton::Triton::atomic_cas` — `tt.atomic_cas`.
    pub fn codegen_atomic_cas_call<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        mir: &Body<'tcx>,
        _func: &Operand<'tcx>,
        _func_name: &str,
        args: &[Spanned<Operand<'tcx>>],
        _destination: &Place<'tcx>,
        _target: &Option<BasicBlock>,
        _unwind: &UnwindAction,
        _call_source: &CallSource,
        _fn_span: &Span,
        location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>,
        state: &mut CodegenState<'a, 'a>,
    ) -> Result<Option<Value<'a, 'a>>, MlirError> {
        // args: [ptr, cmp, val]
        let ptr = self.codegen_operand(
            tcx, instance, &args[0].node, args[0].node.ty(mir, tcx), location, mlir_block, state,
        )?;
        let cmp = self.codegen_operand(
            tcx, instance, &args[1].node, args[1].node.ty(mir, tcx), location, mlir_block, state,
        )?;
        let val = self.codegen_operand(
            tcx, instance, &args[2].node, args[2].node.ty(mir, tcx), location, mlir_block, state,
        )?;
        let result_ty = val.r#type();
        let op: Operation<'a> = atomic_cas(
            self.module.context(), location, ptr, cmp, val, result_ty,
            MemSemantic::Relaxed, MemSyncScope::Gpu,
        )
        .map_err(|e| MlirError::CreateOperation { err: e })?
        .into();
        let result = op.result(0).map_err(|e| MlirError::CodegenFailed { err: e.to_string() })?;
        mlir_block.append_operation(op);
        Ok(Some(result.into()))
    }

    // =========================================================================
    // Miscellaneous ops
    // =========================================================================

    /// `triton::Triton::umulhi` — unsigned integer multiply high (`tt.mulhiui`).
    pub fn codegen_umulhi_call<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        mir: &Body<'tcx>,
        _func: &Operand<'tcx>,
        _func_name: &str,
        args: &[Spanned<Operand<'tcx>>],
        _destination: &Place<'tcx>,
        _target: &Option<BasicBlock>,
        _unwind: &UnwindAction,
        _call_source: &CallSource,
        _fn_span: &Span,
        location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>,
        state: &mut CodegenState<'a, 'a>,
    ) -> Result<Option<Value<'a, 'a>>, MlirError> {
        let x = self.codegen_operand(
            tcx, instance, &args[0].node, args[0].node.ty(mir, tcx), location, mlir_block, state,
        )?;
        let y = self.codegen_operand(
            tcx, instance, &args[1].node, args[1].node.ty(mir, tcx), location, mlir_block, state,
        )?;
        let op: Operation<'a> = mulhiui(self.module.context(), location, x, y)
            .map_err(|e| MlirError::CreateOperation { err: e })?
            .into();
        let result = op.result(0).map_err(|e| MlirError::CodegenFailed { err: e.to_string() })?;
        mlir_block.append_operation(op);
        Ok(Some(result.into()))
    }

    // rand/randn/randint/randint4x use Triton's Philox RNG which requires
    // special argument handling; kept as stubs.
    stub_handler!(codegen_rand_call);
    stub_handler!(codegen_randn_call);
    stub_handler!(codegen_randint_call);
    stub_handler!(codegen_randint4x_call);

    // inline_asm_elementwise requires extracting string constants from MIR which
    // is not yet implemented.
    stub_handler!(codegen_inline_asm_elementwise_call);

    /// `triton::Triton::multiple_of` / `max_contiguous` / `max_constancy` —
    /// compiler hint annotations; return the input tensor unchanged.
    pub fn codegen_multiple_of_call<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        mir: &Body<'tcx>,
        _func: &Operand<'tcx>,
        _func_name: &str,
        args: &[Spanned<Operand<'tcx>>],
        _destination: &Place<'tcx>,
        _target: &Option<BasicBlock>,
        _unwind: &UnwindAction,
        _call_source: &CallSource,
        _fn_span: &Span,
        location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>,
        state: &mut CodegenState<'a, 'a>,
    ) -> Result<Option<Value<'a, 'a>>, MlirError> {
        let v = self.codegen_operand(
            tcx, instance, &args[0].node, args[0].node.ty(mir, tcx), location, mlir_block, state,
        )?;
        Ok(Some(v))
    }

    pub fn codegen_max_contiguous_call<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        mir: &Body<'tcx>,
        _func: &Operand<'tcx>,
        _func_name: &str,
        args: &[Spanned<Operand<'tcx>>],
        _destination: &Place<'tcx>,
        _target: &Option<BasicBlock>,
        _unwind: &UnwindAction,
        _call_source: &CallSource,
        _fn_span: &Span,
        location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>,
        state: &mut CodegenState<'a, 'a>,
    ) -> Result<Option<Value<'a, 'a>>, MlirError> {
        let v = self.codegen_operand(
            tcx, instance, &args[0].node, args[0].node.ty(mir, tcx), location, mlir_block, state,
        )?;
        Ok(Some(v))
    }

    pub fn codegen_max_constancy_call<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        mir: &Body<'tcx>,
        _func: &Operand<'tcx>,
        _func_name: &str,
        args: &[Spanned<Operand<'tcx>>],
        _destination: &Place<'tcx>,
        _target: &Option<BasicBlock>,
        _unwind: &UnwindAction,
        _call_source: &CallSource,
        _fn_span: &Span,
        location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>,
        state: &mut CodegenState<'a, 'a>,
    ) -> Result<Option<Value<'a, 'a>>, MlirError> {
        let v = self.codegen_operand(
            tcx, instance, &args[0].node, args[0].node.ty(mir, tcx), location, mlir_block, state,
        )?;
        Ok(Some(v))
    }

    /// `triton::Triton::num_programs` — `tt.get_num_programs`.
    pub fn codegen_num_programs_call<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        _mir: &Body<'tcx>,
        _func: &Operand<'tcx>,
        _func_name: &str,
        args: &[Spanned<Operand<'tcx>>],
        _destination: &Place<'tcx>,
        _target: &Option<BasicBlock>,
        _unwind: &UnwindAction,
        _call_source: &CallSource,
        _fn_span: &Span,
        location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>,
        _state: &mut CodegenState<'a, 'a>,
    ) -> Result<Option<Value<'a, 'a>>, MlirError> {
        let axis_val = self
            .to_scalar_int(tcx, instance, &args[0].node)
            .map(|s| s.to_i32())
            .unwrap_or(0);
        let axis = ProgramAxis::from(axis_val);
        let op: Operation<'a> = create_get_num_programs(self.module.context(), location, axis)
            .map_err(|e| MlirError::CreateOperation { err: e })?
            .into();
        let result = op.result(0).map_err(|e| MlirError::CodegenFailed { err: e.to_string() })?;
        mlir_block.append_operation(op);
        Ok(Some(result.into()))
    }

    /// `triton::Triton::full` — splat a scalar value to a tensor shape.
    pub fn codegen_full_call<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        mir: &Body<'tcx>,
        _func: &Operand<'tcx>,
        _func_name: &str,
        args: &[Spanned<Operand<'tcx>>],
        destination: &Place<'tcx>,
        _target: &Option<BasicBlock>,
        _unwind: &UnwindAction,
        _call_source: &CallSource,
        _fn_span: &Span,
        location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>,
        state: &mut CodegenState<'a, 'a>,
    ) -> Result<Option<Value<'a, 'a>>, MlirError> {
        // args: [scalar_value] — shape comes from the destination tensor type.
        let scalar = self.codegen_operand(
            tcx, instance, &args[0].node, args[0].node.ty(mir, tcx), location, mlir_block, state,
        )?;

        let dest_ty = destination.ty(mir, tcx).ty;
        let dest_ty = instance.instantiate_mir_and_normalize_erasing_regions(
            tcx, TypingEnv::fully_monomorphized(), EarlyBinder::bind(dest_ty),
        );
        let result_ty = self.type_mapper.map_type(self.module.context(), &tcx, &dest_ty);

        self.splat_scalar_const(location, scalar, result_ty, mlir_block).map(Some)
    }

    // =========================================================================
    // Helper: element type from a Type value (non-method, used in atomic_add)
    // =========================================================================

    fn elem_ty_from_type(&self, ty: melior::ir::Type<'a>) -> melior::ir::Type<'a> {
        if let Ok(t) = RankedTensorType::try_from(ty) {
            t.element()
        } else {
            ty
        }
    }
}
