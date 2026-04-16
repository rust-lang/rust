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
use melior::ir::operation::OperationLike;
use melior::ir::r#type::RankedTensorType;
use melior::ir::{BlockLike, BlockRef, Location, Operation, ShapedTypeLike, TypeLike, Value, ValueLike};
use rustc_middle::mir::{BasicBlock, Body, CallSource, Operand, Place, UnwindAction};
use rustc_middle::ty::{EarlyBinder, Instance, TyCtxt, TyKind, TypingEnv};
use rustc_mlir::shared::arith::{Int, create_int_constant};
use rustc_mlir::shared::builtin::tensor_type;
use rustc_mlir::triton::tensor::{CacheModifier, EvictionPolicy, add_ptr, make_range, load, splat, store};
use rustc_span::Span;
use rustc_span::source_map::Spanned;

use crate::mlir::codegen::triton::{CodegenState, TritonCodegen};
use crate::mlir::errors::MlirError;

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
        state: &mut CodegenState<'a, 'a>,
    ) -> Result<Option<Value<'a, 'a>>, MlirError> {
        println!(
            "[DEBUG] TritonCodegen::codegen_load: func: {:?} args: {:?} destination: {:?} target: {:?} unwind: {:?} call_source: {:?} fn_span: {:?}",
            func, args, destination, target, unwind, call_source, fn_span
        );

        debug_assert!(
            args.len() == 2,
            "TritonCodegen::codegen_load: args length must be 2: {:?}",
            args
        );

        let arg0 = &args[0].node;
        let arg1 = &args[1].node;

        let ptr = self.codegen_operand(
            tcx, instance, arg0, arg0.ty(mir, tcx), location, mlir_block, state,
        )?;
        let mask = self.codegen_option_operand(tcx, instance, mir, arg1, location, mlir_block, state)?;

        // Derive the result type from the MIR destination place type.
        let dest_ty = instance.instantiate_mir_and_normalize_erasing_regions(
            tcx,
            TypingEnv::fully_monomorphized(),
            EarlyBinder::bind(destination.ty(mir, tcx).ty),
        );
        let result_ty = self.type_mapper.map_type(self.module.context(), &tcx, &dest_ty);

        let load_op: Operation<'a> =
            load(
                self.module.context(),
                location,
                ptr,
                mask,
                None,
                result_ty,
                CacheModifier::None,
                EvictionPolicy::Normal,
                false,
            )
            .map_err(|e| MlirError::CreateOperation { err: e })?
            .into();
        let result = load_op.result(0).expect("Load operation result not found");
        eprintln!("[DEBUG] AXM TritonCodegen::codegen_load: {:?}", load_op.to_string());
        mlir_block.append_operation(load_op);
        Ok(Some(result.into()))
    }

    pub fn codegen_store<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        mir: &Body<'tcx>,
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
        state: &mut CodegenState<'a, 'a>,
    ) -> Result<Option<Value<'a, 'a>>, MlirError> {
        println!(
            "[DEBUG] TritonCodegen::codegen_store: func: {:?} args: {:?} destination: {:?} target: {:?} unwind: {:?} call_source: {:?} fn_span: {:?}",
            func, args, destination, target, unwind, call_source, fn_span
        );

        debug_assert!(
            args.len() == 3,
            "TritonCodegen::codegen_store: args length must be 3: {:?}",
            args
        );

        let arg0 = &args[0].node;
        let arg1 = &args[1].node;
        let arg2 = &args[2].node;

        let dest = self.codegen_operand(
            tcx, instance, arg0, arg0.ty(mir, tcx), location, mlir_block, state,
        )?;
        let src = self.codegen_operand(
            tcx, instance, arg1, arg1.ty(mir, tcx), location, mlir_block, state,
        )?;
        let mask = self.codegen_option_operand(tcx, instance, mir, arg2, location, mlir_block, state)?;

        let store_op: Operation<'a> =
            store(
                self.module.context(),
                location,
                dest,
                src,
                mask,
                CacheModifier::None,
                EvictionPolicy::Normal,
            )
            .map_err(|e| MlirError::CreateOperation { err: e })?
            .into();

        eprintln!("[DEBUG] AXM TritonCodegen::codegen_store: {:?}", store_op.to_string());
        mlir_block.append_operation(store_op);

        Ok(None)
    }

    pub fn codegen_maximum<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        mir: &Body<'tcx>,
        func: &Operand<'tcx>,
        _func_name: &str,
        args: &[Spanned<Operand<'tcx>>],
        destination: &Place<'tcx>,
        target: &Option<BasicBlock>,
        unwind: &UnwindAction,
        call_source: &CallSource,
        fn_span: &Span,
        _location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>,
        state: &mut CodegenState<'a, 'a>,
    ) -> Result<Option<Value<'a, 'a>>, MlirError> {
        println!(
            "[DEBUG] TritonCodegen::codegen_maximum: func: {:?} args: {:?} destination: {:?} target: {:?} unwind: {:?} call_source: {:?} fn_span: {:?}",
            func, args, destination, target, unwind, call_source, fn_span
        );

        debug_assert!(
            args.len() == 2,
            "TritonCodegen::codegen_maximum: args length must be 2: {:?}",
            args
        );

        let arg0 = &args[0].node;
        let arg1 = &args[1].node;

        let _x = self.codegen_operand(
            tcx, instance, arg0, arg0.ty(mir, tcx), _location, mlir_block, state,
        )?;
        let _y = self.codegen_operand(
            tcx, instance, arg1, arg1.ty(mir, tcx), _location, mlir_block, state,
        )?;

        todo!()
        // let maximum_op: Operation<'a> =
        //     maximumf(self.module.context(), _location, x, y)
        //         .map_err(|e| MlirError::CreateOperation { err: e })?
        //         .into();
        // let result = maximum_op.result(0).expect("Maximum operation result not found");
        // eprintln!("[DEBUG] AXM TritonCodegen::codegen_maximum: {:?}", maximum_op.to_string());
        // mlir_block.append_operation(maximum_op);
        // Ok(Some(result.into()))
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
}
