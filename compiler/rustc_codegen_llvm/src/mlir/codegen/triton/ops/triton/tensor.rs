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

use melior::ir::operation::OperationLike;
use melior::ir::{BlockLike, BlockRef, Location, Operation, TypeLike, Value, ValueLike};
use rustc_middle::mir::{BasicBlock, Body, CallSource, Operand, Place, UnwindAction};
use rustc_middle::ty::{EarlyBinder, Instance, TyCtxt, TypingEnv};
use rustc_mlir::triton::tensor::{CacheModifier, EvictionPolicy, add_ptr, arange, load, store};
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

        let arange_op: Operation<'a> = arange(self.module.context(), location, start, end)
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

    pub fn codegen_zeros_like<'tcx>(
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
            "[DEBUG] TritonCodegen::codegen_zeros_like: func: {:?} args: {:?} destination: {:?} target: {:?} unwind: {:?} call_source: {:?} fn_span: {:?}",
            func, args, destination, target, unwind, call_source, fn_span
        );

        debug_assert!(
            args.len() == 1,
            "TritonCodegen::codegen_zeros_like: args length must be 1: {:?}",
            args
        );

        let arg0 = &args[0].node;
        let _tensor = self.codegen_operand(
            tcx, instance, arg0, arg0.ty(mir, tcx), _location, mlir_block, state,
        )?;

        todo!()
        // let zeros_like_op: Operation<'a> =
        //     zeros_like(self.module.context(), _location, tensor)
        //         .map_err(|e| MlirError::CreateOperation { err: e })?
        //         .into();
        // let result = zeros_like_op.result(0).expect("ZerosLike operation result not found");
        // eprintln!("[DEBUG] AXM TritonCodegen::codegen_zeros_like: {:?}", zeros_like_op.to_string());
        // mlir_block.append_operation(zeros_like_op);
        // Ok(Some(result.into()))
    }
}
