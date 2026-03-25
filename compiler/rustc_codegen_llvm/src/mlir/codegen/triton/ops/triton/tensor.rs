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
use rustc_middle::mir::interpret::Scalar;
use rustc_middle::mir::{
    BasicBlock, Body, CallSource, Const, ConstValue, Operand, Place, UnwindAction,
};
use rustc_middle::ty::{Instance, TyCtxt};
use rustc_mlir::shared::arith::{Int, create_int_constant};
use rustc_mlir::triton::program::{ProgramAxis, create_get_program_id};
use rustc_mlir::triton::tensor::{add_ptr, arange, load, store};
use rustc_span::Span;
use rustc_span::source_map::Spanned;

use crate::mlir::codegen::triton::{SsaValues, TritonCodegen};
use crate::mlir::errors::MlirError;

impl<'a> TritonCodegen<'a> {
    pub fn codegen_arange<'tcx>(
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
        mlir_block: &BlockRef<'a, 'a>,
        ssa_values: &mut SsaValues<'a, 'a>,
    ) -> Result<Option<Value<'a, 'a>>, MlirError> {
        println!(
            "[DEBUG] TritonCodegen::codegen_program_id: func: {:?} args: {:?} destination: {:?} target: {:?} unwind: {:?} call_source: {:?} fn_span: {:?}",
            func, args, destination, target, unwind, call_source, fn_span
        );

        debug_assert!(
            args.len() == 2,
            "TritonCodegen::codegen_arange: args length must be 2: {:?}",
            args
        );

        let start = self.to_scalar_int(tcx, instance, &args[0].node)?.to_i32();
        let end = self.to_scalar_int(tcx, instance, &args[1].node)?.to_i32();

        let arange_op: Operation<'a> =
            arange(self.module.context(), Location::unknown(self.module.context()), start, end)
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
        mlir_block: &BlockRef<'a, 'a>,
        ssa_values: &mut SsaValues<'a, 'a>,
    ) -> Result<Option<Value<'a, 'a>>, MlirError> {
        debug_assert!(
            args.len() == 2,
            "TritonCodegen::codegen_add_offsets_call: args length must be 2"
        );

        let arg0 = &args[0].node;
        let arg1 = &args[1].node;

        let ptr =
            self.codegen_operand(tcx, instance, arg0, arg0.ty(mir, tcx), mlir_block, ssa_values)?;
        let offset =
            self.codegen_operand(tcx, instance, arg1, arg1.ty(mir, tcx), mlir_block, ssa_values)?;

        debug_assert!(
            offset.r#type().is_tensor(),
            "TritonCodegen::codegen_add_offset: rhs is not a tensor"
        );

        let location = Location::unknown(self.module.context());
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
        mlir_block: &BlockRef<'a, 'a>,
        ssa_values: &mut SsaValues<'a, 'a>,
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

        let ptr =
            self.codegen_operand(tcx, instance, arg0, arg0.ty(mir, tcx), mlir_block, ssa_values)?;
        let mask =
            self.codegen_operand(tcx, instance, arg1, arg1.ty(mir, tcx), mlir_block, ssa_values)?;

        let load_op: Operation<'a> =
            load(self.module.context(), Location::unknown(self.module.context()), ptr, mask)
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
        mlir_block: &BlockRef<'a, 'a>,
        ssa_values: &mut SsaValues<'a, 'a>,
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

        let dest =
            self.codegen_operand(tcx, instance, arg0, arg0.ty(mir, tcx), mlir_block, ssa_values)?;
        let src =
            self.codegen_operand(tcx, instance, arg1, arg1.ty(mir, tcx), mlir_block, ssa_values)?;
        let mask =
            self.codegen_operand(tcx, instance, arg2, arg2.ty(mir, tcx), mlir_block, ssa_values)?;

        let store_op: Operation<'a> =
            store(self.module.context(), Location::unknown(self.module.context()), dest, src, mask)
                .map_err(|e| MlirError::CreateOperation { err: e })?
                .into();

        eprintln!("[DEBUG] AXM TritonCodegen::codegen_store: {:?}", store_op.to_string());
        mlir_block.append_operation(store_op);

        Ok(None)
    }
}
