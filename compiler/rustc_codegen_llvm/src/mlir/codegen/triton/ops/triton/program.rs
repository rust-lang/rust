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
use melior::ir::{BlockLike, BlockRef, Location, Operation, Value};
use rustc_middle::mir::{BasicBlock, Body, CallSource, Operand, Place, UnwindAction};
use rustc_middle::ty::{Instance, TyCtxt};
use rustc_mlir::triton::program::{ProgramAxis, create_get_program_id};
use rustc_span::Span;
use rustc_span::source_map::Spanned;

use crate::mlir::codegen::triton::{SsaValues, TritonCodegen};
use crate::mlir::errors::MlirError;

impl<'a> TritonCodegen<'a> {
    pub fn codegen_program_id<'tcx>(
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
        mlir_block: &BlockRef,
        _ssa_values: &mut SsaValues<'a, 'a>,
    ) -> Result<Option<Value<'a, 'a>>, MlirError> {
        debug_assert!(args.len() == 1, "TritonCodegen::codegen_program_id: args length must be 1");

        let value = self.to_scalar_int(tcx, instance, &args[0].node)?;
        let axis = <ProgramAxis as From<i32>>::from(value.to_bits(value.size()) as i32);

        let program_id_op: Operation<'a> =
            create_get_program_id(self.module.context(), location, axis)
                .map_err(|e| MlirError::CreateOperation { err: e })?
                .into();

        let result = program_id_op.result(0).expect("Program ID operation result not found");
        eprintln!(
            "[DEBUG] AXM TritonCodegen::codegen_program_id: result: {:?}",
            program_id_op.to_string()
        );
        mlir_block.append_operation(program_id_op);
        Ok(Some(result.into()))
    }
}
