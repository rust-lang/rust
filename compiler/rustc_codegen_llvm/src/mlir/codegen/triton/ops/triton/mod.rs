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

use melior::ir::{BlockLike, BlockRef, Location};
use rustc_middle::mir::{Local, Terminator};
use rustc_mlir::triton::create_return;

use crate::mlir::codegen::triton::{CodegenState, TritonCodegen};
use crate::mlir::errors::MlirError;

pub mod program;
pub mod tensor;

impl<'a> TritonCodegen<'a> {
    pub fn codegen_return<'tcx>(
        &self,
        location: Location<'a>,
        _terminator: &Terminator<'tcx>,
        mlir_block: &BlockRef,
        state: &mut CodegenState<'a, 'a>,
    ) -> Result<(), MlirError> {
        println!("[DEBUG] TritonCodegen::codegen_return: ssa_values: {:?}", state.ssa_values);
        println!("[DEBUG] TritonCodegen::codegen_return: terminator: {:?}", _terminator);

        // If the return place _0 holds a tuple (multiple return values), return all fields.
        let return_values: Vec<_> = if let Some(fields) = state.tuple_fields.get(&Local::ZERO) {
            fields.to_vec()
        } else {
            state.ssa_values.get(&Local::ZERO).copied().into_iter().collect()
        };

        let return_op = create_return(self.module.context(), location, &return_values)
            .map_err(|e| MlirError::CreateOperation { err: e })?;
        eprintln!(
            "[DEBUG] AXM TritonCodegen::codegen_return: return_op: {:?}",
            return_op.as_operation().to_string()
        );
        mlir_block.append_operation(return_op.into());
        Ok(())
    }
}
