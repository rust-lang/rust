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

use rustc_macros::Diagnostic;

#[derive(Diagnostic, Debug)]
pub enum MlirError {
    #[diag(codegen_llvm_mlir_codegen_failed)]
    CodegenFailed { err: String },

    #[diag(codegen_llvm_mlir_create_operation_failed)]
    CreateOperation { err: rustc_mlir::errors::Error },

    #[diag(codegen_llvm_mlir_invalid_scalar_operand)]
    InvalidScalar { node: String },

    #[diag(codegen_llvm_mlir_invalid_type)]
    InvalidType { msg: String },

    #[diag(codegen_llvm_mlir_incompatible_types)]
    IncompatibleTypes { msg: String },
}
