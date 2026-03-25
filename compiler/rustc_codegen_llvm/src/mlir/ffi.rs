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

//! Bindings to the MLIR C API and our own `extern "C"` wrapper functions
//! around MLIR functionality (`MLIRRust*`).

#![allow(non_camel_case_types)]

// Opaque pointer types
unsafe extern "C" {
    pub(crate) type MLIRContext;
    pub(crate) type OpBuilder;
    pub(crate) type ModuleOp;
}

#[derive(Copy, Clone, PartialEq)]
#[repr(C)]
#[allow(dead_code)] // Variants constructed by C++.
pub(crate) enum MLIRRustResult {
    Success,
    Failure,
}

#[link(name = "llvm-wrapper", kind = "static")]
unsafe extern "C" {
    pub(crate) fn MLIRRustContextCreate() -> &'static mut MLIRContext;

    pub(crate) fn MLIRRustInitTriton(context: &MLIRContext) -> MLIRRustResult;

    pub(crate) fn MLIRRustModuleBuilderCreate(context: &MLIRContext) -> &'static mut OpBuilder;

    pub(crate) fn MLIRRustModuleCreate(builder: &OpBuilder) -> &'static mut ModuleOp;

}
