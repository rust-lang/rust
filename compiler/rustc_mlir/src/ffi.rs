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

use std::ffi::{c_char, c_void};

use mlir_sys::{MlirContext, MlirModule, MlirType};

/// Opaque handle to the Triton compiler (C ABI: `struct MlirTritonCompiler { void *ptr; }`).
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct MlirTritonCompiler {
    pub ptr: *mut c_void,
}

#[link(name = "mlir-wrapper", kind = "static")]
unsafe extern "C" {
    pub fn mlirLoadTritonDialect(context: MlirContext);

    pub fn mlirCreateTritonPointerType(pointee: MlirType, address_space: i32) -> MlirType;

    pub fn mlirApplyTritonPasses(module: MlirModule) -> bool;

    // Triton compiler opaque handle API
    pub fn mlirTritonCompilerCreate(
        context: MlirContext,
        target: *const c_char,
        options: *const c_char,
    ) -> MlirTritonCompiler;

    pub fn mlirTritonCompilerCompile(compiler: MlirTritonCompiler, module: MlirModule) -> bool;

    pub fn mlirTritonCompilerGetLLIR(compiler: MlirTritonCompiler) -> *const c_char;
    pub fn mlirTritonCompilerGetTTIR(compiler: MlirTritonCompiler) -> *const c_char;
    pub fn mlirTritonCompilerGetTTGIR(compiler: MlirTritonCompiler) -> *const c_char;
    pub fn mlirTritonCompilerGetLLVMIR(compiler: MlirTritonCompiler) -> *const c_char;
    pub fn mlirTritonCompilerGetASM(compiler: MlirTritonCompiler) -> *const c_char;
    pub fn mlirTritonCompilerGetBIN(compiler: MlirTritonCompiler) -> *const c_char;

    pub fn mlirTritonCompilerFree(compiler: MlirTritonCompiler);
}
