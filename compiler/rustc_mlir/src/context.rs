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

//! MLIR context and module management.
//!
//! This module provides safe wrappers around MLIR context and module types,
//! integrating with both the FFI bindings and melior.

// use std::ffi::CString;
// use std::ptr;

// use crate::errors::{Error, Result};
// use crate::ffi::{self, MLIRContextRef, MLIRLocationRef, MLIRModuleRef, MLIRStringRef};

// /// RAII wrapper for an MLIR context
// pub struct Context {
//     raw: MLIRContextRef,
//     owned: bool,
// }

// impl Context {
//     /// Create a new MLIR context
//     pub fn new() -> Self {
//         let raw = unsafe { ffi::mlirRustContextCreate() };
//         Context { raw, owned: true }
//     }

//     /// Create a context wrapper from a raw pointer (non-owning)
//     ///
//     /// # Safety
//     /// The caller must ensure the pointer is valid for the lifetime of the wrapper
//     pub unsafe fn from_raw(raw: MLIRContextRef) -> Self {
//         Context { raw, owned: false }
//     }

//     /// Get the raw pointer
//     pub fn as_raw(&self) -> MLIRContextRef {
//         self.raw
//     }

//     /// Enable or disable multithreading
//     pub fn enable_multithreading(&self, enable: bool) {
//         unsafe { ffi::mlirRustContextEnableMultithreading(self.raw, enable as i32) }
//     }

//     /// Allow or disallow unregistered dialects
//     pub fn allow_unregistered_dialects(&self, allow: bool) {
//         unsafe { ffi::mlirRustContextSetAllowUnregisteredDialects(self.raw, allow as i32) }
//     }

//     /// Load all available dialects
//     pub fn load_all_available_dialects(&self) {
//         unsafe { ffi::mlirRustContextLoadAllAvailableDialects(self.raw) }
//     }

//     /// Get the number of loaded dialects
//     pub fn num_loaded_dialects(&self) -> usize {
//         unsafe { ffi::mlirRustContextGetNumLoadedDialects(self.raw) }
//     }

//     /// Get an unknown location in this context
//     pub fn unknown_location(&self) -> Location {
//         let raw = unsafe { ffi::mlirRustLocationUnknownGet(self.raw) };
//         Location { raw }
//     }

//     /// Get a file:line:col location
//     pub fn file_line_col_location(&self, filename: &str, line: u32, col: u32) -> Location {
//         let raw = unsafe {
//             ffi::mlirRustLocationFileLineColGet(
//                 self.raw,
//                 MLIRStringRef::from_str(filename),
//                 line,
//                 col,
//             )
//         };
//         Location { raw }
//     }

//     /// Get a named location
//     pub fn named_location(&self, name: &str, child: Option<&Location>) -> Location {
//         let child_raw = child.map(|l| l.raw).unwrap_or(ptr::null_mut());
//         let raw = unsafe {
//             ffi::mlirRustLocationNameGet(self.raw, MLIRStringRef::from_str(name), child_raw)
//         };
//         Location { raw }
//     }

//     /// Create an empty module
//     pub fn create_module(&self) -> Module {
//         let loc = self.unknown_location();
//         let raw = unsafe { ffi::mlirRustModuleCreateEmpty(loc.raw) };
//         Module { raw }
//     }

//     /// Parse a module from MLIR assembly
//     pub fn parse_module(&self, source: &str) -> Result<Module> {
//         let raw =
//             unsafe { ffi::mlirRustModuleCreateParse(self.raw, MLIRStringRef::from_str(source)) };
//         if raw.is_null() {
//             Err(Error::ParseError("Failed to parse MLIR module".to_string()))
//         } else {
//             Ok(Module { raw })
//         }
//     }

//     /// Create an integer type
//     pub fn integer_type(&self, width: u32) -> Type {
//         let raw = unsafe { ffi::mlirRustIntegerTypeGet(self.raw, width) };
//         Type { raw }
//     }

//     /// Create a signless integer type
//     pub fn integer_type_signless(&self, width: u32) -> Type {
//         let raw = unsafe { ffi::mlirRustIntegerTypeSignlessGet(self.raw, width) };
//         Type { raw }
//     }

//     /// Create a signed integer type
//     pub fn integer_type_signed(&self, width: u32) -> Type {
//         let raw = unsafe { ffi::mlirRustIntegerTypeSignedGet(self.raw, width) };
//         Type { raw }
//     }

//     /// Create an unsigned integer type
//     pub fn integer_type_unsigned(&self, width: u32) -> Type {
//         let raw = unsafe { ffi::mlirRustIntegerTypeUnsignedGet(self.raw, width) };
//         Type { raw }
//     }

//     /// Create an index type
//     pub fn index_type(&self) -> Type {
//         let raw = unsafe { ffi::mlirRustIndexTypeGet(self.raw) };
//         Type { raw }
//     }

//     /// Create an f16 type
//     pub fn f16_type(&self) -> Type {
//         let raw = unsafe { ffi::mlirRustF16TypeGet(self.raw) };
//         Type { raw }
//     }

//     /// Create a bf16 type
//     pub fn bf16_type(&self) -> Type {
//         let raw = unsafe { ffi::mlirRustBF16TypeGet(self.raw) };
//         Type { raw }
//     }

//     /// Create an f32 type
//     pub fn f32_type(&self) -> Type {
//         let raw = unsafe { ffi::mlirRustF32TypeGet(self.raw) };
//         Type { raw }
//     }

//     /// Create an f64 type
//     pub fn f64_type(&self) -> Type {
//         let raw = unsafe { ffi::mlirRustF64TypeGet(self.raw) };
//         Type { raw }
//     }

//     /// Create a none type
//     pub fn none_type(&self) -> Type {
//         let raw = unsafe { ffi::mlirRustNoneTypeGet(self.raw) };
//         Type { raw }
//     }

//     /// Create a unit attribute
//     pub fn unit_attr(&self) -> Attribute {
//         let raw = unsafe { ffi::mlirRustUnitAttrGet(self.raw) };
//         Attribute { raw }
//     }

//     /// Create a bool attribute
//     pub fn bool_attr(&self, value: bool) -> Attribute {
//         let raw = unsafe { ffi::mlirRustBoolAttrGet(self.raw, value as i32) };
//         Attribute { raw }
//     }

//     /// Create a string attribute
//     pub fn string_attr(&self, value: &str) -> Attribute {
//         let raw = unsafe { ffi::mlirRustStringAttrGet(self.raw, MLIRStringRef::from_str(value)) };
//         Attribute { raw }
//     }

//     /// Create a flat symbol reference attribute
//     pub fn flat_symbol_ref_attr(&self, symbol: &str) -> Attribute {
//         let raw =
//             unsafe { ffi::mlirRustFlatSymbolRefAttrGet(self.raw, MLIRStringRef::from_str(symbol)) };
//         Attribute { raw }
//     }
// }

// impl Default for Context {
//     fn default() -> Self {
//         Self::new()
//     }
// }

// impl Drop for Context {
//     fn drop(&mut self) {
//         if self.owned && !self.raw.is_null() {
//             unsafe { ffi::mlirRustContextDestroy(self.raw) }
//         }
//     }
// }

// /// RAII wrapper for an MLIR location
// pub struct Location {
//     raw: MLIRLocationRef,
// }

// impl Location {
//     /// Get the raw pointer
//     pub fn as_raw(&self) -> MLIRLocationRef {
//         self.raw
//     }
// }

// /// RAII wrapper for an MLIR module
// pub struct Module {
//     raw: MLIRModuleRef,
// }

// impl Module {
//     /// Get the raw pointer
//     pub fn as_raw(&self) -> MLIRModuleRef {
//         self.raw
//     }

//     /// Get the module body block
//     pub fn body(&self) -> ffi::MLIRBlockRef {
//         unsafe { ffi::mlirRustModuleGetBody(self.raw) }
//     }

//     /// Get the module as an operation
//     pub fn operation(&self) -> ffi::MLIROperationRef {
//         unsafe { ffi::mlirRustModuleGetOperation(self.raw) }
//     }

//     /// Get the context
//     pub fn context(&self) -> Context {
//         let raw = unsafe { ffi::mlirRustModuleGetContext(self.raw) };
//         unsafe { Context::from_raw(raw) }
//     }

//     /// Print the module to a string
//     pub fn to_string(&self) -> String {
//         unsafe {
//             let ptr = ffi::mlirRustModulePrint(self.raw);
//             ffi::c_str_to_string(ptr)
//         }
//     }

//     /// Verify the module
//     pub fn verify(&self) -> Result<()> {
//         let result = unsafe { ffi::mlirRustModuleVerify(self.raw) };
//         if result.is_success() { Ok(()) } else { Err(Error::VerificationFailed) }
//     }
// }

// impl Drop for Module {
//     fn drop(&mut self) {
//         if !self.raw.is_null() {
//             unsafe { ffi::mlirRustModuleDestroy(self.raw) }
//         }
//     }
// }

// impl std::fmt::Display for Module {
//     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
//         write!(f, "{}", self.to_string())
//     }
// }

// /// Wrapper for an MLIR type
// pub struct Type {
//     raw: ffi::MLIRTypeRef,
// }

// impl Type {
//     /// Get the raw pointer
//     pub fn as_raw(&self) -> ffi::MLIRTypeRef {
//         self.raw
//     }

//     /// Create from a raw pointer
//     pub fn from_raw(raw: ffi::MLIRTypeRef) -> Self {
//         Type { raw }
//     }

//     /// Check if this is an integer type
//     pub fn is_integer(&self) -> bool {
//         unsafe { ffi::mlirRustTypeIsInteger(self.raw) != 0 }
//     }

//     /// Check if this is a float type
//     pub fn is_float(&self) -> bool {
//         unsafe { ffi::mlirRustTypeIsFloat(self.raw) != 0 }
//     }

//     /// Check if this is an index type
//     pub fn is_index(&self) -> bool {
//         unsafe { ffi::mlirRustTypeIsIndex(self.raw) != 0 }
//     }

//     /// Get the context
//     pub fn context(&self) -> Context {
//         let raw = unsafe { ffi::mlirRustTypeGetContext(self.raw) };
//         unsafe { Context::from_raw(raw) }
//     }

//     /// Create an integer attribute with this type
//     pub fn integer_attr(&self, value: i64) -> Attribute {
//         let raw = unsafe { ffi::mlirRustIntegerAttrGet(self.raw, value) };
//         Attribute { raw }
//     }

//     /// Create a float attribute with this type
//     pub fn float_attr(&self, value: f64) -> Attribute {
//         let raw = unsafe { ffi::mlirRustFloatAttrGet(self.raw, value) };
//         Attribute { raw }
//     }

//     /// Create a type attribute from this type
//     pub fn type_attr(&self) -> Attribute {
//         let raw = unsafe { ffi::mlirRustTypeAttrGet(self.raw) };
//         Attribute { raw }
//     }
// }

// /// Wrapper for an MLIR attribute
// pub struct Attribute {
//     raw: ffi::MLIRAttributeRef,
// }

// impl Attribute {
//     /// Get the raw pointer
//     pub fn as_raw(&self) -> ffi::MLIRAttributeRef {
//         self.raw
//     }

//     /// Create from a raw pointer
//     pub fn from_raw(raw: ffi::MLIRAttributeRef) -> Self {
//         Attribute { raw }
//     }
// }

// /// Wrapper for an MLIR value
// pub struct Value {
//     raw: ffi::MLIRValueRef,
// }

// impl Value {
//     /// Get the raw pointer
//     pub fn as_raw(&self) -> ffi::MLIRValueRef {
//         self.raw
//     }

//     /// Create from a raw pointer
//     pub fn from_raw(raw: ffi::MLIRValueRef) -> Self {
//         Value { raw }
//     }

//     /// Get the type of this value
//     pub fn get_type(&self) -> Type {
//         let raw = unsafe { ffi::mlirRustValueGetType(self.raw) };
//         Type { raw }
//     }

//     /// Check if this value is a block argument
//     pub fn is_block_argument(&self) -> bool {
//         unsafe { ffi::mlirRustValueIsBlockArgument(self.raw) != 0 }
//     }

//     /// Check if this value is an operation result
//     pub fn is_op_result(&self) -> bool {
//         unsafe { ffi::mlirRustValueIsOpResult(self.raw) != 0 }
//     }

//     /// Print the value to a string
//     pub fn to_string(&self) -> String {
//         unsafe {
//             let ptr = ffi::mlirRustValuePrint(self.raw);
//             ffi::c_str_to_string(ptr)
//         }
//     }
// }

// impl std::fmt::Display for Value {
//     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
//         write!(f, "{}", self.to_string())
//     }
// }

// /// Wrapper for an MLIR operation
// pub struct Operation {
//     raw: ffi::MLIROperationRef,
// }

// impl Operation {
//     /// Get the raw pointer
//     pub fn as_raw(&self) -> ffi::MLIROperationRef {
//         self.raw
//     }

//     /// Create from a raw pointer
//     pub fn from_raw(raw: ffi::MLIROperationRef) -> Self {
//         Operation { raw }
//     }

//     /// Get the number of results
//     pub fn num_results(&self) -> usize {
//         unsafe { ffi::mlirRustOperationGetNumResults(self.raw) }
//     }

//     /// Get a result by index
//     pub fn result(&self, index: usize) -> Value {
//         let raw = unsafe { ffi::mlirRustOperationGetResult(self.raw, index) };
//         Value { raw }
//     }

//     /// Get the number of operands
//     pub fn num_operands(&self) -> usize {
//         unsafe { ffi::mlirRustOperationGetNumOperands(self.raw) }
//     }

//     /// Get an operand by index
//     pub fn operand(&self, index: usize) -> Value {
//         let raw = unsafe { ffi::mlirRustOperationGetOperand(self.raw, index) };
//         Value { raw }
//     }

//     /// Get the number of regions
//     pub fn num_regions(&self) -> usize {
//         unsafe { ffi::mlirRustOperationGetNumRegions(self.raw) }
//     }

//     /// Get a region by index
//     pub fn region(&self, index: usize) -> ffi::MLIRRegionRef {
//         unsafe { ffi::mlirRustOperationGetRegion(self.raw, index) }
//     }

//     /// Get the parent block
//     pub fn block(&self) -> ffi::MLIRBlockRef {
//         unsafe { ffi::mlirRustOperationGetBlock(self.raw) }
//     }

//     /// Print the operation to a string
//     pub fn to_string(&self) -> String {
//         unsafe {
//             let ptr = ffi::mlirRustOperationPrint(self.raw);
//             ffi::c_str_to_string(ptr)
//         }
//     }
// }

// impl std::fmt::Display for Operation {
//     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
//         write!(f, "{}", self.to_string())
//     }
// }
