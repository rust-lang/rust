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

//! High-level builder API for Triton operations.
//!
//! This module provides a builder interface for constructing Triton IR,
//! using the FFI bindings for Triton-specific operations and melior for
//! general MLIR construction.

// use crate::context::{Context, Location, Module, Operation, Type, Value};
// use crate::ffi::{self, MLIROpBuilderRef, MLIRStringRef};
// use crate::triton::{self, CacheModifier, EvictionPolicy, ProgramAxis};
// use std::ptr;

// /// Builder for Triton operations
// ///
// /// This wraps an MLIR OpBuilder and provides methods for creating Triton operations.
// /// For general MLIR construction, use melior's OperationBuilder directly.
// pub struct TritonBuilder {
//     raw: MLIROpBuilderRef,
// }

// impl TritonBuilder {
//     /// Create a new builder for a context
//     pub fn new(ctx: &Context) -> Self {
//         let raw = unsafe { ffi::mlirRustOpBuilderCreate(ctx.as_raw()) };
//         TritonBuilder { raw }
//     }

//     /// Create a builder positioned at the end of a block
//     pub fn at_block_end(block: ffi::MLIRBlockRef) -> Self {
//         let raw = unsafe { ffi::mlirRustOpBuilderCreateAtBlockEnd(block) };
//         TritonBuilder { raw }
//     }

//     /// Get the raw pointer
//     pub fn as_raw(&self) -> MLIROpBuilderRef {
//         self.raw
//     }

//     /// Set insertion point to the end of a block
//     pub fn set_insertion_point_to_end(&self, block: ffi::MLIRBlockRef) {
//         unsafe { ffi::mlirRustOpBuilderSetInsertionPointToEnd(self.raw, block) }
//     }

//     /// Set insertion point to the start of a block
//     pub fn set_insertion_point_to_start(&self, block: ffi::MLIRBlockRef) {
//         unsafe { ffi::mlirRustOpBuilderSetInsertionPointToStart(self.raw, block) }
//     }

//     /// Set insertion point before an operation
//     pub fn set_insertion_point(&self, op: ffi::MLIROperationRef) {
//         unsafe { ffi::mlirRustOpBuilderSetInsertionPoint(self.raw, op) }
//     }

//     /// Set insertion point after an operation
//     pub fn set_insertion_point_after(&self, op: ffi::MLIROperationRef) {
//         unsafe { ffi::mlirRustOpBuilderSetInsertionPointAfter(self.raw, op) }
//     }

//     /// Get the current insertion block
//     pub fn insertion_block(&self) -> ffi::MLIRBlockRef {
//         unsafe { ffi::mlirRustOpBuilderGetInsertionBlock(self.raw) }
//     }

//     /// Get the context
//     pub fn context(&self) -> Context {
//         let raw = unsafe { ffi::mlirRustOpBuilderGetContext(self.raw) };
//         unsafe { Context::from_raw(raw) }
//     }

//     //=========================================================================
//     // Triton operations
//     //=========================================================================

//     /// Create a tt.make_range operation
//     ///
//     /// Creates a 1D tensor with values [start, start+1, ..., end-1]
//     pub fn make_range(&self, loc: &Location, start: i32, end: i32) -> Operation {
//         let raw = unsafe { ffi::tritonRustMakeRangeOp(self.raw, loc.as_raw(), start, end) };
//         Operation::from_raw(raw)
//     }

//     /// Create a tt.splat operation (broadcast scalar to tensor)
//     pub fn splat(&self, loc: &Location, src: &Value, result_type: &Type) -> Operation {
//         let raw = unsafe {
//             ffi::tritonRustSplatOp(self.raw, loc.as_raw(), src.as_raw(), result_type.as_raw())
//         };
//         Operation::from_raw(raw)
//     }

//     /// Create a tt.broadcast operation
//     pub fn broadcast(&self, loc: &Location, src: &Value, result_type: &Type) -> Operation {
//         let raw = unsafe {
//             ffi::tritonRustBroadcastOp(self.raw, loc.as_raw(), src.as_raw(), result_type.as_raw())
//         };
//         Operation::from_raw(raw)
//     }

//     /// Create a tt.expand_dims operation
//     pub fn expand_dims(&self, loc: &Location, src: &Value, axis: i32) -> Operation {
//         let raw = unsafe { ffi::tritonRustExpandDimsOp(self.raw, loc.as_raw(), src.as_raw(), axis) };
//         Operation::from_raw(raw)
//     }

//     /// Create a tt.addptr operation (pointer arithmetic)
//     pub fn addptr(&self, loc: &Location, ptr: &Value, offset: &Value) -> Operation {
//         let raw = unsafe {
//             ffi::tritonRustAddPtrOp(self.raw, loc.as_raw(), ptr.as_raw(), offset.as_raw())
//         };
//         Operation::from_raw(raw)
//     }

//     /// Create a tt.load operation
//     pub fn load(
//         &self,
//         loc: &Location,
//         ptr: &Value,
//         cache: Option<CacheModifier>,
//         evict: Option<EvictionPolicy>,
//         is_volatile: bool,
//     ) -> Operation {
//         let ctx = self.context();
//         let cache_attr = cache
//             .map(|c| triton::attr::cache_modifier(ctx.as_raw(), c))
//             .unwrap_or(ptr::null_mut());
//         let evict_attr = evict
//             .map(|e| triton::attr::eviction_policy(ctx.as_raw(), e))
//             .unwrap_or(ptr::null_mut());

//         let raw = unsafe {
//             ffi::tritonRustLoadOp(
//                 self.raw,
//                 loc.as_raw(),
//                 ptr.as_raw(),
//                 cache_attr,
//                 evict_attr,
//                 is_volatile as i32,
//             )
//         };
//         Operation::from_raw(raw)
//     }

//     /// Create a tt.store operation
//     pub fn store(
//         &self,
//         loc: &Location,
//         ptr: &Value,
//         value: &Value,
//         cache: Option<CacheModifier>,
//     ) -> Operation {
//         let ctx = self.context();
//         let cache_attr = cache
//             .map(|c| triton::attr::cache_modifier(ctx.as_raw(), c))
//             .unwrap_or(ptr::null_mut());

//         let raw = unsafe {
//             ffi::tritonRustStoreOp(
//                 self.raw,
//                 loc.as_raw(),
//                 ptr.as_raw(),
//                 value.as_raw(),
//                 cache_attr,
//             )
//         };
//         Operation::from_raw(raw)
//     }

//     /// Create a tt.dot operation (matrix multiply)
//     pub fn dot(
//         &self,
//         loc: &Location,
//         a: &Value,
//         b: &Value,
//         c: &Value,
//         allow_tf32: bool,
//         max_num_imprecise_acc: i32,
//     ) -> Operation {
//         let raw = unsafe {
//             ffi::tritonRustDotOp(
//                 self.raw,
//                 loc.as_raw(),
//                 a.as_raw(),
//                 b.as_raw(),
//                 c.as_raw(),
//                 allow_tf32 as i32,
//                 max_num_imprecise_acc,
//             )
//         };
//         Operation::from_raw(raw)
//     }

//     /// Create a tt.reduce operation
//     ///
//     /// Note: The reduce operation's region must be populated with combine operations
//     pub fn reduce(&self, loc: &Location, operand: &Value, axis: i32) -> Operation {
//         let raw =
//             unsafe { ffi::tritonRustReduceOp(self.raw, loc.as_raw(), operand.as_raw(), axis) };
//         Operation::from_raw(raw)
//     }

//     /// Create a tt.get_program_id operation
//     pub fn get_program_id(&self, loc: &Location, axis: ProgramAxis) -> Operation {
//         let raw =
//             unsafe { ffi::tritonRustGetProgramIdOp(self.raw, loc.as_raw(), axis as i32) };
//         Operation::from_raw(raw)
//     }

//     /// Create a tt.get_num_programs operation
//     pub fn get_num_programs(&self, loc: &Location, axis: ProgramAxis) -> Operation {
//         let raw =
//             unsafe { ffi::tritonRustGetNumProgramsOp(self.raw, loc.as_raw(), axis as i32) };
//         Operation::from_raw(raw)
//     }

//     /// Create a tt.func operation
//     pub fn func(
//         &self,
//         loc: &Location,
//         name: &str,
//         input_types: &[&Type],
//         result_types: &[&Type],
//     ) -> Operation {
//         let input_raws: Vec<_> = input_types.iter().map(|t| t.as_raw()).collect();
//         let result_raws: Vec<_> = result_types.iter().map(|t| t.as_raw()).collect();

//         let raw = unsafe {
//             ffi::tritonRustFuncOp(
//                 self.raw,
//                 loc.as_raw(),
//                 MLIRStringRef::from_str(name),
//                 input_raws.len(),
//                 input_raws.as_ptr(),
//                 result_raws.len(),
//                 result_raws.as_ptr(),
//             )
//         };
//         Operation::from_raw(raw)
//     }

//     /// Create a tt.return operation
//     pub fn return_op(&self, loc: &Location, values: &[&Value]) -> Operation {
//         let value_raws: Vec<_> = values.iter().map(|v| v.as_raw()).collect();

//         let raw = unsafe {
//             ffi::tritonRustReturnOp(
//                 self.raw,
//                 loc.as_raw(),
//                 value_raws.len(),
//                 value_raws.as_ptr(),
//             )
//         };
//         Operation::from_raw(raw)
//     }

//     //=========================================================================
//     // Triton GPU operations
//     //=========================================================================

//     /// Create a triton_gpu.convert_layout operation
//     pub fn convert_layout(&self, loc: &Location, src: &Value, dst_type: &Type) -> Operation {
//         let raw = unsafe {
//             ffi::tritonRustConvertLayoutOp(
//                 self.raw,
//                 loc.as_raw(),
//                 src.as_raw(),
//                 dst_type.as_raw(),
//             )
//         };
//         Operation::from_raw(raw)
//     }

//     /// Create a triton_gpu.alloc_tensor operation (shared memory allocation)
//     pub fn alloc_tensor(&self, loc: &Location, tensor_type: &Type) -> Operation {
//         let raw = unsafe {
//             ffi::tritonRustAllocTensorOp(self.raw, loc.as_raw(), tensor_type.as_raw())
//         };
//         Operation::from_raw(raw)
//     }

//     /// Create a triton_gpu.insert_slice_async operation
//     pub fn insert_slice_async(
//         &self,
//         loc: &Location,
//         src: &Value,
//         dst: &Value,
//         index: &Value,
//         cache: Option<CacheModifier>,
//         evict: Option<EvictionPolicy>,
//         is_volatile: bool,
//     ) -> Operation {
//         let ctx = self.context();
//         let cache_attr = cache
//             .map(|c| triton::attr::cache_modifier(ctx.as_raw(), c))
//             .unwrap_or(ptr::null_mut());
//         let evict_attr = evict
//             .map(|e| triton::attr::eviction_policy(ctx.as_raw(), e))
//             .unwrap_or(ptr::null_mut());

//         let raw = unsafe {
//             ffi::tritonRustInsertSliceAsyncOp(
//                 self.raw,
//                 loc.as_raw(),
//                 src.as_raw(),
//                 dst.as_raw(),
//                 index.as_raw(),
//                 cache_attr,
//                 evict_attr,
//                 is_volatile as i32,
//             )
//         };
//         Operation::from_raw(raw)
//     }

//     /// Create a triton_gpu.async_wait operation
//     pub fn async_wait(&self, loc: &Location, num: i32) -> Operation {
//         let raw = unsafe { ffi::tritonRustAsyncWaitOp(self.raw, loc.as_raw(), num) };
//         Operation::from_raw(raw)
//     }
// }

// impl Drop for TritonBuilder {
//     fn drop(&mut self) {
//         if !self.raw.is_null() {
//             unsafe { ffi::mlirRustOpBuilderDestroy(self.raw) }
//         }
//     }
// }

// /// Extension trait for using melior with Triton types
// ///
// /// This allows mixing melior's OperationBuilder with Triton-specific FFI bindings.
// pub trait MeliorTritonExt {
//     /// Get the raw context pointer for use with Triton FFI
//     fn triton_context(&self) -> ffi::MLIRContextRef;
// }

// // Note: Implementation of MeliorTritonExt for melior types would go here,
// // but requires melior to be properly configured. For now, users can access
// // the raw context via melior's API and use the triton module functions directly.
