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

//! Rust bindings for MLIR and Triton dialects.
//!
//! This crate provides:
//! - FFI bindings to the mlir-wrapper C++ library for Triton-specific types
//! - Re-exports of melior for general MLIR construction
//! - Helper types for building Triton IR from Rust
//!
//! # Architecture
//!
//! The crate is structured in layers:
//! 1. `ffi` - Raw C FFI bindings to mlir-wrapper
//! 2. `triton` - Safe Rust wrappers around Triton types
//! 3. `builder` - High-level builder API using melior's OperationBuilder
//!
//! # Example
//!
//! ```ignore
//! use rustc_mlir::{context::Context, triton};
//!
//! let context = Context::new();
//! triton::register_dialects(&context);
//!
//! // Use melior's OperationBuilder for construction
//! let module = context.create_module("my_kernel");
//! ```

#![allow(non_camel_case_types)]
#![allow(non_upper_case_globals)]
#![allow(dead_code)]

pub mod builder;
pub mod context;
pub mod errors;
pub mod ffi;
pub mod shared;

#[cfg(test)]
mod test;

#[cfg(feature = "triton")]
pub mod triton;

// Re-export melior for convenience
pub use melior;
use melior::Context;
use melior::dialect::DialectRegistry;
use melior::utility::register_all_dialects;

rustc_fluent_macro::fluent_messages! { "../messages.ftl" }

pub fn load_all_dialects(context: &Context) {
    let registry = DialectRegistry::new();
    register_all_dialects(&registry);
    context.append_dialect_registry(&registry);
    context.load_all_available_dialects();
}
