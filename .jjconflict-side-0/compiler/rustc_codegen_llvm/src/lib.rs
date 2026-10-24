//! The Rust compiler.
//!
//! # Note
//!
//! This API is completely unstable and subject to change.

// tidy-alphabetical-start
#![cfg_attr(bootstrap, feature(trim_prefix_suffix))]
#![feature(extern_types)]
#![feature(file_buffered)]
#![feature(impl_trait_in_assoc_type)]
#![feature(iter_intersperse)]
#![feature(macro_derive)]
#![feature(once_cell_try)]
#![feature(try_blocks)]
// tidy-alphabetical-end

pub use crate::back::llvm_backend::{LlvmCodegenBackend, ModuleLlvm};

mod abi;
mod allocator;
mod asm;
mod attributes;
mod back;
mod base;
mod builder;
mod callee;
mod common;
mod consts;
mod context;
mod coverageinfo;
mod debuginfo;
mod declare;
mod diagnostics;
mod intrinsic;
mod llvm;
mod llvm_util;
mod macros;
mod mono_item;
mod type_;
mod type_of;
mod typetree;
mod va_arg;
mod value;
