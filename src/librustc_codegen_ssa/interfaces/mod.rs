// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Interface of a Rust codegen backend
//!
//! This crate defines all the traits that have to be implemented by a codegen backend in order to
//! use the backend-agnostic codegen code in `rustc_codegen_ssa`.
//!
//! The interface is designed around two backend-specific data structures, the codegen context and
//! the builder. The codegen context is supposed to be read-only after its creation and during the
//! actual codegen, while the builder stores the information about the function during codegen and
//! is used to produce the instructions of the backend IR.
//!
//! Finaly, a third `Backend` structure has to implement methods related to how codegen information
//! is passed to the backend, especially for asynchronous compilation.
//!
//! The traits contain associated types that are backend-specific, such as the backend's value or
//! basic blocks.

use std::fmt;
mod backend;
mod misc;
mod statics;
mod declare;
mod builder;
mod consts;
mod type_;
mod intrinsic;
mod debuginfo;
mod abi;
mod asm;

pub use self::backend::{Backend, ExtraBackendMethods};
pub use self::misc::MiscMethods;
pub use self::statics::StaticMethods;
pub use self::declare::{DeclareMethods, PreDefineMethods};
pub use self::builder::{BuilderMethods, HasCodegen};
pub use self::consts::ConstMethods;
pub use self::type_::{TypeMethods, BaseTypeMethods, DerivedTypeMethods,
    LayoutTypeMethods, ArgTypeMethods};
pub use self::intrinsic::{IntrinsicCallMethods, IntrinsicDeclarationMethods};
pub use self::debuginfo::{DebugInfoMethods, DebugInfoBuilderMethods};
pub use self::abi::{AbiMethods, AbiBuilderMethods};
pub use self::asm::{AsmMethods, AsmBuilderMethods};


pub trait CodegenObject : Copy + PartialEq + fmt::Debug {}

pub trait CodegenMethods<'ll, 'tcx: 'll> :
    Backend<'ll> + TypeMethods<'ll, 'tcx> + MiscMethods<'ll, 'tcx> + ConstMethods<'ll, 'tcx> +
    StaticMethods<'ll> + DebugInfoMethods<'ll, 'tcx> + AbiMethods<'tcx> +
    IntrinsicDeclarationMethods<'ll> + DeclareMethods<'ll, 'tcx> + AsmMethods +
    PreDefineMethods<'ll, 'tcx> {}
