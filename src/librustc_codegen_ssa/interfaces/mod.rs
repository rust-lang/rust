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

mod abi;
mod asm;
mod backend;
mod builder;
mod consts;
mod debuginfo;
mod declare;
mod intrinsic;
mod misc;
mod statics;
mod type_;

pub use self::abi::{AbiBuilderMethods, AbiMethods};
pub use self::asm::{AsmBuilderMethods, AsmMethods};
pub use self::backend::{Backend, BackendTypes, ExtraBackendMethods};
pub use self::builder::BuilderMethods;
pub use self::consts::ConstMethods;
pub use self::debuginfo::{DebugInfoBuilderMethods, DebugInfoMethods};
pub use self::declare::{DeclareMethods, PreDefineMethods};
pub use self::intrinsic::{IntrinsicCallMethods, IntrinsicDeclarationMethods};
pub use self::misc::MiscMethods;
pub use self::statics::StaticMethods;
pub use self::type_::{
    ArgTypeMethods, BaseTypeMethods, DerivedTypeMethods, LayoutTypeMethods, TypeMethods,
};

use std::fmt;

pub trait CodegenObject: Copy + PartialEq + fmt::Debug {}
impl<T: Copy + PartialEq + fmt::Debug> CodegenObject for T {}

pub trait CodegenMethods<'tcx>:
    Backend<'tcx>
    + TypeMethods<'tcx>
    + MiscMethods<'tcx>
    + ConstMethods<'tcx>
    + StaticMethods<'tcx>
    + DebugInfoMethods<'tcx>
    + AbiMethods<'tcx>
    + IntrinsicDeclarationMethods<'tcx>
    + DeclareMethods<'tcx>
    + AsmMethods<'tcx>
    + PreDefineMethods<'tcx>
{
}

impl<'tcx, T> CodegenMethods<'tcx> for T where
    Self: Backend<'tcx>
        + TypeMethods<'tcx>
        + MiscMethods<'tcx>
        + ConstMethods<'tcx>
        + StaticMethods<'tcx>
        + DebugInfoMethods<'tcx>
        + AbiMethods<'tcx>
        + IntrinsicDeclarationMethods<'tcx>
        + DeclareMethods<'tcx>
        + AsmMethods<'tcx>
        + PreDefineMethods<'tcx>
{}

pub trait HasCodegen<'tcx>: Backend<'tcx> {
    type CodegenCx: CodegenMethods<'tcx>
        + BackendTypes<
            Value = Self::Value,
            BasicBlock = Self::BasicBlock,
            Type = Self::Type,
            Context = Self::Context,
            Funclet = Self::Funclet,
            DIScope = Self::DIScope,
        >;
}
