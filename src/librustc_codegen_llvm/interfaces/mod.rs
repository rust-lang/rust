// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

mod abi;
mod asm;
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
pub use rustc_codegen_utils::interfaces::{Backend, BackendMethods, BackendTypes, CodegenObject};

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
