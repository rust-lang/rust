// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

mod backend;
mod builder;
mod consts;
mod debuginfo;
mod intrinsic;
mod misc;
mod statics;
mod type_;

pub use self::backend::{Backend, BackendTypes};
pub use self::builder::BuilderMethods;
pub use self::consts::ConstMethods;
pub use self::debuginfo::DebugInfoMethods;
pub use self::intrinsic::{IntrinsicCallMethods, IntrinsicDeclarationMethods};
pub use self::misc::MiscMethods;
pub use self::statics::StaticMethods;
pub use self::type_::{BaseTypeMethods, DerivedTypeMethods, LayoutTypeMethods, TypeMethods};

use std::fmt;

pub trait CodegenMethods<'tcx>:
    Backend<'tcx>
    + TypeMethods<'tcx>
    + MiscMethods<'tcx>
    + ConstMethods<'tcx>
    + StaticMethods<'tcx>
    + DebugInfoMethods<'tcx>
{
}

impl<'tcx, T> CodegenMethods<'tcx> for T where
    Self: Backend<'tcx>
        + TypeMethods<'tcx>
        + MiscMethods<'tcx>
        + ConstMethods<'tcx>
        + StaticMethods<'tcx>
        + DebugInfoMethods<'tcx>
{}

pub trait HasCodegen<'tcx>: Backend<'tcx> {
    type CodegenCx: CodegenMethods<'tcx>
        + BackendTypes<
            Value = Self::Value,
            BasicBlock = Self::BasicBlock,
            Type = Self::Type,
            Context = Self::Context,
        >;
}

pub trait CodegenObject: Copy + PartialEq + fmt::Debug {}
impl<T: Copy + PartialEq + fmt::Debug> CodegenObject for T {}
