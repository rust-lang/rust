// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

mod builder;
mod backend;
mod consts;
mod type_;
mod intrinsic;
mod statics;
mod misc;
mod debuginfo;
mod abi;
mod declare;
mod asm;

pub use self::builder::{BuilderMethods, HasCodegen};
pub use self::backend::Backend;
pub use self::consts::ConstMethods;
pub use self::type_::{TypeMethods, BaseTypeMethods, DerivedTypeMethods,
    LayoutTypeMethods, ArgTypeMethods};
pub use self::intrinsic::{IntrinsicCallMethods, IntrinsicDeclarationMethods};
pub use self::statics::StaticMethods;
pub use self::misc::MiscMethods;
pub use self::debuginfo::{DebugInfoMethods, DebugInfoBuilderMethods};
pub use self::abi::{AbiMethods, AbiBuilderMethods};
pub use self::declare::DeclareMethods;
pub use self::asm::{AsmMethods, AsmBuilderMethods};

use std::fmt;

pub trait CodegenMethods<'ll, 'tcx: 'll> :
    Backend + TypeMethods<'ll, 'tcx> + MiscMethods<'tcx> + ConstMethods<'tcx> +
    StaticMethods<'tcx> + DebugInfoMethods<'ll, 'tcx> + AbiMethods<'tcx> +
    IntrinsicDeclarationMethods + DeclareMethods<'tcx> + AsmMethods {}

pub trait CodegenObject : Copy + PartialEq + fmt::Debug {}
