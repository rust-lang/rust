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
pub use rustc_codegen_ssa::interfaces::{Backend, BackendMethods, CodegenObject};
pub use self::consts::ConstMethods;
pub use self::type_::{TypeMethods, BaseTypeMethods, DerivedTypeMethods,
    LayoutTypeMethods, ArgTypeMethods};
pub use self::intrinsic::{IntrinsicCallMethods, IntrinsicDeclarationMethods};
pub use self::statics::StaticMethods;
pub use self::misc::MiscMethods;
pub use self::debuginfo::{DebugInfoMethods, DebugInfoBuilderMethods};
pub use self::abi::{AbiMethods, AbiBuilderMethods};
pub use self::declare::{DeclareMethods, PreDefineMethods};
pub use self::asm::{AsmMethods, AsmBuilderMethods};

pub trait CodegenMethods<'ll, 'tcx: 'll> :
    Backend<'ll> + TypeMethods<'ll, 'tcx> + MiscMethods<'ll, 'tcx> + ConstMethods<'ll, 'tcx> +
    StaticMethods<'ll> + DebugInfoMethods<'ll, 'tcx> + AbiMethods<'tcx> +
    IntrinsicDeclarationMethods<'ll> + DeclareMethods<'ll, 'tcx> + AsmMethods +
    PreDefineMethods<'ll, 'tcx> {}
