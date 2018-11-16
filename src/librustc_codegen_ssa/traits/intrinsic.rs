// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use super::Backend;
use super::HasCodegen;
use mir::operand::OperandRef;
use rustc::ty::Ty;
use rustc_target::abi::call::FnType;
use syntax_pos::Span;

pub trait IntrinsicCallMethods<'tcx>: HasCodegen<'tcx> {
    /// Remember to add all intrinsics here, in librustc_typeck/check/mod.rs,
    /// and in libcore/intrinsics.rs; if you need access to any llvm intrinsics,
    /// add them to librustc_codegen_llvm/context.rs
    fn codegen_intrinsic_call(
        &mut self,
        callee_ty: Ty<'tcx>,
        fn_ty: &FnType<'tcx, Ty<'tcx>>,
        args: &[OperandRef<'tcx, Self::Value>],
        llresult: Self::Value,
        span: Span,
    );
}

pub trait IntrinsicDeclarationMethods<'tcx>: Backend<'tcx> {
    fn get_intrinsic(&self, key: &str) -> Self::Value;

    /// Declare any llvm intrinsics that you might need
    fn declare_intrinsic(&self, key: &str) -> Option<Self::Value>;
}
