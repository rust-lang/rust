// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use super::backend::Backend;
use super::builder::HasCodegen;
use mir::operand::OperandRef;
use rustc::ty::Ty;
use abi::FnType;
use syntax_pos::Span;

pub trait IntrinsicCallMethods<'a, 'll: 'a, 'tcx: 'll> : HasCodegen<'a, 'll, 'tcx> {

    /// Remember to add all intrinsics here, in librustc_typeck/check/mod.rs,
    /// and in libcore/intrinsics.rs; if you need access to any llvm intrinsics,
    /// add them to librustc_codegen_llvm/context.rs
    fn codegen_intrinsic_call(
        &self,
        callee_ty: Ty<'tcx>,
        fn_ty: &FnType<'tcx, Ty<'tcx>>,
        args: &[OperandRef<'tcx, <Self::CodegenCx as Backend<'ll>>::Value>],
        llresult: <Self::CodegenCx as Backend<'ll>>::Value,
        span: Span,
    );
}

pub trait IntrinsicDeclarationMethods<'ll> : Backend<'ll> {
    fn get_intrinsic(&self, key: &str) -> Self::Value;

    /// Declare any llvm intrinsics that you might need
    fn declare_intrinsic(
        &self,
        key: &str
    ) -> Option<Self::Value>;
}
