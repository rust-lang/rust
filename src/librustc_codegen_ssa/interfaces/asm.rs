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
use mir::place::PlaceRef;
use rustc::hir::{GlobalAsm, InlineAsm};

pub trait AsmBuilderMethods<'tcx>: HasCodegen<'tcx> {
    // Take an inline assembly expression and splat it out via LLVM
    fn codegen_inline_asm(
        &self,
        ia: &InlineAsm,
        outputs: Vec<PlaceRef<'tcx, Self::Value>>,
        inputs: Vec<Self::Value>,
    ) -> bool;
}

pub trait AsmMethods<'tcx>: Backend<'tcx> {
    fn codegen_global_asm(&self, ga: &GlobalAsm);
}
