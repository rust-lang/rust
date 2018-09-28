// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use rustc::hir::{InlineAsm, GlobalAsm};
use mir::place::PlaceRef;
use super::backend::Backend;
use super::builder::HasCodegen;

pub trait AsmBuilderMethods<'a, 'll: 'a, 'tcx: 'll> : HasCodegen<'a, 'll, 'tcx>{
    // Take an inline assembly expression and splat it out via LLVM
    fn codegen_inline_asm(
        &self,
        ia: &InlineAsm,
        outputs: Vec<PlaceRef<'tcx, <Self::CodegenCx as Backend<'ll>>::Value>>,
        inputs: Vec<<Self::CodegenCx as Backend<'ll>>::Value>
    ) -> bool;
}

pub trait AsmMethods {
    fn codegen_global_asm(&self, ga: &GlobalAsm);
}
