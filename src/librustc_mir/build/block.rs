// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use build::{BlockAnd, Builder};
use hair::*;
use rustc::mir::repr::*;
use rustc_front::hir;

impl<'a,'tcx> Builder<'a,'tcx> {
    pub fn ast_block(&mut self,
                     destination: &Lvalue<'tcx>,
                     mut block: BasicBlock,
                     ast_block: &'tcx hir::Block)
                     -> BlockAnd<()> {
        let this = self;
        let Block { extent, span: _, stmts, expr } = this.hir.mirror(ast_block);
        this.in_scope(extent, block, |this| {
            unpack!(block = this.stmts(block, stmts));
            this.into(destination, block, expr)
        })
    }
}
