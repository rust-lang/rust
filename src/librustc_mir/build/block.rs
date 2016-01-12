// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use build::{BlockAnd, BlockAndExtension, Builder};
use hair::*;
use rustc::mir::repr::*;
use rustc_front::hir;

impl<'a,'tcx> Builder<'a,'tcx> {
    pub fn ast_block(&mut self,
                     destination: &Lvalue<'tcx>,
                     mut block: BasicBlock,
                     ast_block: &'tcx hir::Block)
                     -> BlockAnd<()> {
        let Block { extent, span, stmts, expr } = self.hir.mirror(ast_block);
        self.in_scope(extent, block, move |this| {
            unpack!(block = this.stmts(block, stmts));
            match expr {
                Some(expr) => this.into(destination, block, expr),
                None => {
                    this.cfg.push_assign_unit(block, span, destination);
                    block.unit()
                }
            }
        })
    }
}
