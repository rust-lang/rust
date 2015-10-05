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
use repr::*;

impl<'a,'tcx> Builder<'a,'tcx> {
    pub fn stmts(&mut self, mut block: BasicBlock, stmts: Vec<StmtRef<'tcx>>) -> BlockAnd<()> {
        for stmt in stmts {
            unpack!(block = self.stmt(block, stmt));
        }
        block.unit()
    }

    pub fn stmt(&mut self, mut block: BasicBlock, stmt: StmtRef<'tcx>) -> BlockAnd<()> {
        let this = self;
        let Stmt { span, kind } = this.hir.mirror(stmt);
        match kind {
            StmtKind::Let { remainder_scope,
                            init_scope,
                            pattern,
                            initializer: Some(initializer),
                            stmts } => {
                this.in_scope(remainder_scope, block, |this| {
                    unpack!(block = this.in_scope(init_scope, block, |this| {
                        this.expr_into_pattern(block, remainder_scope, pattern, initializer)
                    }));
                    this.stmts(block, stmts)
                })
            }

            StmtKind::Let { remainder_scope, init_scope, pattern, initializer: None, stmts } => {
                this.in_scope(remainder_scope, block, |this| {
                    unpack!(block = this.in_scope(init_scope, block, |this| {
                        this.declare_bindings(remainder_scope, pattern);
                        block.unit()
                    }));
                    this.stmts(block, stmts)
                })
            }

            StmtKind::Expr { scope, expr } => {
                this.in_scope(scope, block, |this| {
                    let expr = this.hir.mirror(expr);
                    let temp = this.temp(expr.ty.clone());
                    unpack!(block = this.into(&temp, block, expr));
                    this.cfg.push_drop(block, span, DropKind::Deep, &temp);
                    block.unit()
                })
            }
        }
    }
}
