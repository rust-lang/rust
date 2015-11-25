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
        // This convoluted structure is to avoid using recursion as we walk down a list
        // of statements. Basically, the structure we get back is something like:
        //
        //    let x = <init> in {
        //       let y = <init> in {
        //           expr1;
        //           expr2;
        //       }
        //    }
        //
        // To process this, we keep a stack of (Option<CodeExtent>,
        // vec::IntoIter<Stmt>) pairs.  At each point we pull off the
        // top most pair and extract one statement from the
        // iterator. Once it's complete, we pop the scope from the
        // first half the pair.
        let this = self;
        let mut stmt_lists = vec![(None, stmts.into_iter())];
        while !stmt_lists.is_empty() {
            let stmt = {
                let &mut (_, ref mut stmts) = stmt_lists.last_mut().unwrap();
                stmts.next()
            };

            let stmt = match stmt {
                Some(stmt) => stmt,
                None => {
                    let (extent, _) = stmt_lists.pop().unwrap();
                    if let Some(extent) = extent {
                        this.pop_scope(extent, block);
                    }
                    continue
                }
            };

            let Stmt { span, kind } = this.hir.mirror(stmt);
            match kind {
                StmtKind::Let { remainder_scope, init_scope, pattern, initializer, stmts } => {
                    this.push_scope(remainder_scope, block);
                    stmt_lists.push((Some(remainder_scope), stmts.into_iter()));
                    unpack!(block = this.in_scope(init_scope, block, move |this| {
                        // FIXME #30046                              ^~~~
                        match initializer {
                            Some(initializer) => {
                                this.expr_into_pattern(block, remainder_scope, pattern, initializer)
                            }
                            None => {
                                this.declare_bindings(remainder_scope, &pattern);
                                block.unit()
                            }
                        }
                    }));
                }

                StmtKind::Expr { scope, expr } => {
                    unpack!(block = this.in_scope(scope, block, |this| {
                        let expr = this.hir.mirror(expr);
                        let temp = this.temp(expr.ty.clone());
                        unpack!(block = this.into(&temp, block, expr));
                        this.cfg.push_drop(block, span, DropKind::Deep, &temp);
                        block.unit()
                    }));
                }
            }
        }
        block.unit()
    }
}
