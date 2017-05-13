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
use rustc::mir::*;
use rustc::hir;

impl<'a, 'gcx, 'tcx> Builder<'a, 'gcx, 'tcx> {
    pub fn ast_block(&mut self,
                     destination: &Lvalue<'tcx>,
                     // FIXME(#32959): temporary measure for the issue
                     dest_is_unit: bool,
                     mut block: BasicBlock,
                     ast_block: &'tcx hir::Block)
                     -> BlockAnd<()> {
        let Block { extent, span, stmts, expr } = self.hir.mirror(ast_block);
        self.in_scope(extent, block, move |this| {
            // This convoluted structure is to avoid using recursion as we walk down a list
            // of statements. Basically, the structure we get back is something like:
            //
            //    let x = <init> in {
            //       expr1;
            //       let y = <init> in {
            //           expr2;
            //           expr3;
            //           ...
            //       }
            //    }
            //
            // The let bindings are valid till the end of block so all we have to do is to pop all
            // the let-scopes at the end.
            //
            // First we build all the statements in the block.
            let mut let_extent_stack = Vec::with_capacity(8);
            let outer_visibility_scope = this.visibility_scope;
            for stmt in stmts {
                let Stmt { span: _, kind } = this.hir.mirror(stmt);
                match kind {
                    StmtKind::Expr { scope, expr } => {
                        unpack!(block = this.in_scope(scope, block, |this| {
                            let expr = this.hir.mirror(expr);
                            this.stmt_expr(block, expr)
                        }));
                    }
                    StmtKind::Let { remainder_scope, init_scope, pattern, initializer } => {
                        let tcx = this.hir.tcx();

                        // Enter the remainder scope, i.e. the bindings' destruction scope.
                        this.push_scope(remainder_scope);
                        let_extent_stack.push(remainder_scope);

                        // Declare the bindings, which may create a visibility scope.
                        let remainder_span = remainder_scope.span(&tcx.region_maps, &tcx.map);
                        let remainder_span = remainder_span.unwrap_or(span);
                        let scope = this.declare_bindings(None, remainder_span, &pattern);

                        // Evaluate the initializer, if present.
                        if let Some(init) = initializer {
                            unpack!(block = this.in_scope(init_scope, block, move |this| {
                                // FIXME #30046                              ^~~~
                                this.expr_into_pattern(block, pattern, init)
                            }));
                        } else {
                            this.storage_live_for_bindings(block, &pattern);
                        }

                        // Enter the visibility scope, after evaluating the initializer.
                        if let Some(visibility_scope) = scope {
                            this.visibility_scope = visibility_scope;
                        }
                    }
                }
            }
            // Then, the block may have an optional trailing expression which is a “return” value
            // of the block.
            if let Some(expr) = expr {
                unpack!(block = this.into(destination, block, expr));
            } else if dest_is_unit {
                // FIXME(#31472)
                let source_info = this.source_info(span);
                this.cfg.push_assign_unit(block, source_info, destination);
            }
            // Finally, we pop all the let scopes before exiting out from the scope of block
            // itself.
            for extent in let_extent_stack.into_iter().rev() {
                unpack!(block = this.pop_scope(extent, block));
            }
            // Restore the original visibility scope.
            this.visibility_scope = outer_visibility_scope;
            block.unit()
        })
    }
}
