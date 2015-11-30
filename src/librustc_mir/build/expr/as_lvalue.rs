// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! See docs in build/expr/mod.rs

use build::{BlockAnd, BlockAndExtension, Builder};
use build::expr::category::Category;
use hair::*;
use rustc::mir::repr::*;

impl<'a,'tcx> Builder<'a,'tcx> {
    /// Compile `expr`, yielding an lvalue that we can move from etc.
    pub fn as_lvalue<M>(&mut self,
                        block: BasicBlock,
                        expr: M)
                        -> BlockAnd<Lvalue<'tcx>>
        where M: Mirror<'tcx, Output=Expr<'tcx>>
    {
        let expr = self.hir.mirror(expr);
        self.expr_as_lvalue(block, expr)
    }

    fn expr_as_lvalue(&mut self,
                      mut block: BasicBlock,
                      expr: Expr<'tcx>)
                      -> BlockAnd<Lvalue<'tcx>> {
        debug!("expr_as_lvalue(block={:?}, expr={:?})", block, expr);

        let this = self;
        let expr_span = expr.span;
        match expr.kind {
            ExprKind::Scope { extent, value } => {
                this.in_scope(extent, block, |this| this.as_lvalue(block, value))
            }
            ExprKind::Field { lhs, name } => {
                let lvalue = unpack!(block = this.as_lvalue(block, lhs));
                let lvalue = lvalue.field(name);
                block.and(lvalue)
            }
            ExprKind::Deref { arg } => {
                let lvalue = unpack!(block = this.as_lvalue(block, arg));
                let lvalue = lvalue.deref();
                block.and(lvalue)
            }
            ExprKind::Index { lhs, index } => {
                let (usize_ty, bool_ty) = (this.hir.usize_ty(), this.hir.bool_ty());

                let slice = unpack!(block = this.as_lvalue(block, lhs));

                let idx = unpack!(block = this.as_operand(block, index));

                // bounds check:
                let (len, lt) = (this.temp(usize_ty.clone()), this.temp(bool_ty));
                this.cfg.push_assign(block, expr_span, // len = len(slice)
                                     &len, Rvalue::Len(slice.clone()));
                this.cfg.push_assign(block, expr_span, // lt = idx < len
                                     &lt, Rvalue::BinaryOp(BinOp::Lt,
                                                           idx.clone(),
                                                           Operand::Consume(len)));

                let (success, failure) = (this.cfg.start_new_block(), this.cfg.start_new_block());
                this.cfg.terminate(block,
                                   Terminator::If {
                                       cond: Operand::Consume(lt),
                                       targets: [success, failure],
                                   });
                this.panic(failure);
                success.and(slice.index(idx))
            }
            ExprKind::SelfRef => {
                block.and(Lvalue::Arg(0))
            }
            ExprKind::VarRef { id } => {
                let index = this.var_indices[&id];
                block.and(Lvalue::Var(index))
            }
            ExprKind::StaticRef { id } => {
                block.and(Lvalue::Static(id))
            }

            ExprKind::Vec { .. } |
            ExprKind::Tuple { .. } |
            ExprKind::Adt { .. } |
            ExprKind::Closure { .. } |
            ExprKind::Unary { .. } |
            ExprKind::Binary { .. } |
            ExprKind::LogicalOp { .. } |
            ExprKind::Box { .. } |
            ExprKind::Cast { .. } |
            ExprKind::ReifyFnPointer { .. } |
            ExprKind::UnsafeFnPointer { .. } |
            ExprKind::Unsize { .. } |
            ExprKind::Repeat { .. } |
            ExprKind::Borrow { .. } |
            ExprKind::If { .. } |
            ExprKind::Match { .. } |
            ExprKind::Loop { .. } |
            ExprKind::Block { .. } |
            ExprKind::Assign { .. } |
            ExprKind::AssignOp { .. } |
            ExprKind::Break { .. } |
            ExprKind::Continue { .. } |
            ExprKind::Return { .. } |
            ExprKind::Literal { .. } |
            ExprKind::InlineAsm { .. } |
            ExprKind::Call { .. } => {
                // these are not lvalues, so we need to make a temporary.
                debug_assert!(match Category::of(&expr.kind) {
                    Some(Category::Lvalue) => false,
                    _ => true,
                });
                this.as_temp(block, expr)
            }
        }
    }
}
