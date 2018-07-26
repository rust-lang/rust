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
use build::ForGuard::{OutsideGuard, RefWithinGuard, ValWithinGuard};
use build::expr::category::Category;
use hair::*;
use rustc::mir::*;
use rustc::mir::interpret::EvalErrorKind::BoundsCheck;

use rustc_data_structures::indexed_vec::Idx;

impl<'a, 'gcx, 'tcx> Builder<'a, 'gcx, 'tcx> {
    /// Compile `expr`, yielding a place that we can move from etc.
    pub fn as_place<M>(&mut self,
                        block: BasicBlock,
                        expr: M)
                        -> BlockAnd<Place<'tcx>>
        where M: Mirror<'tcx, Output=Expr<'tcx>>
    {
        let expr = self.hir.mirror(expr);
        self.expr_as_place(block, expr)
    }

    fn expr_as_place(&mut self,
                      mut block: BasicBlock,
                      expr: Expr<'tcx>)
                      -> BlockAnd<Place<'tcx>> {
        debug!("expr_as_place(block={:?}, expr={:?})", block, expr);

        let this = self;
        let expr_span = expr.span;
        let source_info = this.source_info(expr_span);
        match expr.kind {
            ExprKind::Scope { region_scope, lint_level, value } => {
                this.in_scope((region_scope, source_info), lint_level, block, |this| {
                    this.as_place(block, value)
                })
            }
            ExprKind::Field { lhs, name } => {
                let place = unpack!(block = this.as_place(block, lhs));
                let place = place.field(name, expr.ty);
                block.and(place)
            }
            ExprKind::Deref { arg } => {
                let place = unpack!(block = this.as_place(block, arg));
                let place = place.deref();
                block.and(place)
            }
            ExprKind::Index { lhs, index } => {
                let (usize_ty, bool_ty) = (this.hir.usize_ty(), this.hir.bool_ty());

                let slice = unpack!(block = this.as_place(block, lhs));
                // region_scope=None so place indexes live forever. They are scalars so they
                // do not need storage annotations, and they are often copied between
                // places.
                let idx = unpack!(block = this.as_temp(block, None, index));

                // bounds check:
                let (len, lt) = (this.temp(usize_ty.clone(), expr_span),
                                 this.temp(bool_ty, expr_span));
                this.cfg.push_assign(block, source_info, // len = len(slice)
                                     &len, Rvalue::Len(slice.clone()));
                this.cfg.push_assign(block, source_info, // lt = idx < len
                                     &lt, Rvalue::BinaryOp(BinOp::Lt,
                                                           Operand::Copy(Place::Local(idx)),
                                                           Operand::Copy(len.clone())));

                let msg = BoundsCheck {
                    len: Operand::Move(len),
                    index: Operand::Copy(Place::Local(idx))
                };
                let success = this.assert(block, Operand::Move(lt), true,
                                          msg, expr_span);
                success.and(slice.index(idx))
            }
            ExprKind::SelfRef => {
                block.and(Place::Local(Local::new(1)))
            }
            ExprKind::VarRef { id } => {
                let place = if this.is_bound_var_in_guard(id) {
                    if this.hir.tcx().all_pat_vars_are_implicit_refs_within_guards() {
                        let index = this.var_local_id(id, RefWithinGuard);
                        Place::Local(index).deref()
                    } else {
                        let index = this.var_local_id(id, ValWithinGuard);
                        Place::Local(index)
                    }
                } else {
                    let index = this.var_local_id(id, OutsideGuard);
                    Place::Local(index)
                };
                block.and(place)
            }
            ExprKind::StaticRef { id } => {
                block.and(Place::Static(Box::new(Static { def_id: id, ty: expr.ty })))
            }

            ExprKind::Array { .. } |
            ExprKind::Tuple { .. } |
            ExprKind::Adt { .. } |
            ExprKind::Closure { .. } |
            ExprKind::Unary { .. } |
            ExprKind::Binary { .. } |
            ExprKind::LogicalOp { .. } |
            ExprKind::Box { .. } |
            ExprKind::Cast { .. } |
            ExprKind::Use { .. } |
            ExprKind::NeverToAny { .. } |
            ExprKind::ReifyFnPointer { .. } |
            ExprKind::ClosureFnPointer { .. } |
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
            ExprKind::Yield { .. } |
            ExprKind::Call { .. } => {
                // these are not places, so we need to make a temporary.
                debug_assert!(match Category::of(&expr.kind) {
                    Some(Category::Place) => false,
                    _ => true,
                });
                let temp = unpack!(block = this.as_temp(block, expr.temp_lifetime, expr));
                block.and(Place::Local(temp))
            }
        }
    }
}
