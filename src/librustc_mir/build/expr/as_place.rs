//! See docs in build/expr/mod.rs

use crate::build::expr::category::Category;
use crate::build::ForGuard::{OutsideGuard, RefWithinGuard};
use crate::build::{BlockAnd, BlockAndExtension, Builder};
use crate::hair::*;
use rustc::mir::interpret::InterpError::BoundsCheck;
use rustc::mir::*;
use rustc::ty::{CanonicalUserTypeAnnotation, Variance};

use rustc_data_structures::indexed_vec::Idx;

impl<'a, 'tcx> Builder<'a, 'tcx> {
    /// Compile `expr`, yielding a place that we can move from etc.
    pub fn as_place<M>(&mut self, block: BasicBlock, expr: M) -> BlockAnd<Place<'tcx>>
    where
        M: Mirror<'tcx, Output = Expr<'tcx>>,
    {
        let expr = self.hir.mirror(expr);
        self.expr_as_place(block, expr, Mutability::Mut)
    }

    /// Compile `expr`, yielding a place that we can move from etc.
    /// Mutability note: The caller of this method promises only to read from the resulting
    /// place. The place itself may or may not be mutable:
    /// * If this expr is a place expr like a.b, then we will return that place.
    /// * Otherwise, a temporary is created: in that event, it will be an immutable temporary.
    pub fn as_read_only_place<M>(&mut self, block: BasicBlock, expr: M) -> BlockAnd<Place<'tcx>>
    where
        M: Mirror<'tcx, Output = Expr<'tcx>>,
    {
        let expr = self.hir.mirror(expr);
        self.expr_as_place(block, expr, Mutability::Not)
    }

    fn expr_as_place(
        &mut self,
        mut block: BasicBlock,
        expr: Expr<'tcx>,
        mutability: Mutability,
    ) -> BlockAnd<Place<'tcx>> {
        debug!(
            "expr_as_place(block={:?}, expr={:?}, mutability={:?})",
            block, expr, mutability
        );

        let this = self;
        let expr_span = expr.span;
        let source_info = this.source_info(expr_span);
        match expr.kind {
            ExprKind::Scope {
                region_scope,
                lint_level,
                value,
            } => this.in_scope((region_scope, source_info), lint_level, |this| {
                if mutability == Mutability::Not {
                    this.as_read_only_place(block, value)
                } else {
                    this.as_place(block, value)
                }
            }),
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
                // Making this a *fresh* temporary also means we do not have to worry about
                // the index changing later: Nothing will ever change this temporary.
                // The "retagging" transformation (for Stacked Borrows) relies on this.
                let idx = unpack!(block = this.as_temp(block, None, index, Mutability::Mut));

                // bounds check:
                let (len, lt) = (
                    this.temp(usize_ty.clone(), expr_span),
                    this.temp(bool_ty, expr_span),
                );
                this.cfg.push_assign(
                    block,
                    source_info, // len = len(slice)
                    &len,
                    Rvalue::Len(slice.clone()),
                );
                this.cfg.push_assign(
                    block,
                    source_info, // lt = idx < len
                    &lt,
                    Rvalue::BinaryOp(
                        BinOp::Lt,
                        Operand::Copy(Place::from(idx)),
                        Operand::Copy(len.clone()),
                    ),
                );

                let msg = BoundsCheck {
                    len: Operand::Move(len),
                    index: Operand::Copy(Place::from(idx)),
                };
                let success = this.assert(block, Operand::Move(lt), true, msg, expr_span);
                success.and(slice.index(idx))
            }
            ExprKind::SelfRef => block.and(Place::from(Local::new(1))),
            ExprKind::VarRef { id } => {
                let place = if this.is_bound_var_in_guard(id) {
                    let index = this.var_local_id(id, RefWithinGuard);
                    Place::from(index).deref()
                } else {
                    let index = this.var_local_id(id, OutsideGuard);
                    Place::from(index)
                };
                block.and(place)
            }
            ExprKind::StaticRef { id } => block.and(Place::Base(PlaceBase::Static(Box::new(Static {
                ty: expr.ty,
                kind: StaticKind::Static(id),
            })))),

            ExprKind::PlaceTypeAscription { source, user_ty } => {
                let place = unpack!(block = this.as_place(block, source));
                if let Some(user_ty) = user_ty {
                    let annotation_index = this.canonical_user_type_annotations.push(
                        CanonicalUserTypeAnnotation {
                            span: source_info.span,
                            user_ty,
                            inferred_ty: expr.ty,
                        }
                    );
                    this.cfg.push(
                        block,
                        Statement {
                            source_info,
                            kind: StatementKind::AscribeUserType(
                                place.clone(),
                                Variance::Invariant,
                                box UserTypeProjection { base: annotation_index, projs: vec![], },
                            ),
                        },
                    );
                }
                block.and(place)
            }
            ExprKind::ValueTypeAscription { source, user_ty } => {
                let source = this.hir.mirror(source);
                let temp = unpack!(
                    block = this.as_temp(block, source.temp_lifetime, source, mutability)
                );
                if let Some(user_ty) = user_ty {
                    let annotation_index = this.canonical_user_type_annotations.push(
                        CanonicalUserTypeAnnotation {
                            span: source_info.span,
                            user_ty,
                            inferred_ty: expr.ty,
                        }
                    );
                    this.cfg.push(
                        block,
                        Statement {
                            source_info,
                            kind: StatementKind::AscribeUserType(
                                Place::from(temp.clone()),
                                Variance::Invariant,
                                box UserTypeProjection { base: annotation_index, projs: vec![], },
                            ),
                        },
                    );
                }
                block.and(Place::from(temp))
            }

            ExprKind::Array { .. }
            | ExprKind::Tuple { .. }
            | ExprKind::Adt { .. }
            | ExprKind::Closure { .. }
            | ExprKind::Unary { .. }
            | ExprKind::Binary { .. }
            | ExprKind::LogicalOp { .. }
            | ExprKind::Box { .. }
            | ExprKind::Cast { .. }
            | ExprKind::Use { .. }
            | ExprKind::NeverToAny { .. }
            | ExprKind::Pointer { .. }
            | ExprKind::Repeat { .. }
            | ExprKind::Borrow { .. }
            | ExprKind::Match { .. }
            | ExprKind::Loop { .. }
            | ExprKind::Block { .. }
            | ExprKind::Assign { .. }
            | ExprKind::AssignOp { .. }
            | ExprKind::Break { .. }
            | ExprKind::Continue { .. }
            | ExprKind::Return { .. }
            | ExprKind::Literal { .. }
            | ExprKind::InlineAsm { .. }
            | ExprKind::Yield { .. }
            | ExprKind::Call { .. } => {
                // these are not places, so we need to make a temporary.
                debug_assert!(match Category::of(&expr.kind) {
                    Some(Category::Place) => false,
                    _ => true,
                });
                let temp =
                    unpack!(block = this.as_temp(block, expr.temp_lifetime, expr, mutability));
                block.and(Place::from(temp))
            }
        }
    }
}
