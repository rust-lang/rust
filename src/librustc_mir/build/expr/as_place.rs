//! See docs in build/expr/mod.rs

use crate::build::expr::category::Category;
use crate::build::ForGuard::{OutsideGuard, RefWithinGuard};
use crate::build::{BlockAnd, BlockAndExtension, Builder};
use crate::hair::*;
use rustc::mir::interpret::{PanicInfo::BoundsCheck};
use rustc::mir::*;
use rustc::ty::{CanonicalUserTypeAnnotation, Ty, TyCtxt, Variance};

use rustc_index::vec::Idx;

/// `PlaceBuilder` is used to create places during MIR construction. It allows you to "build up" a
/// place by pushing more and more projections onto the end, and then convert the final set into a
/// place using the `into_place` method.
///
/// This is used internally when building a place for an expression like `a.b.c`. The fields `b`
/// and `c` can be progressively pushed onto the place builder that is created when converting `a`.
#[derive(Clone)]
struct PlaceBuilder<'tcx> {
    base: PlaceBase<'tcx>,
    projection: Vec<PlaceElem<'tcx>>,
}

impl PlaceBuilder<'tcx> {
    fn into_place(self, tcx: TyCtxt<'tcx>) -> Place<'tcx> {
        Place {
            base: self.base,
            projection: tcx.intern_place_elems(&self.projection),
        }
    }

    fn field(self, f: Field, ty: Ty<'tcx>) -> Self {
        self.project(PlaceElem::Field(f, ty))
    }

    fn deref(self) -> Self {
        self.project(PlaceElem::Deref)
    }

    fn index(self, index: Local) -> Self {
        self.project(PlaceElem::Index(index))
    }

    fn project(mut self, elem: PlaceElem<'tcx>) -> Self {
        self.projection.push(elem);
        self
    }
}

impl From<Local> for PlaceBuilder<'tcx> {
    fn from(local: Local) -> Self {
        Self {
            base: local.into(),
            projection: Vec::new(),
        }
    }
}

impl From<PlaceBase<'tcx>> for PlaceBuilder<'tcx> {
    fn from(base: PlaceBase<'tcx>) -> Self {
        Self {
            base,
            projection: Vec::new(),
        }
    }
}

impl<'a, 'tcx> Builder<'a, 'tcx> {
    /// Compile `expr`, yielding a place that we can move from etc.
    pub fn as_place<M>(&mut self, mut block: BasicBlock, expr: M) -> BlockAnd<Place<'tcx>>
    where
        M: Mirror<'tcx, Output = Expr<'tcx>>,
    {
        let place_builder = unpack!(block = self.as_place_builder(block, expr));
        block.and(place_builder.into_place(self.hir.tcx()))
    }

    /// This is used when constructing a compound `Place`, so that we can avoid creating
    /// intermediate `Place` values until we know the full set of projections.
    fn as_place_builder<M>(&mut self, block: BasicBlock, expr: M) -> BlockAnd<PlaceBuilder<'tcx>>
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
    pub fn as_read_only_place<M>(&mut self, mut block: BasicBlock, expr: M) -> BlockAnd<Place<'tcx>>
    where
        M: Mirror<'tcx, Output = Expr<'tcx>>,
    {
        let place_builder = unpack!(block = self.as_read_only_place_builder(block, expr));
        block.and(place_builder.into_place(self.hir.tcx()))
    }

    /// This is used when constructing a compound `Place`, so that we can avoid creating
    /// intermediate `Place` values until we know the full set of projections.
    /// Mutability note: The caller of this method promises only to read from the resulting
    /// place. The place itself may or may not be mutable:
    /// * If this expr is a place expr like a.b, then we will return that place.
    /// * Otherwise, a temporary is created: in that event, it will be an immutable temporary.
    fn as_read_only_place_builder<M>(
        &mut self,
        block: BasicBlock,
        expr: M,
    ) -> BlockAnd<PlaceBuilder<'tcx>>
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
    ) -> BlockAnd<PlaceBuilder<'tcx>> {
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
                    this.as_read_only_place_builder(block, value)
                } else {
                    this.as_place_builder(block, value)
                }
            }),
            ExprKind::Field { lhs, name } => {
                let place_builder = unpack!(block = this.as_place_builder(block, lhs));
                block.and(place_builder.field(name, expr.ty))
            }
            ExprKind::Deref { arg } => {
                let place_builder = unpack!(block = this.as_place_builder(block, arg));
                block.and(place_builder.deref())
            }
            ExprKind::Index { lhs, index } => {
                let (usize_ty, bool_ty) = (this.hir.usize_ty(), this.hir.bool_ty());

                let place_builder = unpack!(block = this.as_place_builder(block, lhs));
                // Making this a *fresh* temporary also means we do not have to worry about
                // the index changing later: Nothing will ever change this temporary.
                // The "retagging" transformation (for Stacked Borrows) relies on this.
                let idx = unpack!(block = this.as_temp(
                    block,
                    expr.temp_lifetime,
                    index,
                    Mutability::Not,
                ));

                let slice = place_builder.clone().into_place(this.hir.tcx());
                // bounds check:
                let (len, lt) = (
                    this.temp(usize_ty.clone(), expr_span),
                    this.temp(bool_ty, expr_span),
                );
                this.cfg.push_assign(
                    block,
                    source_info, // len = len(slice)
                    &len,
                    Rvalue::Len(slice),
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
                success.and(place_builder.index(idx))
            }
            ExprKind::SelfRef => block.and(PlaceBuilder::from(Local::new(1))),
            ExprKind::VarRef { id } => {
                let place_builder = if this.is_bound_var_in_guard(id) {
                    let index = this.var_local_id(id, RefWithinGuard);
                    PlaceBuilder::from(index).deref()
                } else {
                    let index = this.var_local_id(id, OutsideGuard);
                    PlaceBuilder::from(index)
                };
                block.and(place_builder)
            }
            ExprKind::StaticRef { id } => block.and(PlaceBuilder::from(
                PlaceBase::Static(Box::new(Static {
                    ty: expr.ty,
                    kind: StaticKind::Static,
                    def_id: id,
                }))
            )),

            ExprKind::PlaceTypeAscription { source, user_ty } => {
                let place_builder = unpack!(block = this.as_place_builder(block, source));
                if let Some(user_ty) = user_ty {
                    let annotation_index = this.canonical_user_type_annotations.push(
                        CanonicalUserTypeAnnotation {
                            span: source_info.span,
                            user_ty,
                            inferred_ty: expr.ty,
                        }
                    );

                    let place = place_builder.clone().into_place(this.hir.tcx());
                    this.cfg.push(
                        block,
                        Statement {
                            source_info,
                            kind: StatementKind::AscribeUserType(
                                box(
                                    place,
                                    UserTypeProjection { base: annotation_index, projs: vec![], }
                                ),
                                Variance::Invariant,
                            ),
                        },
                    );
                }
                block.and(place_builder)
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
                                box(
                                    Place::from(temp.clone()),
                                    UserTypeProjection { base: annotation_index, projs: vec![], },
                                ),
                                Variance::Invariant,
                            ),
                        },
                    );
                }
                block.and(PlaceBuilder::from(temp))
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
                block.and(PlaceBuilder::from(temp))
            }
        }
    }
}
