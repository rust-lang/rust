//! A different sort of visitor for walking fn bodies. Unlike the
//! normal visitor, which just walks the entire body in one shot, the
//! `ExprUseVisitor` determines how expressions are being used.
//!
//! In the compiler, this is only used for upvar inference, but there
//! are many uses within clippy.

use std::cell::{Ref, RefCell};
use std::ops::Deref;
use std::slice::from_ref;

use hir::Expr;
use hir::def::DefKind;
use hir::pat_util::EnumerateAndAdjustIterator as _;
use rustc_abi::{FIRST_VARIANT, FieldIdx, VariantIdx};
use rustc_data_structures::fx::FxIndexMap;
use rustc_hir::def::{CtorOf, Res};
use rustc_hir::def_id::LocalDefId;
use rustc_hir::{self as hir, HirId, PatExpr, PatExprKind, PatKind};
use rustc_lint::LateContext;
use rustc_middle::hir::place::ProjectionKind;
// Export these here so that Clippy can use them.
pub use rustc_middle::hir::place::{Place, PlaceBase, PlaceWithHirId, Projection};
use rustc_middle::mir::FakeReadCause;
use rustc_middle::ty::{
    self, BorrowKind, Ty, TyCtxt, TypeFoldable, TypeVisitableExt as _, adjustment,
};
use rustc_middle::{bug, span_bug};
use rustc_span::{ErrorGuaranteed, Span};
use rustc_trait_selection::infer::InferCtxtExt;
use tracing::{debug, instrument, trace};

use crate::fn_ctxt::FnCtxt;

/// This trait defines the callbacks you can expect to receive when
/// employing the ExprUseVisitor.
pub trait Delegate<'tcx> {
    /// The value found at `place` is moved, depending
    /// on `mode`. Where `diag_expr_id` is the id used for diagnostics for `place`.
    ///
    /// If the value is `Copy`, [`copy`][Self::copy] is called instead, which
    /// by default falls back to [`borrow`][Self::borrow].
    ///
    /// The parameter `diag_expr_id` indicates the HIR id that ought to be used for
    /// diagnostics. Around pattern matching such as `let pat = expr`, the diagnostic
    /// id will be the id of the expression `expr` but the place itself will have
    /// the id of the binding in the pattern `pat`.
    fn consume(&mut self, place_with_id: &PlaceWithHirId<'tcx>, diag_expr_id: HirId);

    /// The value found at `place` is used, depending
    /// on `mode`. Where `diag_expr_id` is the id used for diagnostics for `place`.
    ///
    /// Use of a `Copy` type in a ByUse context is considered a use
    /// by `ImmBorrow` and `borrow` is called instead. This is because
    /// a shared borrow is the "minimum access" that would be needed
    /// to perform a copy.
    ///
    ///
    /// The parameter `diag_expr_id` indicates the HIR id that ought to be used for
    /// diagnostics. Around pattern matching such as `let pat = expr`, the diagnostic
    /// id will be the id of the expression `expr` but the place itself will have
    /// the id of the binding in the pattern `pat`.
    fn use_cloned(&mut self, place_with_id: &PlaceWithHirId<'tcx>, diag_expr_id: HirId);

    /// The value found at `place` is being borrowed with kind `bk`.
    /// `diag_expr_id` is the id used for diagnostics (see `consume` for more details).
    fn borrow(
        &mut self,
        place_with_id: &PlaceWithHirId<'tcx>,
        diag_expr_id: HirId,
        bk: ty::BorrowKind,
    );

    /// The value found at `place` is being copied.
    /// `diag_expr_id` is the id used for diagnostics (see `consume` for more details).
    ///
    /// If an implementation is not provided, use of a `Copy` type in a ByValue context is instead
    /// considered a use by `ImmBorrow` and `borrow` is called instead. This is because a shared
    /// borrow is the "minimum access" that would be needed to perform a copy.
    fn copy(&mut self, place_with_id: &PlaceWithHirId<'tcx>, diag_expr_id: HirId) {
        // In most cases, copying data from `x` is equivalent to doing `*&x`, so by default
        // we treat a copy of `x` as a borrow of `x`.
        self.borrow(place_with_id, diag_expr_id, ty::BorrowKind::Immutable)
    }

    /// The path at `assignee_place` is being assigned to.
    /// `diag_expr_id` is the id used for diagnostics (see `consume` for more details).
    fn mutate(&mut self, assignee_place: &PlaceWithHirId<'tcx>, diag_expr_id: HirId);

    /// The path at `binding_place` is a binding that is being initialized.
    ///
    /// This covers cases such as `let x = 42;`
    fn bind(&mut self, binding_place: &PlaceWithHirId<'tcx>, diag_expr_id: HirId) {
        // Bindings can normally be treated as a regular assignment, so by default we
        // forward this to the mutate callback.
        self.mutate(binding_place, diag_expr_id)
    }

    /// The `place` should be a fake read because of specified `cause`.
    fn fake_read(
        &mut self,
        place_with_id: &PlaceWithHirId<'tcx>,
        cause: FakeReadCause,
        diag_expr_id: HirId,
    );
}

impl<'tcx, D: Delegate<'tcx>> Delegate<'tcx> for &mut D {
    fn consume(&mut self, place_with_id: &PlaceWithHirId<'tcx>, diag_expr_id: HirId) {
        (**self).consume(place_with_id, diag_expr_id)
    }

    fn use_cloned(&mut self, place_with_id: &PlaceWithHirId<'tcx>, diag_expr_id: HirId) {
        (**self).use_cloned(place_with_id, diag_expr_id)
    }

    fn borrow(
        &mut self,
        place_with_id: &PlaceWithHirId<'tcx>,
        diag_expr_id: HirId,
        bk: ty::BorrowKind,
    ) {
        (**self).borrow(place_with_id, diag_expr_id, bk)
    }

    fn copy(&mut self, place_with_id: &PlaceWithHirId<'tcx>, diag_expr_id: HirId) {
        (**self).copy(place_with_id, diag_expr_id)
    }

    fn mutate(&mut self, assignee_place: &PlaceWithHirId<'tcx>, diag_expr_id: HirId) {
        (**self).mutate(assignee_place, diag_expr_id)
    }

    fn bind(&mut self, binding_place: &PlaceWithHirId<'tcx>, diag_expr_id: HirId) {
        (**self).bind(binding_place, diag_expr_id)
    }

    fn fake_read(
        &mut self,
        place_with_id: &PlaceWithHirId<'tcx>,
        cause: FakeReadCause,
        diag_expr_id: HirId,
    ) {
        (**self).fake_read(place_with_id, cause, diag_expr_id)
    }
}

/// This trait makes `ExprUseVisitor` usable with both [`FnCtxt`]
/// and [`LateContext`], depending on where in the compiler it is used.
pub trait TypeInformationCtxt<'tcx> {
    type TypeckResults<'a>: Deref<Target = ty::TypeckResults<'tcx>>
    where
        Self: 'a;

    type Error;

    fn typeck_results(&self) -> Self::TypeckResults<'_>;

    fn resolve_vars_if_possible<T: TypeFoldable<TyCtxt<'tcx>>>(&self, t: T) -> T;

    fn try_structurally_resolve_type(&self, span: Span, ty: Ty<'tcx>) -> Ty<'tcx>;

    fn report_bug(&self, span: Span, msg: impl ToString) -> Self::Error;

    fn error_reported_in_ty(&self, ty: Ty<'tcx>) -> Result<(), Self::Error>;

    fn tainted_by_errors(&self) -> Result<(), Self::Error>;

    fn type_is_copy_modulo_regions(&self, ty: Ty<'tcx>) -> bool;

    fn type_is_use_cloned_modulo_regions(&self, ty: Ty<'tcx>) -> bool;

    fn body_owner_def_id(&self) -> LocalDefId;

    fn tcx(&self) -> TyCtxt<'tcx>;
}

impl<'tcx> TypeInformationCtxt<'tcx> for &FnCtxt<'_, 'tcx> {
    type TypeckResults<'a>
        = Ref<'a, ty::TypeckResults<'tcx>>
    where
        Self: 'a;

    type Error = ErrorGuaranteed;

    fn typeck_results(&self) -> Self::TypeckResults<'_> {
        self.typeck_results.borrow()
    }

    fn resolve_vars_if_possible<T: TypeFoldable<TyCtxt<'tcx>>>(&self, t: T) -> T {
        self.infcx.resolve_vars_if_possible(t)
    }

    fn try_structurally_resolve_type(&self, sp: Span, ty: Ty<'tcx>) -> Ty<'tcx> {
        (**self).try_structurally_resolve_type(sp, ty)
    }

    fn report_bug(&self, span: Span, msg: impl ToString) -> Self::Error {
        self.dcx().span_delayed_bug(span, msg.to_string())
    }

    fn error_reported_in_ty(&self, ty: Ty<'tcx>) -> Result<(), Self::Error> {
        ty.error_reported()
    }

    fn tainted_by_errors(&self) -> Result<(), ErrorGuaranteed> {
        if let Some(guar) = self.infcx.tainted_by_errors() { Err(guar) } else { Ok(()) }
    }

    fn type_is_copy_modulo_regions(&self, ty: Ty<'tcx>) -> bool {
        self.infcx.type_is_copy_modulo_regions(self.param_env, ty)
    }

    fn type_is_use_cloned_modulo_regions(&self, ty: Ty<'tcx>) -> bool {
        self.infcx.type_is_use_cloned_modulo_regions(self.param_env, ty)
    }

    fn body_owner_def_id(&self) -> LocalDefId {
        self.body_id
    }

    fn tcx(&self) -> TyCtxt<'tcx> {
        self.tcx
    }
}

impl<'tcx> TypeInformationCtxt<'tcx> for (&LateContext<'tcx>, LocalDefId) {
    type TypeckResults<'a>
        = &'tcx ty::TypeckResults<'tcx>
    where
        Self: 'a;

    type Error = !;

    fn typeck_results(&self) -> Self::TypeckResults<'_> {
        self.0.maybe_typeck_results().expect("expected typeck results")
    }

    fn try_structurally_resolve_type(&self, _span: Span, ty: Ty<'tcx>) -> Ty<'tcx> {
        // FIXME: Maybe need to normalize here.
        ty
    }

    fn resolve_vars_if_possible<T: TypeFoldable<TyCtxt<'tcx>>>(&self, t: T) -> T {
        t
    }

    fn report_bug(&self, span: Span, msg: impl ToString) -> ! {
        span_bug!(span, "{}", msg.to_string())
    }

    fn error_reported_in_ty(&self, _ty: Ty<'tcx>) -> Result<(), !> {
        Ok(())
    }

    fn tainted_by_errors(&self) -> Result<(), !> {
        Ok(())
    }

    fn type_is_copy_modulo_regions(&self, ty: Ty<'tcx>) -> bool {
        self.0.type_is_copy_modulo_regions(ty)
    }

    fn type_is_use_cloned_modulo_regions(&self, ty: Ty<'tcx>) -> bool {
        self.0.type_is_use_cloned_modulo_regions(ty)
    }

    fn body_owner_def_id(&self) -> LocalDefId {
        self.1
    }

    fn tcx(&self) -> TyCtxt<'tcx> {
        self.0.tcx
    }
}

/// A visitor that reports how each expression is being used.
///
/// See [module-level docs][self] and [`Delegate`] for details.
pub struct ExprUseVisitor<'tcx, Cx: TypeInformationCtxt<'tcx>, D: Delegate<'tcx>> {
    cx: Cx,
    /// We use a `RefCell` here so that delegates can mutate themselves, but we can
    /// still have calls to our own helper functions.
    delegate: RefCell<D>,
    upvars: Option<&'tcx FxIndexMap<HirId, hir::Upvar>>,
}

impl<'a, 'tcx, D: Delegate<'tcx>> ExprUseVisitor<'tcx, (&'a LateContext<'tcx>, LocalDefId), D> {
    pub fn for_clippy(cx: &'a LateContext<'tcx>, body_def_id: LocalDefId, delegate: D) -> Self {
        Self::new((cx, body_def_id), delegate)
    }
}

impl<'tcx, Cx: TypeInformationCtxt<'tcx>, D: Delegate<'tcx>> ExprUseVisitor<'tcx, Cx, D> {
    /// Creates the ExprUseVisitor, configuring it with the various options provided:
    ///
    /// - `delegate` -- who receives the callbacks
    /// - `param_env` --- parameter environment for trait lookups (esp. pertaining to `Copy`)
    /// - `typeck_results` --- typeck results for the code being analyzed
    pub(crate) fn new(cx: Cx, delegate: D) -> Self {
        ExprUseVisitor {
            delegate: RefCell::new(delegate),
            upvars: cx.tcx().upvars_mentioned(cx.body_owner_def_id()),
            cx,
        }
    }

    pub fn consume_body(&self, body: &hir::Body<'_>) -> Result<(), Cx::Error> {
        for param in body.params {
            let param_ty = self.pat_ty_adjusted(param.pat)?;
            debug!("consume_body: param_ty = {:?}", param_ty);

            let param_place = self.cat_rvalue(param.hir_id, param_ty);

            self.walk_irrefutable_pat(&param_place, param.pat)?;
        }

        self.consume_expr(body.value)?;

        Ok(())
    }

    #[instrument(skip(self), level = "debug")]
    fn consume_or_copy(&self, place_with_id: &PlaceWithHirId<'tcx>, diag_expr_id: HirId) {
        if self.cx.type_is_copy_modulo_regions(place_with_id.place.ty()) {
            self.delegate.borrow_mut().copy(place_with_id, diag_expr_id);
        } else {
            self.delegate.borrow_mut().consume(place_with_id, diag_expr_id);
        }
    }

    #[instrument(skip(self), level = "debug")]
    pub fn consume_clone_or_copy(&self, place_with_id: &PlaceWithHirId<'tcx>, diag_expr_id: HirId) {
        // `x.use` will do one of the following
        // * if it implements `Copy`, it will be a copy
        // * if it implements `UseCloned`, it will be a call to `clone`
        // * otherwise, it is a move
        //
        // we do a conservative approximation of this, treating it as a move unless we know that it implements copy or `UseCloned`
        if self.cx.type_is_copy_modulo_regions(place_with_id.place.ty()) {
            self.delegate.borrow_mut().copy(place_with_id, diag_expr_id);
        } else if self.cx.type_is_use_cloned_modulo_regions(place_with_id.place.ty()) {
            self.delegate.borrow_mut().use_cloned(place_with_id, diag_expr_id);
        } else {
            self.delegate.borrow_mut().consume(place_with_id, diag_expr_id);
        }
    }

    fn consume_exprs(&self, exprs: &[hir::Expr<'_>]) -> Result<(), Cx::Error> {
        for expr in exprs {
            self.consume_expr(expr)?;
        }

        Ok(())
    }

    // FIXME: It's suspicious that this is public; clippy should probably use `walk_expr`.
    #[instrument(skip(self), level = "debug")]
    pub fn consume_expr(&self, expr: &hir::Expr<'_>) -> Result<(), Cx::Error> {
        let place_with_id = self.cat_expr(expr)?;
        self.consume_or_copy(&place_with_id, place_with_id.hir_id);
        self.walk_expr(expr)?;
        Ok(())
    }

    #[instrument(skip(self), level = "debug")]
    pub fn consume_or_clone_expr(&self, expr: &hir::Expr<'_>) -> Result<(), Cx::Error> {
        let place_with_id = self.cat_expr(expr)?;
        self.consume_clone_or_copy(&place_with_id, place_with_id.hir_id);
        self.walk_expr(expr)?;
        Ok(())
    }

    fn mutate_expr(&self, expr: &hir::Expr<'_>) -> Result<(), Cx::Error> {
        let place_with_id = self.cat_expr(expr)?;
        self.delegate.borrow_mut().mutate(&place_with_id, place_with_id.hir_id);
        self.walk_expr(expr)?;
        Ok(())
    }

    #[instrument(skip(self), level = "debug")]
    fn borrow_expr(&self, expr: &hir::Expr<'_>, bk: ty::BorrowKind) -> Result<(), Cx::Error> {
        let place_with_id = self.cat_expr(expr)?;
        self.delegate.borrow_mut().borrow(&place_with_id, place_with_id.hir_id, bk);
        self.walk_expr(expr)
    }

    #[instrument(skip(self), level = "debug")]
    pub fn walk_expr(&self, expr: &hir::Expr<'_>) -> Result<(), Cx::Error> {
        self.walk_adjustment(expr)?;

        match expr.kind {
            hir::ExprKind::Path(_) => {}

            hir::ExprKind::Type(subexpr, _) => {
                self.walk_expr(subexpr)?;
            }

            hir::ExprKind::UnsafeBinderCast(_, subexpr, _) => {
                self.walk_expr(subexpr)?;
            }

            hir::ExprKind::Unary(hir::UnOp::Deref, base) => {
                // *base
                self.walk_expr(base)?;
            }

            hir::ExprKind::Field(base, _) => {
                // base.f
                self.walk_expr(base)?;
            }

            hir::ExprKind::Index(lhs, rhs, _) => {
                // lhs[rhs]
                self.walk_expr(lhs)?;
                self.consume_expr(rhs)?;
            }

            hir::ExprKind::Call(callee, args) => {
                // callee(args)
                self.consume_expr(callee)?;
                self.consume_exprs(args)?;
            }

            hir::ExprKind::Use(expr, _) => {
                self.consume_or_clone_expr(expr)?;
            }

            hir::ExprKind::MethodCall(.., receiver, args, _) => {
                // callee.m(args)
                self.consume_expr(receiver)?;
                self.consume_exprs(args)?;
            }

            hir::ExprKind::Struct(_, fields, ref opt_with) => {
                self.walk_struct_expr(fields, opt_with)?;
            }

            hir::ExprKind::Tup(exprs) => {
                self.consume_exprs(exprs)?;
            }

            hir::ExprKind::If(cond_expr, then_expr, ref opt_else_expr) => {
                self.consume_expr(cond_expr)?;
                self.consume_expr(then_expr)?;
                if let Some(else_expr) = *opt_else_expr {
                    self.consume_expr(else_expr)?;
                }
            }

            hir::ExprKind::Let(hir::LetExpr { pat, init, .. }) => {
                self.walk_local(init, pat, None, || self.borrow_expr(init, BorrowKind::Immutable))?;
            }

            hir::ExprKind::Match(discr, arms, _) => {
                let discr_place = self.cat_expr(discr)?;
                self.maybe_read_scrutinee(
                    discr,
                    discr_place.clone(),
                    arms.iter().map(|arm| arm.pat),
                )?;

                // treatment of the discriminant is handled while walking the arms.
                for arm in arms {
                    self.walk_arm(&discr_place, arm)?;
                }
            }

            hir::ExprKind::Array(exprs) => {
                self.consume_exprs(exprs)?;
            }

            hir::ExprKind::AddrOf(_, m, base) => {
                // &base
                // make sure that the thing we are pointing out stays valid
                // for the lifetime `scope_r` of the resulting ptr:
                let bk = ty::BorrowKind::from_mutbl(m);
                self.borrow_expr(base, bk)?;
            }

            hir::ExprKind::InlineAsm(asm) => {
                for (op, _op_sp) in asm.operands {
                    match op {
                        hir::InlineAsmOperand::In { expr, .. } => {
                            self.consume_expr(expr)?;
                        }
                        hir::InlineAsmOperand::Out { expr: Some(expr), .. }
                        | hir::InlineAsmOperand::InOut { expr, .. } => {
                            self.mutate_expr(expr)?;
                        }
                        hir::InlineAsmOperand::SplitInOut { in_expr, out_expr, .. } => {
                            self.consume_expr(in_expr)?;
                            if let Some(out_expr) = out_expr {
                                self.mutate_expr(out_expr)?;
                            }
                        }
                        hir::InlineAsmOperand::Out { expr: None, .. }
                        | hir::InlineAsmOperand::Const { .. }
                        | hir::InlineAsmOperand::SymFn { .. }
                        | hir::InlineAsmOperand::SymStatic { .. } => {}
                        hir::InlineAsmOperand::Label { block } => {
                            self.walk_block(block)?;
                        }
                    }
                }
            }

            hir::ExprKind::Continue(..)
            | hir::ExprKind::Lit(..)
            | hir::ExprKind::ConstBlock(..)
            | hir::ExprKind::OffsetOf(..)
            | hir::ExprKind::Err(_) => {}

            hir::ExprKind::Loop(blk, ..) => {
                self.walk_block(blk)?;
            }

            hir::ExprKind::Unary(_, lhs) => {
                self.consume_expr(lhs)?;
            }

            hir::ExprKind::Binary(_, lhs, rhs) => {
                self.consume_expr(lhs)?;
                self.consume_expr(rhs)?;
            }

            hir::ExprKind::Block(blk, _) => {
                self.walk_block(blk)?;
            }

            hir::ExprKind::Break(_, ref opt_expr) | hir::ExprKind::Ret(ref opt_expr) => {
                if let Some(expr) = *opt_expr {
                    self.consume_expr(expr)?;
                }
            }

            hir::ExprKind::Become(call) => {
                self.consume_expr(call)?;
            }

            hir::ExprKind::Assign(lhs, rhs, _) => {
                self.mutate_expr(lhs)?;
                self.consume_expr(rhs)?;
            }

            hir::ExprKind::Cast(base, _) => {
                self.consume_expr(base)?;
            }

            hir::ExprKind::DropTemps(expr) => {
                self.consume_expr(expr)?;
            }

            hir::ExprKind::AssignOp(_, lhs, rhs) => {
                if self.cx.typeck_results().is_method_call(expr) {
                    self.consume_expr(lhs)?;
                } else {
                    self.mutate_expr(lhs)?;
                }
                self.consume_expr(rhs)?;
            }

            hir::ExprKind::Repeat(base, _) => {
                self.consume_expr(base)?;
            }

            hir::ExprKind::Closure(closure) => {
                self.walk_captures(closure)?;
            }

            hir::ExprKind::Yield(value, _) => {
                self.consume_expr(value)?;
            }
        }

        Ok(())
    }

    fn walk_stmt(&self, stmt: &hir::Stmt<'_>) -> Result<(), Cx::Error> {
        match stmt.kind {
            hir::StmtKind::Let(hir::LetStmt { pat, init: Some(expr), els, .. }) => {
                self.walk_local(expr, pat, *els, || Ok(()))?;
            }

            hir::StmtKind::Let(_) => {}

            hir::StmtKind::Item(_) => {
                // We don't visit nested items in this visitor,
                // only the fn body we were given.
            }

            hir::StmtKind::Expr(expr) | hir::StmtKind::Semi(expr) => {
                self.consume_expr(expr)?;
            }
        }

        Ok(())
    }

    fn maybe_read_scrutinee<'t>(
        &self,
        discr: &Expr<'_>,
        discr_place: PlaceWithHirId<'tcx>,
        pats: impl Iterator<Item = &'t hir::Pat<'t>>,
    ) -> Result<(), Cx::Error> {
        // Matching should not always be considered a use of the place, hence
        // discr does not necessarily need to be borrowed.
        // We only want to borrow discr if the pattern contain something other
        // than wildcards.
        let mut needs_to_be_read = false;
        for pat in pats {
            self.cat_pattern(discr_place.clone(), pat, &mut |place, pat| {
                match &pat.kind {
                    PatKind::Missing => unreachable!(),
                    PatKind::Binding(.., opt_sub_pat) => {
                        // If the opt_sub_pat is None, then the binding does not count as
                        // a wildcard for the purpose of borrowing discr.
                        if opt_sub_pat.is_none() {
                            needs_to_be_read = true;
                        }
                    }
                    PatKind::Never => {
                        // A never pattern reads the value.
                        // FIXME(never_patterns): does this do what I expect?
                        needs_to_be_read = true;
                    }
                    PatKind::Expr(PatExpr { kind: PatExprKind::Path(qpath), hir_id, span }) => {
                        // A `Path` pattern is just a name like `Foo`. This is either a
                        // named constant or else it refers to an ADT variant

                        let res = self.cx.typeck_results().qpath_res(qpath, *hir_id);
                        match res {
                            Res::Def(DefKind::Const, _) | Res::Def(DefKind::AssocConst, _) => {
                                // Named constants have to be equated with the value
                                // being matched, so that's a read of the value being matched.
                                //
                                // FIXME: We don't actually reads for ZSTs.
                                needs_to_be_read = true;
                            }
                            _ => {
                                // Otherwise, this is a struct/enum variant, and so it's
                                // only a read if we need to read the discriminant.
                                needs_to_be_read |=
                                    self.is_multivariant_adt(place.place.ty(), *span);
                            }
                        }
                    }
                    PatKind::TupleStruct(..) | PatKind::Struct(..) | PatKind::Tuple(..) => {
                        // For `Foo(..)`, `Foo { ... }` and `(...)` patterns, check if we are matching
                        // against a multivariant enum or struct. In that case, we have to read
                        // the discriminant. Otherwise this kind of pattern doesn't actually
                        // read anything (we'll get invoked for the `...`, which may indeed
                        // perform some reads).

                        let place_ty = place.place.ty();
                        needs_to_be_read |= self.is_multivariant_adt(place_ty, pat.span);
                    }
                    PatKind::Expr(_) | PatKind::Range(..) => {
                        // If the PatKind is a Lit or a Range then we want
                        // to borrow discr.
                        needs_to_be_read = true;
                    }
                    PatKind::Slice(lhs, wild, rhs) => {
                        // We don't need to test the length if the pattern is `[..]`
                        if matches!((lhs, wild, rhs), (&[], Some(_), &[]))
                            // Arrays have a statically known size, so
                            // there is no need to read their length
                            || place.place.ty().peel_refs().is_array()
                        {
                        } else {
                            needs_to_be_read = true;
                        }
                    }
                    PatKind::Or(_)
                    | PatKind::Box(_)
                    | PatKind::Deref(_)
                    | PatKind::Ref(..)
                    | PatKind::Guard(..)
                    | PatKind::Wild
                    | PatKind::Err(_) => {
                        // If the PatKind is Or, Box, or Ref, the decision is made later
                        // as these patterns contains subpatterns
                        // If the PatKind is Wild or Err, the decision is made based on the other patterns
                        // being examined
                    }
                }

                Ok(())
            })?
        }

        if needs_to_be_read {
            self.borrow_expr(discr, BorrowKind::Immutable)?;
        } else {
            let closure_def_id = match discr_place.place.base {
                PlaceBase::Upvar(upvar_id) => Some(upvar_id.closure_expr_id),
                _ => None,
            };

            self.delegate.borrow_mut().fake_read(
                &discr_place,
                FakeReadCause::ForMatchedPlace(closure_def_id),
                discr_place.hir_id,
            );

            // We always want to walk the discriminant. We want to make sure, for instance,
            // that the discriminant has been initialized.
            self.walk_expr(discr)?;
        }
        Ok(())
    }

    fn walk_local<F>(
        &self,
        expr: &hir::Expr<'_>,
        pat: &hir::Pat<'_>,
        els: Option<&hir::Block<'_>>,
        mut f: F,
    ) -> Result<(), Cx::Error>
    where
        F: FnMut() -> Result<(), Cx::Error>,
    {
        self.walk_expr(expr)?;
        let expr_place = self.cat_expr(expr)?;
        f()?;
        if let Some(els) = els {
            // borrowing because we need to test the discriminant
            self.maybe_read_scrutinee(expr, expr_place.clone(), from_ref(pat).iter())?;
            self.walk_block(els)?;
        }
        self.walk_irrefutable_pat(&expr_place, pat)?;
        Ok(())
    }

    /// Indicates that the value of `blk` will be consumed, meaning either copied or moved
    /// depending on its type.
    #[instrument(skip(self), level = "debug")]
    fn walk_block(&self, blk: &hir::Block<'_>) -> Result<(), Cx::Error> {
        for stmt in blk.stmts {
            self.walk_stmt(stmt)?;
        }

        if let Some(tail_expr) = blk.expr {
            self.consume_expr(tail_expr)?;
        }

        Ok(())
    }

    fn walk_struct_expr<'hir>(
        &self,
        fields: &[hir::ExprField<'_>],
        opt_with: &hir::StructTailExpr<'hir>,
    ) -> Result<(), Cx::Error> {
        // Consume the expressions supplying values for each field.
        for field in fields {
            self.consume_expr(field.expr)?;

            // The struct path probably didn't resolve
            if self.cx.typeck_results().opt_field_index(field.hir_id).is_none() {
                self.cx
                    .tcx()
                    .dcx()
                    .span_delayed_bug(field.span, "couldn't resolve index for field");
            }
        }

        let with_expr = match *opt_with {
            hir::StructTailExpr::Base(w) => &*w,
            hir::StructTailExpr::DefaultFields(_) | hir::StructTailExpr::None => {
                return Ok(());
            }
        };

        let with_place = self.cat_expr(with_expr)?;

        // Select just those fields of the `with`
        // expression that will actually be used
        match self.cx.try_structurally_resolve_type(with_expr.span, with_place.place.ty()).kind() {
            ty::Adt(adt, args) if adt.is_struct() => {
                // Consume those fields of the with expression that are needed.
                for (f_index, with_field) in adt.non_enum_variant().fields.iter_enumerated() {
                    let is_mentioned = fields.iter().any(|f| {
                        self.cx.typeck_results().opt_field_index(f.hir_id) == Some(f_index)
                    });
                    if !is_mentioned {
                        let field_place = self.cat_projection(
                            with_expr.hir_id,
                            with_place.clone(),
                            with_field.ty(self.cx.tcx(), args),
                            ProjectionKind::Field(f_index, FIRST_VARIANT),
                        );
                        self.consume_or_copy(&field_place, field_place.hir_id);
                    }
                }
            }
            _ => {
                // the base expression should always evaluate to a
                // struct; however, when EUV is run during typeck, it
                // may not. This will generate an error earlier in typeck,
                // so we can just ignore it.
                if self.cx.tainted_by_errors().is_ok() {
                    span_bug!(with_expr.span, "with expression doesn't evaluate to a struct");
                }
            }
        }

        // walk the with expression so that complex expressions
        // are properly handled.
        self.walk_expr(with_expr)?;

        Ok(())
    }

    /// Invoke the appropriate delegate calls for anything that gets
    /// consumed or borrowed as part of the automatic adjustment
    /// process.
    fn walk_adjustment(&self, expr: &hir::Expr<'_>) -> Result<(), Cx::Error> {
        let typeck_results = self.cx.typeck_results();
        let adjustments = typeck_results.expr_adjustments(expr);
        let mut place_with_id = self.cat_expr_unadjusted(expr)?;
        for adjustment in adjustments {
            debug!("walk_adjustment expr={:?} adj={:?}", expr, adjustment);
            match adjustment.kind {
                adjustment::Adjust::NeverToAny | adjustment::Adjust::Pointer(_) => {
                    // Creating a closure/fn-pointer or unsizing consumes
                    // the input and stores it into the resulting rvalue.
                    self.consume_or_copy(&place_with_id, place_with_id.hir_id);
                }

                adjustment::Adjust::Deref(None) => {}

                // Autoderefs for overloaded Deref calls in fact reference
                // their receiver. That is, if we have `(*x)` where `x`
                // is of type `Rc<T>`, then this in fact is equivalent to
                // `x.deref()`. Since `deref()` is declared with `&self`,
                // this is an autoref of `x`.
                adjustment::Adjust::Deref(Some(ref deref)) => {
                    let bk = ty::BorrowKind::from_mutbl(deref.mutbl);
                    self.delegate.borrow_mut().borrow(&place_with_id, place_with_id.hir_id, bk);
                }

                adjustment::Adjust::Borrow(ref autoref) => {
                    self.walk_autoref(expr, &place_with_id, autoref);
                }

                adjustment::Adjust::ReborrowPin(mutbl) => {
                    // Reborrowing a Pin is like a combinations of a deref and a borrow, so we do
                    // both.
                    let bk = match mutbl {
                        ty::Mutability::Not => ty::BorrowKind::Immutable,
                        ty::Mutability::Mut => ty::BorrowKind::Mutable,
                    };
                    self.delegate.borrow_mut().borrow(&place_with_id, place_with_id.hir_id, bk);
                }
            }
            place_with_id = self.cat_expr_adjusted(expr, place_with_id, adjustment)?;
        }

        Ok(())
    }

    /// Walks the autoref `autoref` applied to the autoderef'd
    /// `expr`. `base_place` is `expr` represented as a place,
    /// after all relevant autoderefs have occurred.
    fn walk_autoref(
        &self,
        expr: &hir::Expr<'_>,
        base_place: &PlaceWithHirId<'tcx>,
        autoref: &adjustment::AutoBorrow,
    ) {
        debug!(
            "walk_autoref(expr.hir_id={} base_place={:?} autoref={:?})",
            expr.hir_id, base_place, autoref
        );

        match *autoref {
            adjustment::AutoBorrow::Ref(m) => {
                self.delegate.borrow_mut().borrow(
                    base_place,
                    base_place.hir_id,
                    ty::BorrowKind::from_mutbl(m.into()),
                );
            }

            adjustment::AutoBorrow::RawPtr(m) => {
                debug!("walk_autoref: expr.hir_id={} base_place={:?}", expr.hir_id, base_place);

                self.delegate.borrow_mut().borrow(
                    base_place,
                    base_place.hir_id,
                    ty::BorrowKind::from_mutbl(m),
                );
            }
        }
    }

    fn walk_arm(
        &self,
        discr_place: &PlaceWithHirId<'tcx>,
        arm: &hir::Arm<'_>,
    ) -> Result<(), Cx::Error> {
        let closure_def_id = match discr_place.place.base {
            PlaceBase::Upvar(upvar_id) => Some(upvar_id.closure_expr_id),
            _ => None,
        };

        self.delegate.borrow_mut().fake_read(
            discr_place,
            FakeReadCause::ForMatchedPlace(closure_def_id),
            discr_place.hir_id,
        );
        self.walk_pat(discr_place, arm.pat, arm.guard.is_some())?;

        if let Some(ref e) = arm.guard {
            self.consume_expr(e)?;
        }

        self.consume_expr(arm.body)?;
        Ok(())
    }

    /// Walks a pat that occurs in isolation (i.e., top-level of fn argument or
    /// let binding, and *not* a match arm or nested pat.)
    fn walk_irrefutable_pat(
        &self,
        discr_place: &PlaceWithHirId<'tcx>,
        pat: &hir::Pat<'_>,
    ) -> Result<(), Cx::Error> {
        let closure_def_id = match discr_place.place.base {
            PlaceBase::Upvar(upvar_id) => Some(upvar_id.closure_expr_id),
            _ => None,
        };

        self.delegate.borrow_mut().fake_read(
            discr_place,
            FakeReadCause::ForLet(closure_def_id),
            discr_place.hir_id,
        );
        self.walk_pat(discr_place, pat, false)?;
        Ok(())
    }

    /// The core driver for walking a pattern
    #[instrument(skip(self), level = "debug")]
    fn walk_pat(
        &self,
        discr_place: &PlaceWithHirId<'tcx>,
        pat: &hir::Pat<'_>,
        has_guard: bool,
    ) -> Result<(), Cx::Error> {
        let tcx = self.cx.tcx();
        self.cat_pattern(discr_place.clone(), pat, &mut |place, pat| {
            match pat.kind {
                PatKind::Binding(_, canonical_id, ..) => {
                    debug!("walk_pat: binding place={:?} pat={:?}", place, pat);
                    let bm = self
                        .cx
                        .typeck_results()
                        .extract_binding_mode(tcx.sess, pat.hir_id, pat.span);
                    debug!("walk_pat: pat.hir_id={:?} bm={:?}", pat.hir_id, bm);

                    // pat_ty: the type of the binding being produced.
                    let pat_ty = self.node_ty(pat.hir_id)?;
                    debug!("walk_pat: pat_ty={:?}", pat_ty);

                    let def = Res::Local(canonical_id);
                    if let Ok(ref binding_place) = self.cat_res(pat.hir_id, pat.span, pat_ty, def) {
                        self.delegate.borrow_mut().bind(binding_place, binding_place.hir_id);
                    }

                    // Subtle: MIR desugaring introduces immutable borrows for each pattern
                    // binding when lowering pattern guards to ensure that the guard does not
                    // modify the scrutinee.
                    if has_guard {
                        self.delegate.borrow_mut().borrow(
                            place,
                            discr_place.hir_id,
                            BorrowKind::Immutable,
                        );
                    }

                    // It is also a borrow or copy/move of the value being matched.
                    // In a cases of pattern like `let pat = upvar`, don't use the span
                    // of the pattern, as this just looks confusing, instead use the span
                    // of the discriminant.
                    match bm.0 {
                        hir::ByRef::Yes(m) => {
                            let bk = ty::BorrowKind::from_mutbl(m);
                            self.delegate.borrow_mut().borrow(place, discr_place.hir_id, bk);
                        }
                        hir::ByRef::No => {
                            debug!("walk_pat binding consuming pat");
                            self.consume_or_copy(place, discr_place.hir_id);
                        }
                    }
                }
                PatKind::Deref(subpattern) => {
                    // A deref pattern is a bit special: the binding mode of its inner bindings
                    // determines whether to borrow *at the level of the deref pattern* rather than
                    // borrowing the bound place (since that inner place is inside the temporary that
                    // stores the result of calling `deref()`/`deref_mut()` so can't be captured).
                    let mutable = self.cx.typeck_results().pat_has_ref_mut_binding(subpattern);
                    let mutability =
                        if mutable { hir::Mutability::Mut } else { hir::Mutability::Not };
                    let bk = ty::BorrowKind::from_mutbl(mutability);
                    self.delegate.borrow_mut().borrow(place, discr_place.hir_id, bk);
                }
                PatKind::Never => {
                    // A `!` pattern always counts as an immutable read of the discriminant,
                    // even in an irrefutable pattern.
                    self.delegate.borrow_mut().borrow(
                        place,
                        discr_place.hir_id,
                        BorrowKind::Immutable,
                    );
                }
                _ => {}
            }

            Ok(())
        })
    }

    /// Handle the case where the current body contains a closure.
    ///
    /// When the current body being handled is a closure, then we must make sure that
    /// - The parent closure only captures Places from the nested closure that are not local to it.
    ///
    /// In the following example the closures `c` only captures `p.x` even though `incr`
    /// is a capture of the nested closure
    ///
    /// ```
    /// struct P { x: i32 }
    /// let mut p = P { x: 4 };
    /// let c = || {
    ///    let incr = 10;
    ///    let nested = || p.x += incr;
    /// };
    /// ```
    ///
    /// - When reporting the Place back to the Delegate, ensure that the UpvarId uses the enclosing
    /// closure as the DefId.
    #[instrument(skip(self), level = "debug")]
    fn walk_captures(&self, closure_expr: &hir::Closure<'_>) -> Result<(), Cx::Error> {
        fn upvar_is_local_variable(
            upvars: Option<&FxIndexMap<HirId, hir::Upvar>>,
            upvar_id: HirId,
            body_owner_is_closure: bool,
        ) -> bool {
            upvars.map(|upvars| !upvars.contains_key(&upvar_id)).unwrap_or(body_owner_is_closure)
        }

        let tcx = self.cx.tcx();
        let closure_def_id = closure_expr.def_id;
        // For purposes of this function, coroutine and closures are equivalent.
        let body_owner_is_closure = matches!(
            tcx.hir_body_owner_kind(self.cx.body_owner_def_id()),
            hir::BodyOwnerKind::Closure
        );

        // If we have a nested closure, we want to include the fake reads present in the nested
        // closure.
        if let Some(fake_reads) = self.cx.typeck_results().closure_fake_reads.get(&closure_def_id) {
            for (fake_read, cause, hir_id) in fake_reads.iter() {
                match fake_read.base {
                    PlaceBase::Upvar(upvar_id) => {
                        if upvar_is_local_variable(
                            self.upvars,
                            upvar_id.var_path.hir_id,
                            body_owner_is_closure,
                        ) {
                            // The nested closure might be fake reading the current (enclosing) closure's local variables.
                            // The only places we want to fake read before creating the parent closure are the ones that
                            // are not local to it/ defined by it.
                            //
                            // ```rust,ignore(cannot-test-this-because-pseudo-code)
                            // let v1 = (0, 1);
                            // let c = || { // fake reads: v1
                            //    let v2 = (0, 1);
                            //    let e = || { // fake reads: v1, v2
                            //       let (_, t1) = v1;
                            //       let (_, t2) = v2;
                            //    }
                            // }
                            // ```
                            // This check is performed when visiting the body of the outermost closure (`c`) and ensures
                            // that we don't add a fake read of v2 in c.
                            continue;
                        }
                    }
                    _ => {
                        bug!(
                            "Do not know how to get HirId out of Rvalue and StaticItem {:?}",
                            fake_read.base
                        );
                    }
                };
                self.delegate.borrow_mut().fake_read(
                    &PlaceWithHirId { place: fake_read.clone(), hir_id: *hir_id },
                    *cause,
                    *hir_id,
                );
            }
        }

        if let Some(min_captures) =
            self.cx.typeck_results().closure_min_captures.get(&closure_def_id)
        {
            for (var_hir_id, min_list) in min_captures.iter() {
                if self
                    .upvars
                    .map_or(body_owner_is_closure, |upvars| !upvars.contains_key(var_hir_id))
                {
                    // The nested closure might be capturing the current (enclosing) closure's local variables.
                    // We check if the root variable is ever mentioned within the enclosing closure, if not
                    // then for the current body (if it's a closure) these aren't captures, we will ignore them.
                    continue;
                }
                for captured_place in min_list {
                    let place = &captured_place.place;
                    let capture_info = captured_place.info;

                    let place_base = if body_owner_is_closure {
                        // Mark the place to be captured by the enclosing closure
                        PlaceBase::Upvar(ty::UpvarId::new(*var_hir_id, self.cx.body_owner_def_id()))
                    } else {
                        // If the body owner isn't a closure then the variable must
                        // be a local variable
                        PlaceBase::Local(*var_hir_id)
                    };
                    let closure_hir_id = tcx.local_def_id_to_hir_id(closure_def_id);
                    let place_with_id = PlaceWithHirId::new(
                        capture_info
                            .path_expr_id
                            .unwrap_or(capture_info.capture_kind_expr_id.unwrap_or(closure_hir_id)),
                        place.base_ty,
                        place_base,
                        place.projections.clone(),
                    );

                    match capture_info.capture_kind {
                        ty::UpvarCapture::ByValue => {
                            self.consume_or_copy(&place_with_id, place_with_id.hir_id);
                        }
                        ty::UpvarCapture::ByUse => {
                            self.consume_clone_or_copy(&place_with_id, place_with_id.hir_id);
                        }
                        ty::UpvarCapture::ByRef(upvar_borrow) => {
                            self.delegate.borrow_mut().borrow(
                                &place_with_id,
                                place_with_id.hir_id,
                                upvar_borrow,
                            );
                        }
                    }
                }
            }
        }

        Ok(())
    }
}

/// The job of the methods whose name starts with `cat_` is to analyze
/// expressions and construct the corresponding [`Place`]s. The `cat`
/// stands for "categorize", this is a leftover from long ago when
/// places were called "categorizations".
///
/// Note that a [`Place`] differs somewhat from the expression itself. For
/// example, auto-derefs are explicit. Also, an index `a[b]` is decomposed into
/// two operations: a dereference to reach the array data and then an index to
/// jump forward to the relevant item.
impl<'tcx, Cx: TypeInformationCtxt<'tcx>, D: Delegate<'tcx>> ExprUseVisitor<'tcx, Cx, D> {
    fn resolve_type_vars_or_bug(
        &self,
        id: HirId,
        ty: Option<Ty<'tcx>>,
    ) -> Result<Ty<'tcx>, Cx::Error> {
        match ty {
            Some(ty) => {
                let ty = self.cx.resolve_vars_if_possible(ty);
                self.cx.error_reported_in_ty(ty)?;
                if ty.is_ty_var() {
                    debug!("resolve_type_vars_or_bug: infer var from {:?}", ty);
                    Err(self.cx.report_bug(self.cx.tcx().hir_span(id), "encountered type variable"))
                } else {
                    Ok(ty)
                }
            }
            None => {
                // FIXME: We shouldn't be relying on the infcx being tainted.
                self.cx.tainted_by_errors()?;
                bug!("no type for node {} in ExprUseVisitor", self.cx.tcx().hir_id_to_string(id));
            }
        }
    }

    fn node_ty(&self, hir_id: HirId) -> Result<Ty<'tcx>, Cx::Error> {
        self.resolve_type_vars_or_bug(hir_id, self.cx.typeck_results().node_type_opt(hir_id))
    }

    fn expr_ty(&self, expr: &hir::Expr<'_>) -> Result<Ty<'tcx>, Cx::Error> {
        self.resolve_type_vars_or_bug(expr.hir_id, self.cx.typeck_results().expr_ty_opt(expr))
    }

    fn expr_ty_adjusted(&self, expr: &hir::Expr<'_>) -> Result<Ty<'tcx>, Cx::Error> {
        self.resolve_type_vars_or_bug(
            expr.hir_id,
            self.cx.typeck_results().expr_ty_adjusted_opt(expr),
        )
    }

    /// Returns the type of value that this pattern matches against.
    /// Some non-obvious cases:
    ///
    /// - a `ref x` binding matches against a value of type `T` and gives
    ///   `x` the type `&T`; we return `T`.
    /// - a pattern with implicit derefs (thanks to default binding
    ///   modes #42640) may look like `Some(x)` but in fact have
    ///   implicit deref patterns attached (e.g., it is really
    ///   `&Some(x)`). In that case, we return the "outermost" type
    ///   (e.g., `&Option<T>`).
    fn pat_ty_adjusted(&self, pat: &hir::Pat<'_>) -> Result<Ty<'tcx>, Cx::Error> {
        // Check for implicit `&` types wrapping the pattern; note
        // that these are never attached to binding patterns, so
        // actually this is somewhat "disjoint" from the code below
        // that aims to account for `ref x`.
        if let Some(vec) = self.cx.typeck_results().pat_adjustments().get(pat.hir_id) {
            if let Some(first_ty) = vec.first() {
                debug!("pat_ty(pat={:?}) found adjusted ty `{:?}`", pat, first_ty);
                return Ok(*first_ty);
            }
        } else if let PatKind::Ref(subpat, _) = pat.kind
            && self.cx.typeck_results().skipped_ref_pats().contains(pat.hir_id)
        {
            return self.pat_ty_adjusted(subpat);
        }

        self.pat_ty_unadjusted(pat)
    }

    /// Like [`Self::pat_ty_adjusted`], but ignores implicit `&` patterns.
    fn pat_ty_unadjusted(&self, pat: &hir::Pat<'_>) -> Result<Ty<'tcx>, Cx::Error> {
        let base_ty = self.node_ty(pat.hir_id)?;
        trace!(?base_ty);

        // This code detects whether we are looking at a `ref x`,
        // and if so, figures out what the type *being borrowed* is.
        match pat.kind {
            PatKind::Binding(..) => {
                let bm = *self
                    .cx
                    .typeck_results()
                    .pat_binding_modes()
                    .get(pat.hir_id)
                    .expect("missing binding mode");

                if matches!(bm.0, hir::ByRef::Yes(_)) {
                    // a bind-by-ref means that the base_ty will be the type of the ident itself,
                    // but what we want here is the type of the underlying value being borrowed.
                    // So peel off one-level, turning the &T into T.
                    match self
                        .cx
                        .try_structurally_resolve_type(pat.span, base_ty)
                        .builtin_deref(false)
                    {
                        Some(ty) => Ok(ty),
                        None => {
                            debug!("By-ref binding of non-derefable type");
                            Err(self
                                .cx
                                .report_bug(pat.span, "by-ref binding of non-derefable type"))
                        }
                    }
                } else {
                    Ok(base_ty)
                }
            }
            _ => Ok(base_ty),
        }
    }

    fn cat_expr(&self, expr: &hir::Expr<'_>) -> Result<PlaceWithHirId<'tcx>, Cx::Error> {
        self.cat_expr_(expr, self.cx.typeck_results().expr_adjustments(expr))
    }

    /// This recursion helper avoids going through *too many*
    /// adjustments, since *only* non-overloaded deref recurses.
    fn cat_expr_(
        &self,
        expr: &hir::Expr<'_>,
        adjustments: &[adjustment::Adjustment<'tcx>],
    ) -> Result<PlaceWithHirId<'tcx>, Cx::Error> {
        match adjustments.split_last() {
            None => self.cat_expr_unadjusted(expr),
            Some((adjustment, previous)) => {
                self.cat_expr_adjusted_with(expr, || self.cat_expr_(expr, previous), adjustment)
            }
        }
    }

    fn cat_expr_adjusted(
        &self,
        expr: &hir::Expr<'_>,
        previous: PlaceWithHirId<'tcx>,
        adjustment: &adjustment::Adjustment<'tcx>,
    ) -> Result<PlaceWithHirId<'tcx>, Cx::Error> {
        self.cat_expr_adjusted_with(expr, || Ok(previous), adjustment)
    }

    fn cat_expr_adjusted_with<F>(
        &self,
        expr: &hir::Expr<'_>,
        previous: F,
        adjustment: &adjustment::Adjustment<'tcx>,
    ) -> Result<PlaceWithHirId<'tcx>, Cx::Error>
    where
        F: FnOnce() -> Result<PlaceWithHirId<'tcx>, Cx::Error>,
    {
        let target = self.cx.resolve_vars_if_possible(adjustment.target);
        match adjustment.kind {
            adjustment::Adjust::Deref(overloaded) => {
                // Equivalent to *expr or something similar.
                let base = if let Some(deref) = overloaded {
                    let ref_ty = Ty::new_ref(
                        self.cx.tcx(),
                        self.cx.tcx().lifetimes.re_erased,
                        target,
                        deref.mutbl,
                    );
                    self.cat_rvalue(expr.hir_id, ref_ty)
                } else {
                    previous()?
                };
                self.cat_deref(expr.hir_id, base)
            }

            adjustment::Adjust::NeverToAny
            | adjustment::Adjust::Pointer(_)
            | adjustment::Adjust::Borrow(_)
            | adjustment::Adjust::ReborrowPin(..) => {
                // Result is an rvalue.
                Ok(self.cat_rvalue(expr.hir_id, target))
            }
        }
    }

    fn cat_expr_unadjusted(&self, expr: &hir::Expr<'_>) -> Result<PlaceWithHirId<'tcx>, Cx::Error> {
        let expr_ty = self.expr_ty(expr)?;
        match expr.kind {
            hir::ExprKind::Unary(hir::UnOp::Deref, e_base) => {
                if self.cx.typeck_results().is_method_call(expr) {
                    self.cat_overloaded_place(expr, e_base)
                } else {
                    let base = self.cat_expr(e_base)?;
                    self.cat_deref(expr.hir_id, base)
                }
            }

            hir::ExprKind::Field(base, _) => {
                let base = self.cat_expr(base)?;
                debug!(?base);

                let field_idx = self
                    .cx
                    .typeck_results()
                    .field_indices()
                    .get(expr.hir_id)
                    .cloned()
                    .expect("Field index not found");

                Ok(self.cat_projection(
                    expr.hir_id,
                    base,
                    expr_ty,
                    ProjectionKind::Field(field_idx, FIRST_VARIANT),
                ))
            }

            hir::ExprKind::Index(base, _, _) => {
                if self.cx.typeck_results().is_method_call(expr) {
                    // If this is an index implemented by a method call, then it
                    // will include an implicit deref of the result.
                    // The call to index() returns a `&T` value, which
                    // is an rvalue. That is what we will be
                    // dereferencing.
                    self.cat_overloaded_place(expr, base)
                } else {
                    let base = self.cat_expr(base)?;
                    Ok(self.cat_projection(expr.hir_id, base, expr_ty, ProjectionKind::Index))
                }
            }

            hir::ExprKind::Path(ref qpath) => {
                let res = self.cx.typeck_results().qpath_res(qpath, expr.hir_id);
                self.cat_res(expr.hir_id, expr.span, expr_ty, res)
            }

            // both type ascription and unsafe binder casts don't affect
            // the place-ness of the subexpression.
            hir::ExprKind::Type(e, _) => self.cat_expr(e),
            hir::ExprKind::UnsafeBinderCast(_, e, _) => self.cat_expr(e),

            hir::ExprKind::AddrOf(..)
            | hir::ExprKind::Call(..)
            | hir::ExprKind::Use(..)
            | hir::ExprKind::Assign(..)
            | hir::ExprKind::AssignOp(..)
            | hir::ExprKind::Closure { .. }
            | hir::ExprKind::Ret(..)
            | hir::ExprKind::Become(..)
            | hir::ExprKind::Unary(..)
            | hir::ExprKind::Yield(..)
            | hir::ExprKind::MethodCall(..)
            | hir::ExprKind::Cast(..)
            | hir::ExprKind::DropTemps(..)
            | hir::ExprKind::Array(..)
            | hir::ExprKind::If(..)
            | hir::ExprKind::Tup(..)
            | hir::ExprKind::Binary(..)
            | hir::ExprKind::Block(..)
            | hir::ExprKind::Let(..)
            | hir::ExprKind::Loop(..)
            | hir::ExprKind::Match(..)
            | hir::ExprKind::Lit(..)
            | hir::ExprKind::ConstBlock(..)
            | hir::ExprKind::Break(..)
            | hir::ExprKind::Continue(..)
            | hir::ExprKind::Struct(..)
            | hir::ExprKind::Repeat(..)
            | hir::ExprKind::InlineAsm(..)
            | hir::ExprKind::OffsetOf(..)
            | hir::ExprKind::Err(_) => Ok(self.cat_rvalue(expr.hir_id, expr_ty)),
        }
    }

    fn cat_res(
        &self,
        hir_id: HirId,
        span: Span,
        expr_ty: Ty<'tcx>,
        res: Res,
    ) -> Result<PlaceWithHirId<'tcx>, Cx::Error> {
        match res {
            Res::Def(
                DefKind::Ctor(..)
                | DefKind::Const
                | DefKind::ConstParam
                | DefKind::AssocConst
                | DefKind::Fn
                | DefKind::AssocFn,
                _,
            )
            | Res::SelfCtor(..) => Ok(self.cat_rvalue(hir_id, expr_ty)),

            Res::Def(DefKind::Static { .. }, _) => {
                Ok(PlaceWithHirId::new(hir_id, expr_ty, PlaceBase::StaticItem, Vec::new()))
            }

            Res::Local(var_id) => {
                if self.upvars.is_some_and(|upvars| upvars.contains_key(&var_id)) {
                    self.cat_upvar(hir_id, var_id)
                } else {
                    Ok(PlaceWithHirId::new(hir_id, expr_ty, PlaceBase::Local(var_id), Vec::new()))
                }
            }

            def => span_bug!(span, "unexpected definition in ExprUseVisitor: {:?}", def),
        }
    }

    /// Categorize an upvar.
    ///
    /// Note: the actual upvar access contains invisible derefs of closure
    /// environment and upvar reference as appropriate. Only regionck cares
    /// about these dereferences, so we let it compute them as needed.
    fn cat_upvar(&self, hir_id: HirId, var_id: HirId) -> Result<PlaceWithHirId<'tcx>, Cx::Error> {
        let closure_expr_def_id = self.cx.body_owner_def_id();

        let upvar_id = ty::UpvarId {
            var_path: ty::UpvarPath { hir_id: var_id },
            closure_expr_id: closure_expr_def_id,
        };
        let var_ty = self.node_ty(var_id)?;

        Ok(PlaceWithHirId::new(hir_id, var_ty, PlaceBase::Upvar(upvar_id), Vec::new()))
    }

    fn cat_rvalue(&self, hir_id: HirId, expr_ty: Ty<'tcx>) -> PlaceWithHirId<'tcx> {
        PlaceWithHirId::new(hir_id, expr_ty, PlaceBase::Rvalue, Vec::new())
    }

    fn cat_projection(
        &self,
        node: HirId,
        base_place: PlaceWithHirId<'tcx>,
        ty: Ty<'tcx>,
        kind: ProjectionKind,
    ) -> PlaceWithHirId<'tcx> {
        let place_ty = base_place.place.ty();
        let mut projections = base_place.place.projections;

        let node_ty = self.cx.typeck_results().node_type(node);
        // Opaque types can't have field projections, but we can instead convert
        // the current place in-place (heh) to the hidden type, and then apply all
        // follow up projections on that.
        if node_ty != place_ty
            && self
                .cx
                .try_structurally_resolve_type(self.cx.tcx().hir_span(base_place.hir_id), place_ty)
                .is_impl_trait()
        {
            projections.push(Projection { kind: ProjectionKind::OpaqueCast, ty: node_ty });
        }
        projections.push(Projection { kind, ty });
        PlaceWithHirId::new(node, base_place.place.base_ty, base_place.place.base, projections)
    }

    fn cat_overloaded_place(
        &self,
        expr: &hir::Expr<'_>,
        base: &hir::Expr<'_>,
    ) -> Result<PlaceWithHirId<'tcx>, Cx::Error> {
        // Reconstruct the output assuming it's a reference with the
        // same region and mutability as the receiver. This holds for
        // `Deref(Mut)::Deref(_mut)` and `Index(Mut)::index(_mut)`.
        let place_ty = self.expr_ty(expr)?;
        let base_ty = self.expr_ty_adjusted(base)?;

        let ty::Ref(region, _, mutbl) =
            *self.cx.try_structurally_resolve_type(base.span, base_ty).kind()
        else {
            span_bug!(expr.span, "cat_overloaded_place: base is not a reference");
        };
        let ref_ty = Ty::new_ref(self.cx.tcx(), region, place_ty, mutbl);

        let base = self.cat_rvalue(expr.hir_id, ref_ty);
        self.cat_deref(expr.hir_id, base)
    }

    fn cat_deref(
        &self,
        node: HirId,
        base_place: PlaceWithHirId<'tcx>,
    ) -> Result<PlaceWithHirId<'tcx>, Cx::Error> {
        let base_curr_ty = base_place.place.ty();
        let deref_ty = match self
            .cx
            .try_structurally_resolve_type(self.cx.tcx().hir_span(base_place.hir_id), base_curr_ty)
            .builtin_deref(true)
        {
            Some(ty) => ty,
            None => {
                debug!("explicit deref of non-derefable type: {:?}", base_curr_ty);
                return Err(self.cx.report_bug(
                    self.cx.tcx().hir_span(node),
                    "explicit deref of non-derefable type",
                ));
            }
        };
        let mut projections = base_place.place.projections;
        projections.push(Projection { kind: ProjectionKind::Deref, ty: deref_ty });

        Ok(PlaceWithHirId::new(node, base_place.place.base_ty, base_place.place.base, projections))
    }

    /// Returns the variant index for an ADT used within a Struct or TupleStruct pattern
    /// Here `pat_hir_id` is the HirId of the pattern itself.
    fn variant_index_for_adt(
        &self,
        qpath: &hir::QPath<'_>,
        pat_hir_id: HirId,
        span: Span,
    ) -> Result<VariantIdx, Cx::Error> {
        let res = self.cx.typeck_results().qpath_res(qpath, pat_hir_id);
        let ty = self.cx.typeck_results().node_type(pat_hir_id);
        let ty::Adt(adt_def, _) = self.cx.try_structurally_resolve_type(span, ty).kind() else {
            return Err(self
                .cx
                .report_bug(span, "struct or tuple struct pattern not applied to an ADT"));
        };

        match res {
            Res::Def(DefKind::Variant, variant_id) => Ok(adt_def.variant_index_with_id(variant_id)),
            Res::Def(DefKind::Ctor(CtorOf::Variant, ..), variant_ctor_id) => {
                Ok(adt_def.variant_index_with_ctor_id(variant_ctor_id))
            }
            Res::Def(DefKind::Ctor(CtorOf::Struct, ..), _)
            | Res::Def(DefKind::Struct | DefKind::Union | DefKind::TyAlias | DefKind::AssocTy, _)
            | Res::SelfCtor(..)
            | Res::SelfTyParam { .. }
            | Res::SelfTyAlias { .. } => {
                // Structs and Unions have only have one variant.
                Ok(FIRST_VARIANT)
            }
            _ => bug!("expected ADT path, found={:?}", res),
        }
    }

    /// Returns the total number of fields in an ADT variant used within a pattern.
    /// Here `pat_hir_id` is the HirId of the pattern itself.
    fn total_fields_in_adt_variant(
        &self,
        pat_hir_id: HirId,
        variant_index: VariantIdx,
        span: Span,
    ) -> Result<usize, Cx::Error> {
        let ty = self.cx.typeck_results().node_type(pat_hir_id);
        match self.cx.try_structurally_resolve_type(span, ty).kind() {
            ty::Adt(adt_def, _) => Ok(adt_def.variant(variant_index).fields.len()),
            _ => {
                self.cx
                    .tcx()
                    .dcx()
                    .span_bug(span, "struct or tuple struct pattern not applied to an ADT");
            }
        }
    }

    /// Returns the total number of fields in a tuple used within a Tuple pattern.
    /// Here `pat_hir_id` is the HirId of the pattern itself.
    fn total_fields_in_tuple(&self, pat_hir_id: HirId, span: Span) -> Result<usize, Cx::Error> {
        let ty = self.cx.typeck_results().node_type(pat_hir_id);
        match self.cx.try_structurally_resolve_type(span, ty).kind() {
            ty::Tuple(args) => Ok(args.len()),
            _ => Err(self.cx.report_bug(span, "tuple pattern not applied to a tuple")),
        }
    }

    /// Here, `place` is the `PlaceWithHirId` being matched and pat is the pattern it
    /// is being matched against.
    ///
    /// In general, the way that this works is that we walk down the pattern,
    /// constructing a `PlaceWithHirId` that represents the path that will be taken
    /// to reach the value being matched.
    fn cat_pattern<F>(
        &self,
        mut place_with_id: PlaceWithHirId<'tcx>,
        pat: &hir::Pat<'_>,
        op: &mut F,
    ) -> Result<(), Cx::Error>
    where
        F: FnMut(&PlaceWithHirId<'tcx>, &hir::Pat<'_>) -> Result<(), Cx::Error>,
    {
        // If (pattern) adjustments are active for this pattern, adjust the `PlaceWithHirId` correspondingly.
        // `PlaceWithHirId`s are constructed differently from patterns. For example, in
        //
        // ```
        // match foo {
        //     &&Some(x, ) => { ... },
        //     _ => { ... },
        // }
        // ```
        //
        // the pattern `&&Some(x,)` is represented as `Ref { Ref { TupleStruct }}`. To build the
        // corresponding `PlaceWithHirId` we start with the `PlaceWithHirId` for `foo`, and then, by traversing the
        // pattern, try to answer the question: given the address of `foo`, how is `x` reached?
        //
        // `&&Some(x,)` `place_foo`
        //  `&Some(x,)` `deref { place_foo}`
        //   `Some(x,)` `deref { deref { place_foo }}`
        //       `(x,)` `field0 { deref { deref { place_foo }}}` <- resulting place
        //
        // The above example has no adjustments. If the code were instead the (after adjustments,
        // equivalent) version
        //
        // ```
        // match foo {
        //     Some(x, ) => { ... },
        //     _ => { ... },
        // }
        // ```
        //
        // Then we see that to get the same result, we must start with
        // `deref { deref { place_foo }}` instead of `place_foo` since the pattern is now `Some(x,)`
        // and not `&&Some(x,)`, even though its assigned type is that of `&&Some(x,)`.
        for _ in
            0..self.cx.typeck_results().pat_adjustments().get(pat.hir_id).map_or(0, |v| v.len())
        {
            debug!("applying adjustment to place_with_id={:?}", place_with_id);
            place_with_id = self.cat_deref(pat.hir_id, place_with_id)?;
        }
        let place_with_id = place_with_id; // lose mutability
        debug!("applied adjustment derefs to get place_with_id={:?}", place_with_id);

        // Invoke the callback, but only now, after the `place_with_id` has adjusted.
        //
        // To see that this makes sense, consider `match &Some(3) { Some(x) => { ... }}`. In that
        // case, the initial `place_with_id` will be that for `&Some(3)` and the pattern is `Some(x)`. We
        // don't want to call `op` with these incompatible values. As written, what happens instead
        // is that `op` is called with the adjusted place (that for `*&Some(3)`) and the pattern
        // `Some(x)` (which matches). Recursing once more, `*&Some(3)` and the pattern `Some(x)`
        // result in the place `Downcast<Some>(*&Some(3)).0` associated to `x` and invoke `op` with
        // that (where the `ref` on `x` is implied).
        op(&place_with_id, pat)?;

        match pat.kind {
            PatKind::Tuple(subpats, dots_pos) => {
                // (p1, ..., pN)
                let total_fields = self.total_fields_in_tuple(pat.hir_id, pat.span)?;

                for (i, subpat) in subpats.iter().enumerate_and_adjust(total_fields, dots_pos) {
                    let subpat_ty = self.pat_ty_adjusted(subpat)?;
                    let projection_kind =
                        ProjectionKind::Field(FieldIdx::from_usize(i), FIRST_VARIANT);
                    let sub_place = self.cat_projection(
                        pat.hir_id,
                        place_with_id.clone(),
                        subpat_ty,
                        projection_kind,
                    );
                    self.cat_pattern(sub_place, subpat, op)?;
                }
            }

            PatKind::TupleStruct(ref qpath, subpats, dots_pos) => {
                // S(p1, ..., pN)
                let variant_index = self.variant_index_for_adt(qpath, pat.hir_id, pat.span)?;
                let total_fields =
                    self.total_fields_in_adt_variant(pat.hir_id, variant_index, pat.span)?;

                for (i, subpat) in subpats.iter().enumerate_and_adjust(total_fields, dots_pos) {
                    let subpat_ty = self.pat_ty_adjusted(subpat)?;
                    let projection_kind =
                        ProjectionKind::Field(FieldIdx::from_usize(i), variant_index);
                    let sub_place = self.cat_projection(
                        pat.hir_id,
                        place_with_id.clone(),
                        subpat_ty,
                        projection_kind,
                    );
                    self.cat_pattern(sub_place, subpat, op)?;
                }
            }

            PatKind::Struct(ref qpath, field_pats, _) => {
                // S { f1: p1, ..., fN: pN }

                let variant_index = self.variant_index_for_adt(qpath, pat.hir_id, pat.span)?;

                for fp in field_pats {
                    let field_ty = self.pat_ty_adjusted(fp.pat)?;
                    let field_index = self
                        .cx
                        .typeck_results()
                        .field_indices()
                        .get(fp.hir_id)
                        .cloned()
                        .expect("no index for a field");

                    let field_place = self.cat_projection(
                        pat.hir_id,
                        place_with_id.clone(),
                        field_ty,
                        ProjectionKind::Field(field_index, variant_index),
                    );
                    self.cat_pattern(field_place, fp.pat, op)?;
                }
            }

            PatKind::Or(pats) => {
                for pat in pats {
                    self.cat_pattern(place_with_id.clone(), pat, op)?;
                }
            }

            PatKind::Binding(.., Some(subpat)) | PatKind::Guard(subpat, _) => {
                self.cat_pattern(place_with_id, subpat, op)?;
            }

            PatKind::Ref(subpat, _)
                if self.cx.typeck_results().skipped_ref_pats().contains(pat.hir_id) =>
            {
                self.cat_pattern(place_with_id, subpat, op)?;
            }

            PatKind::Box(subpat) | PatKind::Ref(subpat, _) => {
                // box p1, &p1, &mut p1. we can ignore the mutability of
                // PatKind::Ref since that information is already contained
                // in the type.
                let subplace = self.cat_deref(pat.hir_id, place_with_id)?;
                self.cat_pattern(subplace, subpat, op)?;
            }
            PatKind::Deref(subpat) => {
                let mutable = self.cx.typeck_results().pat_has_ref_mut_binding(subpat);
                let mutability = if mutable { hir::Mutability::Mut } else { hir::Mutability::Not };
                let re_erased = self.cx.tcx().lifetimes.re_erased;
                let ty = self.pat_ty_adjusted(subpat)?;
                let ty = Ty::new_ref(self.cx.tcx(), re_erased, ty, mutability);
                // A deref pattern generates a temporary.
                let base = self.cat_rvalue(pat.hir_id, ty);
                let place = self.cat_deref(pat.hir_id, base)?;
                self.cat_pattern(place, subpat, op)?;
            }

            PatKind::Slice(before, ref slice, after) => {
                let Some(element_ty) = self
                    .cx
                    .try_structurally_resolve_type(pat.span, place_with_id.place.ty())
                    .builtin_index()
                else {
                    debug!("explicit index of non-indexable type {:?}", place_with_id);
                    return Err(self
                        .cx
                        .report_bug(pat.span, "explicit index of non-indexable type"));
                };
                let elt_place = self.cat_projection(
                    pat.hir_id,
                    place_with_id.clone(),
                    element_ty,
                    ProjectionKind::Index,
                );
                for before_pat in before {
                    self.cat_pattern(elt_place.clone(), before_pat, op)?;
                }
                if let Some(slice_pat) = *slice {
                    let slice_pat_ty = self.pat_ty_adjusted(slice_pat)?;
                    let slice_place = self.cat_projection(
                        pat.hir_id,
                        place_with_id,
                        slice_pat_ty,
                        ProjectionKind::Subslice,
                    );
                    self.cat_pattern(slice_place, slice_pat, op)?;
                }
                for after_pat in after {
                    self.cat_pattern(elt_place.clone(), after_pat, op)?;
                }
            }

            PatKind::Binding(.., None)
            | PatKind::Expr(..)
            | PatKind::Range(..)
            | PatKind::Never
            | PatKind::Missing
            | PatKind::Wild
            | PatKind::Err(_) => {
                // always ok
            }
        }

        Ok(())
    }

    fn is_multivariant_adt(&self, ty: Ty<'tcx>, span: Span) -> bool {
        if let ty::Adt(def, _) = self.cx.try_structurally_resolve_type(span, ty).kind() {
            // Note that if a non-exhaustive SingleVariant is defined in another crate, we need
            // to assume that more cases will be added to the variant in the future. This mean
            // that we should handle non-exhaustive SingleVariant the same way we would handle
            // a MultiVariant.
            def.variants().len() > 1 || def.variant_list_has_applicable_non_exhaustive()
        } else {
            false
        }
    }
}
