//! A different sort of visitor for walking fn bodies. Unlike the
//! normal visitor, which just walks the entire body in one shot, the
//! `ExprUseVisitor` determines how expressions are being used.
//!
//! This is only used for upvar inference.

use either::Either;
use hir_def::{
    AdtId, HasModule, VariantId,
    attrs::AttrFlags,
    hir::{
        Array, AsmOperand, BindingId, Expr, ExprId, ExprOrPatId, MatchArm, Pat, PatId,
        RecordLitField, RecordSpread, Statement,
    },
    resolver::ValueNs,
};
use rustc_ast_ir::{try_visit, visit::VisitorResult};
use rustc_type_ir::{
    FallibleTypeFolder, TypeFoldable, TypeFolder, TypeVisitable, TypeVisitor,
    inherent::{AdtDef, IntoKind, Ty as _},
};
use smallvec::{SmallVec, smallvec};
use syntax::ast::{BinaryOp, UnaryOp};
use tracing::{debug, instrument};

use crate::{
    Adjust, Adjustment, AutoBorrow, BindingMode,
    infer::{CaptureSourceStack, InferenceContext, UpvarCapture, closure::analysis::BorrowKind},
    method_resolution::CandidateId,
    next_solver::{DbInterner, ErrorGuaranteed, StoredTy, Ty, TyKind},
    upvars::UpvarsRef,
    utils::EnumerateAndAdjustIterator,
};

type Result<T = (), E = ErrorGuaranteed> = std::result::Result<T, E>;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ProjectionKind {
    /// A dereference of a pointer, reference or `Box<T>` of the given type.
    Deref,

    /// `B.F` where `B` is the base expression and `F` is
    /// the field. The field is identified by which variant
    /// it appears in along with a field index. The variant
    /// is used for enums.
    Field { field_idx: u32, variant_idx: u32 },

    /// Some index like `B[x]`, where `B` is the base
    /// expression. We don't preserve the index `x` because
    /// we won't need it.
    Index,

    /// A subslice covering a range of values like `B[x..y]`.
    Subslice,

    /// `unwrap_binder!(expr)`
    UnwrapUnsafeBinder,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PlaceBase {
    /// A temporary variable.
    Rvalue,
    /// A named `static` item.
    StaticItem,
    /// A named local variable.
    Local(BindingId),
    /// An upvar referenced by closure env.
    Upvar { closure: ExprId, var_id: BindingId },
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Projection {
    /// Type after the projection is applied.
    pub ty: StoredTy,

    /// Defines the kind of access made by the projection.
    pub kind: ProjectionKind,
}

/// A `Place` represents how a value is located in memory. This does not
/// always correspond to a syntactic place expression. For example, when
/// processing a pattern, a `Place` can be used to refer to the sub-value
/// currently being inspected.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Place {
    /// The type of the `PlaceBase`
    pub base_ty: StoredTy,
    /// The "outermost" place that holds this value.
    pub base: PlaceBase,
    /// How this place is derived from the base place.
    pub projections: Vec<Projection>,
}

impl<'db> TypeVisitable<DbInterner<'db>> for Place {
    fn visit_with<V: TypeVisitor<DbInterner<'db>>>(&self, visitor: &mut V) -> V::Result {
        let Self { base_ty, base: _, projections } = self;
        try_visit!(base_ty.as_ref().visit_with(visitor));
        for proj in projections {
            let Projection { ty, kind: _ } = proj;
            try_visit!(ty.as_ref().visit_with(visitor));
        }
        V::Result::output()
    }
}

impl<'db> TypeFoldable<DbInterner<'db>> for Place {
    fn try_fold_with<F: FallibleTypeFolder<DbInterner<'db>>>(
        self,
        folder: &mut F,
    ) -> Result<Self, F::Error> {
        let Self { base_ty, base, projections } = self;
        let base_ty = base_ty.as_ref().try_fold_with(folder)?.store();
        let projections = projections
            .into_iter()
            .map(|proj| {
                let Projection { ty, kind } = proj;
                let ty = ty.as_ref().try_fold_with(folder)?.store();
                Ok(Projection { ty, kind })
            })
            .collect::<Result<_, _>>()?;
        Ok(Self { base_ty, base, projections })
    }

    fn fold_with<F: TypeFolder<DbInterner<'db>>>(self, folder: &mut F) -> Self {
        let Self { base_ty, base, projections } = self;
        let base_ty = base_ty.as_ref().fold_with(folder).store();
        let projections = projections
            .into_iter()
            .map(|proj| {
                let Projection { ty, kind } = proj;
                let ty = ty.as_ref().fold_with(folder).store();
                Projection { ty, kind }
            })
            .collect();
        Self { base_ty, base, projections }
    }
}

impl Place {
    /// Returns an iterator of the types that have to be dereferenced to access
    /// the `Place`.
    ///
    /// The types are in the reverse order that they are applied. So if
    /// `x: &*const u32` and the `Place` is `**x`, then the types returned are
    ///`*const u32` then `&*const u32`.
    pub fn deref_tys<'db>(&self) -> impl Iterator<Item = Ty<'db>> {
        self.projections.iter().enumerate().rev().filter_map(move |(index, proj)| {
            if ProjectionKind::Deref == proj.kind {
                Some(self.ty_before_projection(index))
            } else {
                None
            }
        })
    }

    /// Returns the type of this `Place` after all projections have been applied.
    pub fn ty<'db>(&self) -> Ty<'db> {
        self.projections.last().map_or(self.base_ty.as_ref(), |proj| proj.ty.as_ref())
    }

    /// Returns the type of this `Place` immediately before `projection_index`th projection
    /// is applied.
    pub fn ty_before_projection<'db>(&self, projection_index: usize) -> Ty<'db> {
        assert!(projection_index < self.projections.len());
        if projection_index == 0 {
            self.base_ty.as_ref()
        } else {
            self.projections[projection_index - 1].ty.as_ref()
        }
    }
}

/// A `PlaceWithOrigin` represents how a value is located in memory. This does not
/// always correspond to a syntactic place expression. For example, when
/// processing a pattern, a `Place` can be used to refer to the sub-value
/// currently being inspected.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct PlaceWithOrigin {
    /// `ExprId`s or `PatId`s of the expressions or patterns producing this value.
    pub origins: SmallVec<[CaptureSourceStack; 2]>,

    /// Information about the `Place`.
    pub place: Place,
}

impl PlaceWithOrigin {
    fn new_no_projections<'db>(
        origin: impl Into<ExprOrPatId>,
        base_ty: Ty<'db>,
        base: PlaceBase,
    ) -> PlaceWithOrigin {
        Self::new(
            smallvec![CaptureSourceStack::from_single(origin.into())],
            base_ty,
            base,
            Vec::new(),
        )
    }

    fn new<'db>(
        origins: SmallVec<[CaptureSourceStack; 2]>,
        base_ty: Ty<'db>,
        base: PlaceBase,
        projections: Vec<Projection>,
    ) -> PlaceWithOrigin {
        debug_assert!(origins.iter().all(|origin| origin.len() == projections.len() + 1));
        PlaceWithOrigin { origins, place: Place { base_ty: base_ty.store(), base, projections } }
    }

    fn push_projection(&mut self, projection: Projection, origin: ExprOrPatId) {
        self.place.projections.push(projection);
        for origin_stack in &mut self.origins {
            origin_stack.push(origin);
        }
    }
}

/// The `FakeReadCause` describes the type of pattern why a FakeRead statement exists.
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub enum FakeReadCause {
    /// A fake read injected into a match guard to ensure that the discriminants
    /// that are being matched on aren't modified while the match guard is being
    /// evaluated.
    ///
    /// At the beginning of each match guard, a fake borrow is
    /// inserted for each discriminant accessed in the entire `match` statement.
    ///
    /// Then, at the end of the match guard, a `FakeRead(ForMatchGuard)` is
    /// inserted to keep the fake borrows alive until that point.
    ///
    /// This should ensure that you cannot change the variant for an enum while
    /// you are in the midst of matching on it.
    ForMatchGuard,

    /// Fake read of the scrutinee of a `match` or destructuring `let`
    /// (i.e. `let` with non-trivial pattern).
    ///
    /// In `match x { ... }`, we generate a `FakeRead(ForMatchedPlace, x)`
    /// and insert it into the `otherwise_block` (which is supposed to be
    /// unreachable for irrefutable pattern-matches like `match` or `let`).
    ///
    /// This is necessary because `let x: !; match x {}` doesn't generate any
    /// actual read of x, so we need to generate a `FakeRead` to check that it
    /// is initialized.
    ///
    /// If the `FakeRead(ForMatchedPlace)` is being performed with a closure
    /// that doesn't capture the required upvars, the `FakeRead` within the
    /// closure is omitted entirely.
    ///
    /// To make sure that this is still sound, if a closure matches against
    /// a Place starting with an Upvar, we hoist the `FakeRead` to the
    /// definition point of the closure.
    ///
    /// If the `FakeRead` comes from being hoisted out of a closure like this,
    /// we record the `ExprId` of the closure. Otherwise, the `Option` will be `None`.
    //
    // We can use LocalDefId here since fake read statements are removed
    // before codegen in the `CleanupNonCodegenStatements` pass.
    ForMatchedPlace(Option<ExprId>),

    /// A fake read injected into a match guard to ensure that the places
    /// bound by the pattern are immutable for the duration of the match guard.
    ///
    /// Within a match guard, references are created for each place that the
    /// pattern creates a binding for — this is known as the `RefWithinGuard`
    /// version of the variables. To make sure that the references stay
    /// alive until the end of the match guard, and properly prevent the
    /// places in question from being modified, a `FakeRead(ForGuardBinding)`
    /// is inserted at the end of the match guard.
    ///
    /// For details on how these references are created, see the extensive
    /// documentation on `bind_matched_candidate_for_guard` in
    /// `rustc_mir_build`.
    ForGuardBinding,

    /// Officially, the semantics of
    ///
    /// `let pattern = <expr>;`
    ///
    /// is that `<expr>` is evaluated into a temporary and then this temporary is
    /// into the pattern.
    ///
    /// However, if we see the simple pattern `let var = <expr>`, we optimize this to
    /// evaluate `<expr>` directly into the variable `var`. This is mostly unobservable,
    /// but in some cases it can affect the borrow checker, as in #53695.
    ///
    /// Therefore, we insert a `FakeRead(ForLet)` immediately after each `let`
    /// with a trivial pattern.
    ///
    /// FIXME: `ExprUseVisitor` has an entirely different opinion on what `FakeRead(ForLet)`
    /// is supposed to mean. If it was accurate to what MIR lowering does,
    /// would it even make sense to hoist these out of closures like
    /// `ForMatchedPlace`?
    ForLet(Option<ExprId>),

    /// Currently, index expressions overloaded through the `Index` trait
    /// get lowered differently than index expressions with builtin semantics
    /// for arrays and slices — the latter will emit code to perform
    /// bound checks, and then return a MIR place that will only perform the
    /// indexing "for real" when it gets incorporated into an instruction.
    ///
    /// This is observable in the fact that the following compiles:
    ///
    /// ```
    /// fn f(x: &mut [&mut [u32]], i: usize) {
    ///     x[i][x[i].len() - 1] += 1;
    /// }
    /// ```
    ///
    /// However, we need to be careful to not let the user invalidate the
    /// bound check with an expression like
    ///
    /// `(*x)[1][{ x = y; 4}]`
    ///
    /// Here, the first bounds check would be invalidated when we evaluate the
    /// second index expression. To make sure that this doesn't happen, we
    /// create a fake borrow of `x` and hold it while we evaluate the second
    /// index.
    ///
    /// This borrow is kept alive by a `FakeRead(ForIndex)` at the end of its
    /// scope.
    ForIndex,
}

/// This trait defines the callbacks you can expect to receive when
/// employing the ExprUseVisitor.
pub(crate) trait Delegate<'db> {
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
    fn consume(&mut self, place_with_id: PlaceWithOrigin, ctx: &mut InferenceContext<'_, 'db>);

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
    fn use_cloned(&mut self, place_with_id: PlaceWithOrigin, ctx: &mut InferenceContext<'_, 'db>);

    /// The value found at `place` is being borrowed with kind `bk`.
    /// `diag_expr_id` is the id used for diagnostics (see `consume` for more details).
    fn borrow(
        &mut self,
        place_with_id: PlaceWithOrigin,
        bk: BorrowKind,
        ctx: &mut InferenceContext<'_, 'db>,
    );

    /// The value found at `place` is being copied.
    /// `diag_expr_id` is the id used for diagnostics (see `consume` for more details).
    ///
    /// If an implementation is not provided, use of a `Copy` type in a ByValue context is instead
    /// considered a use by `ImmBorrow` and `borrow` is called instead. This is because a shared
    /// borrow is the "minimum access" that would be needed to perform a copy.
    fn copy(&mut self, place_with_id: PlaceWithOrigin, ctx: &mut InferenceContext<'_, 'db>) {
        // In most cases, copying data from `x` is equivalent to doing `*&x`, so by default
        // we treat a copy of `x` as a borrow of `x`.
        self.borrow(place_with_id, BorrowKind::Immutable, ctx)
    }

    /// The path at `assignee_place` is being assigned to.
    /// `diag_expr_id` is the id used for diagnostics (see `consume` for more details).
    fn mutate(&mut self, assignee_place: PlaceWithOrigin, ctx: &mut InferenceContext<'_, 'db>);

    /// The path at `binding_place` is a binding that is being initialized.
    ///
    /// This covers cases such as `let x = 42;`
    fn bind(&mut self, binding_place: PlaceWithOrigin, ctx: &mut InferenceContext<'_, 'db>) {
        // Bindings can normally be treated as a regular assignment, so by default we
        // forward this to the mutate callback.
        self.mutate(binding_place, ctx)
    }

    /// The `place` should be a fake read because of specified `cause`.
    fn fake_read(
        &mut self,
        place_with_id: PlaceWithOrigin,
        cause: FakeReadCause,
        ctx: &mut InferenceContext<'_, 'db>,
    );
}

impl<'db, D: Delegate<'db>> Delegate<'db> for &mut D {
    fn consume(&mut self, place_with_id: PlaceWithOrigin, ctx: &mut InferenceContext<'_, 'db>) {
        (**self).consume(place_with_id, ctx)
    }

    fn use_cloned(&mut self, place_with_id: PlaceWithOrigin, ctx: &mut InferenceContext<'_, 'db>) {
        (**self).use_cloned(place_with_id, ctx)
    }

    fn borrow(
        &mut self,
        place_with_id: PlaceWithOrigin,
        bk: BorrowKind,
        ctx: &mut InferenceContext<'_, 'db>,
    ) {
        (**self).borrow(place_with_id, bk, ctx)
    }

    fn copy(&mut self, place_with_id: PlaceWithOrigin, ctx: &mut InferenceContext<'_, 'db>) {
        (**self).copy(place_with_id, ctx)
    }

    fn mutate(&mut self, assignee_place: PlaceWithOrigin, ctx: &mut InferenceContext<'_, 'db>) {
        (**self).mutate(assignee_place, ctx)
    }

    fn bind(&mut self, binding_place: PlaceWithOrigin, ctx: &mut InferenceContext<'_, 'db>) {
        (**self).bind(binding_place, ctx)
    }

    fn fake_read(
        &mut self,
        place_with_id: PlaceWithOrigin,
        cause: FakeReadCause,
        ctx: &mut InferenceContext<'_, 'db>,
    ) {
        (**self).fake_read(place_with_id, cause, ctx)
    }
}

/// A visitor that reports how each expression is being used.
///
/// See [module-level docs][self] and [`Delegate`] for details.
pub(crate) struct ExprUseVisitor<'a, 'b, 'db, D: Delegate<'db>> {
    cx: &'a mut InferenceContext<'b, 'db>,
    delegate: D,
    closure_expr: ExprId,
    upvars: UpvarsRef<'db>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PatWalkMode {
    /// `let`, `match`.
    Declaration,
    /// Destructuring assignment.
    Assignment,
}

impl<'a, 'b, 'db, D: Delegate<'db>> ExprUseVisitor<'a, 'b, 'db, D> {
    /// Creates the ExprUseVisitor, configuring it with the various options provided:
    ///
    /// - `delegate` -- who receives the callbacks
    /// - `param_env` --- parameter environment for trait lookups (esp. pertaining to `Copy`)
    /// - `typeck_results` --- typeck results for the code being analyzed
    pub(crate) fn new(
        cx: &'a mut InferenceContext<'b, 'db>,
        closure_expr: ExprId,
        upvars: UpvarsRef<'db>,
        delegate: D,
    ) -> Self {
        ExprUseVisitor { delegate, closure_expr, upvars, cx }
    }

    pub(crate) fn consume_closure_body(&mut self, params: &[PatId], body: ExprId) -> Result {
        for &param in params {
            let param_ty = self.pat_ty_adjusted(param)?;
            debug!("consume_body: param_ty = {:?}", param_ty);

            let param_place = self.cat_rvalue(param.into(), param_ty);

            self.fake_read_scrutinee(param_place.clone(), false);
            self.walk_pat(param_place, param, false, PatWalkMode::Declaration)?;
        }

        self.consume_expr(body)?;

        Ok(())
    }

    #[instrument(skip(self), level = "debug")]
    fn consume_or_copy(&mut self, place_with_id: PlaceWithOrigin) {
        if self.cx.table.type_is_copy_modulo_regions(place_with_id.place.ty()) {
            self.delegate.copy(place_with_id, self.cx);
        } else {
            self.delegate.consume(place_with_id, self.cx);
        }
    }

    #[instrument(skip(self), level = "debug")]
    pub(crate) fn consume_clone_or_copy(&mut self, place_with_id: PlaceWithOrigin) {
        // `x.use` will do one of the following
        // * if it implements `Copy`, it will be a copy
        // * if it implements `UseCloned`, it will be a call to `clone`
        // * otherwise, it is a move
        //
        // we do a conservative approximation of this, treating it as a move unless we know that it implements copy or `UseCloned`
        if self.cx.table.type_is_copy_modulo_regions(place_with_id.place.ty()) {
            self.delegate.copy(place_with_id, self.cx);
        } else if self.cx.table.type_is_use_cloned_modulo_regions(place_with_id.place.ty()) {
            self.delegate.use_cloned(place_with_id, self.cx);
        } else {
            self.delegate.consume(place_with_id, self.cx);
        }
    }

    fn consume_exprs(&mut self, exprs: &[ExprId]) -> Result {
        for &expr in exprs {
            self.consume_expr(expr)?;
        }
        Ok(())
    }

    // FIXME: It's suspicious that this is public; clippy should probably use `walk_expr`.
    #[instrument(skip(self), level = "debug")]
    pub(crate) fn consume_expr(&mut self, expr: ExprId) -> Result {
        let place_with_id = self.cat_expr(expr)?;
        self.consume_or_copy(place_with_id);
        self.walk_expr(expr)?;
        Ok(())
    }

    fn mutate_expr(&mut self, expr: ExprId) -> Result {
        let place_with_id = self.cat_expr(expr)?;
        self.delegate.mutate(place_with_id, self.cx);
        self.walk_expr(expr)?;
        Ok(())
    }

    #[instrument(skip(self), level = "debug")]
    fn borrow_expr(&mut self, expr: ExprId, bk: BorrowKind) -> Result {
        let place_with_id = self.cat_expr(expr)?;
        self.delegate.borrow(place_with_id, bk, self.cx);
        self.walk_expr(expr)?;
        Ok(())
    }

    #[instrument(skip(self), level = "debug")]
    pub(crate) fn walk_expr(&mut self, expr: ExprId) -> Result {
        self.walk_adjustment(expr)?;

        match self.cx.store[expr] {
            Expr::Path(_) => {}

            Expr::UnaryOp { op: UnaryOp::Deref, expr: base } => {
                // *base
                self.walk_expr(base)?;
            }

            Expr::Field { expr: base, .. } => {
                // base.f
                self.walk_expr(base)?;
            }

            Expr::Index { base: lhs, index: rhs } => {
                // lhs[rhs]
                self.walk_expr(lhs)?;
                self.consume_expr(rhs)?;
            }

            Expr::Call { callee, ref args } => {
                // callee(args)
                self.consume_expr(callee)?;
                self.consume_exprs(args)?;
            }

            Expr::MethodCall { receiver, ref args, .. } => {
                // callee.m(args)
                self.consume_expr(receiver)?;
                self.consume_exprs(args)?;
            }

            Expr::RecordLit { ref fields, spread, .. } => {
                self.walk_struct_expr(fields, spread)?;
            }

            Expr::Tuple { ref exprs } => {
                self.consume_exprs(exprs)?;
            }

            Expr::If {
                condition: cond_expr,
                then_branch: then_expr,
                else_branch: opt_else_expr,
            } => {
                self.consume_expr(cond_expr)?;
                self.consume_expr(then_expr)?;
                if let Some(else_expr) = opt_else_expr {
                    self.consume_expr(else_expr)?;
                }
            }

            Expr::Let { pat, expr: init } => {
                self.walk_local(init, pat, None, |this| {
                    this.borrow_expr(init, BorrowKind::Immutable)
                })?;
            }

            Expr::Match { expr: discr, ref arms } => {
                let discr_place = self.cat_expr(discr)?;
                self.fake_read_scrutinee(discr_place.clone(), true);
                self.walk_expr(discr)?;

                for arm in arms {
                    self.walk_arm(discr_place.clone(), arm)?;
                }
            }

            Expr::Array(Array::ElementList { elements: ref exprs }) => {
                self.consume_exprs(exprs)?;
            }

            Expr::Ref { expr: base, mutability: m, .. } => {
                // &base
                // make sure that the thing we are pointing out stays valid
                // for the lifetime `scope_r` of the resulting ptr:
                let bk = BorrowKind::from_hir_mutbl(m);
                self.borrow_expr(base, bk)?;
            }

            Expr::InlineAsm(ref asm) => {
                for (_, op) in &asm.operands {
                    match *op {
                        AsmOperand::In { expr, .. } => {
                            self.consume_expr(expr)?;
                        }
                        AsmOperand::Out { expr: Some(expr), .. }
                        | AsmOperand::InOut { expr, .. } => {
                            self.mutate_expr(expr)?;
                        }
                        AsmOperand::SplitInOut { in_expr, out_expr, .. } => {
                            self.consume_expr(in_expr)?;
                            if let Some(out_expr) = out_expr {
                                self.mutate_expr(out_expr)?;
                            }
                        }
                        AsmOperand::Out { expr: None, .. }
                        | AsmOperand::Const { .. }
                        | AsmOperand::Sym { .. } => {}
                        AsmOperand::Label(block) => {
                            self.walk_expr(block)?;
                        }
                    }
                }
            }

            Expr::Continue { .. }
            | Expr::Literal(..)
            | Expr::Const(..)
            | Expr::OffsetOf(..)
            | Expr::Missing
            | Expr::Underscore => {}

            Expr::Loop { body: blk, .. } => {
                self.walk_expr(blk)?;
            }

            Expr::UnaryOp { expr: lhs, .. } => {
                self.consume_expr(lhs)?;
            }

            Expr::BinaryOp {
                lhs,
                rhs,
                op: Some(BinaryOp::ArithOp(..) | BinaryOp::CmpOp(..) | BinaryOp::LogicOp(..)),
            } => {
                self.consume_expr(lhs)?;
                self.consume_expr(rhs)?;
            }

            Expr::Block { ref statements, tail, .. }
            | Expr::Unsafe { ref statements, tail, .. } => {
                for stmt in statements {
                    self.walk_stmt(stmt)?;
                }

                if let Some(tail_expr) = tail {
                    self.consume_expr(tail_expr)?;
                }
            }

            Expr::Break { expr: opt_expr, .. } | Expr::Return { expr: opt_expr } => {
                if let Some(expr) = opt_expr {
                    self.consume_expr(expr)?;
                }
            }

            Expr::Become { expr } | Expr::Await { expr } | Expr::Box { expr } => {
                self.consume_expr(expr)?;
            }

            Expr::Assignment { target, value } => {
                self.walk_expr(value)?;
                let expr_place = self.cat_expr(value)?;
                let update_guard =
                    self.cx.resolver.update_to_inner_scope(self.cx.db, self.cx.owner, expr);
                self.walk_pat(expr_place, target, false, PatWalkMode::Assignment)?;
                self.cx.resolver.reset_to_guard(update_guard);
            }

            Expr::Cast { expr: base, .. } => {
                self.consume_expr(base)?;
            }

            Expr::BinaryOp { lhs, rhs, op: None | Some(BinaryOp::Assignment { .. }) } => {
                self.consume_expr(lhs)?;
                self.consume_expr(rhs)?;
            }

            Expr::Array(Array::Repeat { initializer: base, .. }) => {
                self.consume_expr(base)?;
            }

            Expr::Closure { .. } => {
                self.walk_captures(expr);
            }

            Expr::Yield { expr: value } | Expr::Yeet { expr: value } => {
                if let Some(value) = value {
                    self.consume_expr(value)?;
                }
            }

            Expr::Range { lhs, rhs, .. } => {
                if let Some(lhs) = lhs {
                    self.consume_expr(lhs)?;
                }
                if let Some(rhs) = rhs {
                    self.consume_expr(rhs)?;
                }
            }
        }
        Ok(())
    }

    fn walk_stmt(&mut self, stmt: &Statement) -> Result {
        match *stmt {
            Statement::Let { pat, initializer: Some(expr), else_branch: els, .. } => {
                self.walk_local(expr, pat, els, |_| Ok(()))?;
            }

            Statement::Let { .. } => {}

            Statement::Item(_) => {
                // We don't visit nested items in this visitor,
                // only the fn body we were given.
            }

            Statement::Expr { expr, .. } => {
                self.consume_expr(expr)?;
            }
        }
        Ok(())
    }

    #[instrument(skip(self), level = "debug")]
    fn fake_read_scrutinee(&mut self, discr_place: PlaceWithOrigin, refutable: bool) {
        let closure_def_id = match discr_place.place.base {
            PlaceBase::Upvar { closure, var_id: _ } => Some(closure),
            _ => None,
        };

        let cause = if refutable {
            FakeReadCause::ForMatchedPlace(closure_def_id)
        } else {
            FakeReadCause::ForLet(closure_def_id)
        };

        self.delegate.fake_read(discr_place, cause, self.cx);
    }

    fn walk_local<F>(&mut self, expr: ExprId, pat: PatId, els: Option<ExprId>, mut f: F) -> Result
    where
        F: FnMut(&mut Self) -> Result,
    {
        self.walk_expr(expr)?;
        let expr_place = self.cat_expr(expr)?;
        f(self)?;
        self.fake_read_scrutinee(expr_place.clone(), els.is_some());
        self.walk_pat(expr_place, pat, false, PatWalkMode::Declaration)?;
        if let Some(els) = els {
            self.walk_expr(els)?;
        }
        Ok(())
    }

    fn walk_struct_expr(&mut self, fields: &[RecordLitField], spread: RecordSpread) -> Result {
        // Consume the expressions supplying values for each field.
        for field in fields {
            self.consume_expr(field.expr)?;
        }

        let RecordSpread::Expr(with_expr) = spread else { return Ok(()) };

        let with_place = self.cat_expr(with_expr)?;

        // Select just those fields of the `with`
        // expression that will actually be used
        match self.cx.table.structurally_resolve_type(with_place.place.ty()).kind() {
            TyKind::Adt(adt, args) if adt.is_struct() => {
                let AdtId::StructId(adt) = adt.def_id().0 else { unreachable!() };
                let adt_fields = VariantId::from(adt).fields(self.cx.db).fields();
                let adt_field_types = self.cx.db.field_types(adt.into());
                // Consume those fields of the with expression that are needed.
                for (f_index, with_field) in adt_fields.iter() {
                    let is_mentioned = fields.iter().any(|f| f.name == with_field.name);
                    if !is_mentioned {
                        let field_place = self.cat_projection(
                            with_expr.into(),
                            with_place.clone(),
                            adt_field_types[f_index].get().instantiate(self.cx.interner(), args),
                            ProjectionKind::Field {
                                field_idx: f_index.into_raw().into_u32(),
                                variant_idx: 0,
                            },
                        );
                        self.consume_or_copy(field_place);
                    }
                }
            }
            _ => {}
        }

        // walk the with expression so that complex expressions
        // are properly handled.
        self.walk_expr(with_expr)?;

        Ok(())
    }

    fn expr_adjustments(&self, expr: ExprId) -> SmallVec<[Adjustment; 5]> {
        // Due to borrowck problems, we cannot borrow the adjustments, unfortunately.
        self.cx.result.expr_adjustment(expr).unwrap_or_default().into()
    }

    /// Invoke the appropriate delegate calls for anything that gets
    /// consumed or borrowed as part of the automatic adjustment
    /// process.
    fn walk_adjustment(&mut self, expr: ExprId) -> Result {
        let adjustments = self.expr_adjustments(expr);
        let mut place_with_id = self.cat_expr_unadjusted(expr)?;
        for adjustment in &adjustments {
            debug!("walk_adjustment expr={:?} adj={:?}", expr, adjustment);
            match adjustment.kind {
                Adjust::NeverToAny | Adjust::Pointer(_) => {
                    // Creating a closure/fn-pointer or unsizing consumes
                    // the input and stores it into the resulting rvalue.
                    self.consume_or_copy(place_with_id.clone());
                }

                Adjust::Deref(None) => {}

                // Autoderefs for overloaded Deref calls in fact reference
                // their receiver. That is, if we have `(*x)` where `x`
                // is of type `Rc<T>`, then this in fact is equivalent to
                // `x.deref()`. Since `deref()` is declared with `&self`,
                // this is an autoref of `x`.
                Adjust::Deref(Some(ref deref)) => {
                    let bk = BorrowKind::from_mutbl(deref.0);
                    self.delegate.borrow(place_with_id.clone(), bk, self.cx);
                }

                Adjust::Borrow(ref autoref) => {
                    self.walk_autoref(expr, place_with_id.clone(), autoref);
                }
            }
            place_with_id = self.cat_expr_adjusted(expr, place_with_id, adjustment)?;
        }
        Ok(())
    }

    /// Walks the autoref `autoref` applied to the autoderef'd
    /// `expr`. `base_place` is `expr` represented as a place,
    /// after all relevant autoderefs have occurred.
    fn walk_autoref(&mut self, expr: ExprId, base_place: PlaceWithOrigin, autoref: &AutoBorrow) {
        debug!("walk_autoref(expr={:?} base_place={:?} autoref={:?})", expr, base_place, autoref);

        match *autoref {
            AutoBorrow::Ref(m) => {
                self.delegate.borrow(base_place, BorrowKind::from_mutbl(m.into()), self.cx);
            }

            AutoBorrow::RawPtr(m) => {
                debug!("walk_autoref: expr={:?} base_place={:?}", expr, base_place);

                self.delegate.borrow(base_place, BorrowKind::from_mutbl(m), self.cx);
            }
        }
    }

    fn walk_arm(&mut self, discr_place: PlaceWithOrigin, arm: &MatchArm) -> Result {
        self.walk_pat(discr_place, arm.pat, arm.guard.is_some(), PatWalkMode::Declaration)?;

        if let Some(e) = arm.guard {
            self.consume_expr(e)?;
        }

        self.consume_expr(arm.expr)
    }

    /// The core driver for walking a pattern
    ///
    /// This should mirror how pattern-matching gets lowered to MIR, as
    /// otherwise lowering will ICE when trying to resolve the upvars.
    ///
    /// However, it is okay to approximate it here by doing *more* accesses than
    /// the actual MIR builder will, which is useful when some checks are too
    /// cumbersome to perform here. For example, if after typeck it becomes
    /// clear that only one variant of an enum is inhabited, and therefore a
    /// read of the discriminant is not necessary, `walk_pat` will have
    /// over-approximated the necessary upvar capture granularity.
    ///
    /// Do note that discrepancies like these do still create obscure corners
    /// in the semantics of the language, and should be avoided if possible.
    #[instrument(skip(self), level = "debug")]
    fn walk_pat(
        &mut self,
        discr_place: PlaceWithOrigin,
        pat: PatId,
        has_guard: bool,
        mode: PatWalkMode,
    ) -> Result {
        self.cat_pattern(discr_place.clone(), pat, &mut |this, place, pat| {
            debug!("walk_pat: pat.kind={:?}", this.cx.store[pat]);
            let read_discriminant = {
                let place = place.clone();
                |this: &mut Self| {
                    this.delegate.borrow(place, BorrowKind::Immutable, this.cx);
                }
            };

            match this.cx.store[pat] {
                Pat::Bind { id, .. } => {
                    debug!("walk_pat: binding place={:?} pat={:?}", place, pat);
                    let bm = this.cx.result.binding_modes[pat];
                    debug!("walk_pat: pat.hir_id={:?} bm={:?}", pat, bm);

                    // pat_ty: the type of the binding being produced.
                    let pat_ty = this.node_ty(pat.into())?;
                    debug!("walk_pat: pat_ty={:?}", pat_ty);

                    if let Ok(binding_place) = this.cat_local(pat.into(), pat_ty, id) {
                        this.delegate.bind(binding_place, this.cx);
                    }

                    // Subtle: MIR desugaring introduces immutable borrows for each pattern
                    // binding when lowering pattern guards to ensure that the guard does not
                    // modify the scrutinee.
                    if has_guard {
                        read_discriminant(this);
                    }

                    // It is also a borrow or copy/move of the value being matched.
                    // In a cases of pattern like `let pat = upvar`, don't use the span
                    // of the pattern, as this just looks confusing, instead use the span
                    // of the discriminant.
                    match this.cx.result.binding_mode(pat) {
                        Some(BindingMode::Ref(m)) => {
                            let bk = BorrowKind::from_mutbl(m);
                            this.delegate.borrow(place, bk, this.cx);
                        }
                        None | Some(BindingMode::Move) => {
                            debug!("walk_pat binding consuming pat");
                            this.consume_or_copy(place);
                        }
                    }
                }
                Pat::Path(ref path) => {
                    // A `Path` pattern is just a name like `Foo`. This is either a
                    // named constant or else it refers to an ADT variant

                    let is_assoc_const = this
                        .cx
                        .result
                        .assoc_resolutions_for_pat(pat)
                        .is_some_and(|it| matches!(it.0, CandidateId::ConstId(_)));
                    let resolution = this.cx.resolver.resolve_path_in_value_ns_fully(
                        this.cx.db,
                        path,
                        this.cx.store.pat_path_hygiene(pat),
                    );
                    let is_normal_const = matches!(resolution, Some(ValueNs::ConstId(_)));
                    if mode == PatWalkMode::Assignment
                        && let Some(ValueNs::LocalBinding(local)) = resolution
                    {
                        let pat_ty = this.pat_ty(pat)?;
                        let place = this.cat_local(pat.into(), pat_ty, local)?;
                        this.delegate.mutate(place, this.cx);
                    } else if is_assoc_const || is_normal_const {
                        // Named constants have to be equated with the value
                        // being matched, so that's a read of the value being matched.
                        //
                        // FIXME: Does the MIR code skip this read when matching on a ZST?
                        // If so, we can also skip it here.
                        read_discriminant(this);
                    } else if this.is_multivariant_adt(place.place.ty()) {
                        // Otherwise, this is a struct/enum variant, and so it's
                        // only a read if we need to read the discriminant.
                        read_discriminant(this);
                    }
                }
                Pat::Lit(_) | Pat::ConstBlock(_) | Pat::Range { .. } => {
                    // When matching against a literal or range, we need to
                    // borrow the place to compare it against the pattern.
                    //
                    // Note that we do this read even if the range matches all
                    // possible values, such as 0..=u8::MAX. This is because
                    // we don't want to depend on consteval here.
                    //
                    // FIXME: What if the type being matched only has one
                    // possible value?
                    read_discriminant(this);
                }
                Pat::Record { .. } | Pat::TupleStruct { .. } => {
                    if this.is_multivariant_adt(place.place.ty()) {
                        read_discriminant(this);
                    }
                }
                Pat::Slice { prefix: ref lhs, slice: wild, suffix: ref rhs } => {
                    // We don't need to test the length if the pattern is `[..]`
                    if matches!((&**lhs, wild, &**rhs), (&[], Some(_), &[]))
                        // Arrays have a statically known size, so
                        // there is no need to read their length
                        || place.place.ty().strip_references().is_array()
                    {
                        // No read necessary
                    } else {
                        read_discriminant(this);
                    }
                }
                Pat::Expr(expr) if mode == PatWalkMode::Assignment => {
                    // Destructuring assignment.
                    this.mutate_expr(expr)?;
                }
                Pat::Or(_)
                | Pat::Box { .. }
                | Pat::Ref { .. }
                | Pat::Tuple { .. }
                | Pat::Wild
                | Pat::Missing => {
                    // If the PatKind is Or, Box, Ref, Guard, or Tuple, the relevant accesses
                    // are made later as these patterns contains subpatterns.
                    // If the PatKind is Missing, Wild or Err, any relevant accesses are made when processing
                    // the other patterns that are part of the match
                }
                Pat::Expr(_) => {}
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
    fn walk_captures(&mut self, closure_expr: ExprId) {
        fn upvar_is_local_variable(upvars: UpvarsRef<'_>, var_id: BindingId) -> bool {
            upvars.contains(var_id)
        }

        // If we have a nested closure, we want to include the fake reads present in the nested
        // closure.
        // `remove()` then re-insert and not `get()` due to borrowck errors.
        if let Some(closure_data) = self.cx.result.closures_data.remove(&closure_expr) {
            for (fake_read, cause, origins) in closure_data.fake_reads.iter() {
                match fake_read.base {
                    PlaceBase::Upvar { var_id, closure: _ } => {
                        if upvar_is_local_variable(self.upvars, var_id) {
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
                        panic!(
                            "Do not know how to get ExprId out of Rvalue and StaticItem {:?}",
                            fake_read.base
                        );
                    }
                };
                self.delegate.fake_read(
                    PlaceWithOrigin { place: fake_read.clone(), origins: origins.clone() },
                    *cause,
                    self.cx,
                );
            }

            for (var_id, min_list) in closure_data.min_captures.iter() {
                if !self.upvars.contains(*var_id) {
                    // The nested closure might be capturing the current (enclosing) closure's local variables.
                    // We check if the root variable is ever mentioned within the enclosing closure, if not
                    // then for the current body (if it's a closure) these aren't captures, we will ignore them.
                    continue;
                }
                for captured_place in min_list {
                    let place = &captured_place.place;
                    let capture_info = &captured_place.info;

                    // Mark the place to be captured by the enclosing closure
                    let place_base =
                        PlaceBase::Upvar { var_id: *var_id, closure: self.closure_expr };
                    let place_with_id = PlaceWithOrigin::new(
                        capture_info.sources.clone(),
                        place.base_ty.as_ref(),
                        place_base,
                        place.projections.clone(),
                    );

                    match capture_info.capture_kind {
                        UpvarCapture::ByValue => {
                            self.consume_or_copy(place_with_id);
                        }
                        UpvarCapture::ByUse => {
                            self.consume_clone_or_copy(place_with_id);
                        }
                        UpvarCapture::ByRef(upvar_borrow) => {
                            self.delegate.borrow(place_with_id, upvar_borrow, self.cx);
                        }
                    }
                }
            }

            self.cx.result.closures_data.insert(closure_expr, closure_data);
        }
    }

    fn error_reported_in_ty(&self, ty: Ty<'db>) -> Result {
        if ty.is_ty_error() { Err(ErrorGuaranteed) } else { Ok(()) }
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
impl<'db, D: Delegate<'db>> ExprUseVisitor<'_, '_, 'db, D> {
    fn expect_and_resolve_type(&mut self, ty: Option<Ty<'db>>) -> Result<Ty<'db>> {
        match ty {
            Some(ty) => {
                let ty = self.cx.infcx().resolve_vars_if_possible(ty);
                self.error_reported_in_ty(ty)?;
                Ok(ty)
            }
            None => Err(ErrorGuaranteed),
        }
    }

    fn node_ty(&mut self, id: ExprOrPatId) -> Result<Ty<'db>> {
        self.expect_and_resolve_type(self.cx.result.type_of_expr_or_pat(id))
    }

    fn expr_ty(&mut self, expr: ExprId) -> Result<Ty<'db>> {
        self.node_ty(expr.into())
    }

    fn pat_ty(&mut self, pat: PatId) -> Result<Ty<'db>> {
        self.node_ty(pat.into())
    }

    fn expr_ty_adjusted(&mut self, expr: ExprId) -> Result<Ty<'db>> {
        self.expect_and_resolve_type(self.cx.result.type_of_expr_with_adjust(expr))
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
    fn pat_ty_adjusted(&mut self, pat: PatId) -> Result<Ty<'db>> {
        // Check for implicit `&` types wrapping the pattern; note
        // that these are never attached to binding patterns, so
        // actually this is somewhat "disjoint" from the code below
        // that aims to account for `ref x`.
        if let Some(vec) = self.cx.result.pat_adjustments.get(&pat)
            && let Some(first_adjust) = vec.first()
        {
            debug!("pat_ty(pat={:?}) found adjustment `{:?}`", pat, first_adjust);
            return Ok(first_adjust.as_ref());
        }
        self.pat_ty_unadjusted(pat)
    }

    /// Like [`Self::pat_ty_adjusted`], but ignores implicit `&` patterns.
    fn pat_ty_unadjusted(&mut self, pat: PatId) -> Result<Ty<'db>> {
        Ok(self.cx.result.pat_ty(pat))
    }

    fn cat_expr(&mut self, expr: ExprId) -> Result<PlaceWithOrigin> {
        self.cat_expr_(expr, &self.expr_adjustments(expr))
    }

    /// This recursion helper avoids going through *too many*
    /// adjustments, since *only* non-overloaded deref recurses.
    fn cat_expr_(&mut self, expr: ExprId, adjustments: &[Adjustment]) -> Result<PlaceWithOrigin> {
        match adjustments.split_last() {
            None => self.cat_expr_unadjusted(expr),
            Some((adjustment, previous)) => {
                self.cat_expr_adjusted_with(expr, |this| this.cat_expr_(expr, previous), adjustment)
            }
        }
    }

    fn cat_expr_adjusted(
        &mut self,
        expr: ExprId,
        previous: PlaceWithOrigin,
        adjustment: &Adjustment,
    ) -> Result<PlaceWithOrigin> {
        self.cat_expr_adjusted_with(expr, |_this| Ok(previous), adjustment)
    }

    fn cat_expr_adjusted_with<F>(
        &mut self,
        expr: ExprId,
        previous: F,
        adjustment: &Adjustment,
    ) -> Result<PlaceWithOrigin>
    where
        F: FnOnce(&mut Self) -> Result<PlaceWithOrigin>,
    {
        let target = self.cx.infcx().resolve_vars_if_possible(adjustment.target.as_ref());
        match adjustment.kind {
            Adjust::Deref(overloaded) => {
                // Equivalent to *expr or something similar.
                let base = if let Some(deref) = overloaded {
                    let ref_ty = Ty::new_ref(
                        self.cx.interner(),
                        self.cx.types.regions.erased,
                        target,
                        deref.0,
                    );
                    self.cat_rvalue(expr.into(), ref_ty)
                } else {
                    previous(self)?
                };
                self.cat_deref(expr.into(), base)
            }

            Adjust::NeverToAny | Adjust::Pointer(_) | Adjust::Borrow(_) => {
                // Result is an rvalue.
                Ok(self.cat_rvalue(expr.into(), target))
            }
        }
    }

    fn cat_expr_unadjusted(&mut self, expr: ExprId) -> Result<PlaceWithOrigin> {
        let expr_ty = self.expr_ty(expr)?;
        match self.cx.store[expr] {
            Expr::UnaryOp { expr: e_base, op: UnaryOp::Deref } => {
                if self.cx.result.method_resolutions.contains_key(&expr) {
                    self.cat_overloaded_place(expr, e_base)
                } else {
                    let base = self.cat_expr(e_base)?;
                    self.cat_deref(expr.into(), base)
                }
            }

            Expr::Field { expr: base, .. } => {
                let base = self.cat_expr(base)?;
                debug!(?base);

                let field_idx = self
                    .cx
                    .result
                    .field_resolutions
                    .get(&expr)
                    .map(|field| match *field {
                        Either::Left(field) => field.local_id.into_raw().into_u32(),
                        Either::Right(tuple_field) => tuple_field.index,
                    })
                    .ok_or(ErrorGuaranteed)?;

                Ok(self.cat_projection(
                    expr.into(),
                    base,
                    expr_ty,
                    ProjectionKind::Field { field_idx, variant_idx: 0 },
                ))
            }

            Expr::Index { base, index: _ } => {
                // rustc checks if this is an overloaded index, but the check is buggy and treats any indexing
                // as overloaded, see https://rust-lang.zulipchat.com/#narrow/channel/144729-t-types/topic/.E2.9C.94.20Is.20builtin.20indexing.20any.20special.20in.20typeck.3F/near/565881390.
                // So that's what we do here.
                self.cat_overloaded_place(expr, base)
            }

            Expr::Path(ref path) => {
                let resolver_guard =
                    self.cx.resolver.update_to_inner_scope(self.cx.db, self.cx.owner, expr);
                let resolution = self.cx.resolver.resolve_path_in_value_ns_fully(
                    self.cx.db,
                    path,
                    self.cx.store.expr_path_hygiene(expr),
                );
                self.cx.resolver.reset_to_guard(resolver_guard);
                match (resolution, self.cx.result.assoc_resolutions_for_expr(expr)) {
                    (_, Some((CandidateId::FunctionId(_) | CandidateId::ConstId(_), _)))
                    | (
                        Some(
                            ValueNs::ConstId(_)
                            | ValueNs::GenericParam(_)
                            | ValueNs::FunctionId(_)
                            | ValueNs::ImplSelf(_)
                            | ValueNs::EnumVariantId(_)
                            | ValueNs::StructId(_),
                        ),
                        None,
                    ) => Ok(self.cat_rvalue(expr.into(), expr_ty)),
                    (Some(ValueNs::StaticId(_)), None) => Ok(PlaceWithOrigin::new_no_projections(
                        expr,
                        expr_ty,
                        PlaceBase::StaticItem,
                    )),
                    (Some(ValueNs::LocalBinding(var_id)), None) => {
                        self.cat_local(expr.into(), expr_ty, var_id)
                    }
                    (None, None) => Err(ErrorGuaranteed),
                }
            }

            _ => Ok(self.cat_rvalue(expr.into(), expr_ty)),
        }
    }

    fn cat_local(
        &mut self,
        id: ExprOrPatId,
        expr_ty: Ty<'db>,
        var_id: BindingId,
    ) -> Result<PlaceWithOrigin> {
        if self.upvars.contains(var_id) {
            self.cat_upvar(id, var_id)
        } else {
            Ok(PlaceWithOrigin::new_no_projections(id, expr_ty, PlaceBase::Local(var_id)))
        }
    }

    /// Categorize an upvar.
    ///
    /// Note: the actual upvar access contains invisible derefs of closure
    /// environment and upvar reference as appropriate. Only regionck cares
    /// about these dereferences, so we let it compute them as needed.
    fn cat_upvar(&mut self, hir_id: ExprOrPatId, var_id: BindingId) -> Result<PlaceWithOrigin> {
        let var_ty = self.expect_and_resolve_type(
            self.cx.result.type_of_binding.get(var_id).map(|it| it.as_ref()),
        )?;

        Ok(PlaceWithOrigin::new_no_projections(
            hir_id,
            var_ty,
            PlaceBase::Upvar { closure: self.closure_expr, var_id },
        ))
    }

    fn cat_rvalue(&self, hir_id: ExprOrPatId, expr_ty: Ty<'db>) -> PlaceWithOrigin {
        PlaceWithOrigin::new_no_projections(hir_id, expr_ty, PlaceBase::Rvalue)
    }

    fn cat_projection(
        &self,
        node: ExprOrPatId,
        mut base_place: PlaceWithOrigin,
        ty: Ty<'db>,
        kind: ProjectionKind,
    ) -> PlaceWithOrigin {
        base_place.push_projection(Projection { kind, ty: ty.store() }, node);
        base_place
    }

    fn cat_overloaded_place(&mut self, expr: ExprId, base: ExprId) -> Result<PlaceWithOrigin> {
        // Reconstruct the output assuming it's a reference with the
        // same region and mutability as the receiver. This holds for
        // `Deref(Mut)::Deref(_mut)` and `Index(Mut)::index(_mut)`.
        let place_ty = self.expr_ty(expr)?;
        let base_ty = self.expr_ty_adjusted(base)?;

        let TyKind::Ref(region, _, mutbl) = self.cx.table.structurally_resolve_type(base_ty).kind()
        else {
            return Err(ErrorGuaranteed);
        };
        let ref_ty = Ty::new_ref(self.cx.interner(), region, place_ty, mutbl);

        let base = self.cat_rvalue(expr.into(), ref_ty);
        self.cat_deref(expr.into(), base)
    }

    fn cat_deref(
        &mut self,
        node: ExprOrPatId,
        mut base_place: PlaceWithOrigin,
    ) -> Result<PlaceWithOrigin> {
        let base_curr_ty = base_place.place.ty();
        let Some(deref_ty) =
            self.cx.table.structurally_resolve_type(base_curr_ty).builtin_deref(true)
        else {
            debug!("explicit deref of non-derefable type: {:?}", base_curr_ty);
            return Err(ErrorGuaranteed);
        };
        base_place.push_projection(
            Projection { kind: ProjectionKind::Deref, ty: deref_ty.store() },
            node,
        );
        Ok(base_place)
    }

    /// Returns the variant index for an ADT used within a Struct or TupleStruct pattern
    /// Here `pat_hir_id` is the ExprId of the pattern itself.
    fn variant_index_for_adt(&self, pat_id: PatId) -> Result<(u32, VariantId)> {
        let variant = self.cx.result.variant_resolution_for_pat(pat_id).ok_or(ErrorGuaranteed)?;
        let variant_idx = match variant {
            VariantId::EnumVariantId(variant) => variant.loc(self.cx.db).index,
            VariantId::StructId(_) | VariantId::UnionId(_) => 0,
        };
        Ok((variant_idx, variant))
    }

    /// Returns the total number of fields in a tuple used within a Tuple pattern.
    /// Here `pat_hir_id` is the ExprId of the pattern itself.
    fn total_fields_in_tuple(&mut self, pat_id: PatId) -> usize {
        let ty = self.cx.result.pat_ty(pat_id);
        match self.cx.table.structurally_resolve_type(ty).kind() {
            TyKind::Tuple(args) => args.len(),
            _ => panic!("tuple pattern not applied to a tuple"),
        }
    }

    /// Here, `place` is the `PlaceWithId` being matched and pat is the pattern it
    /// is being matched against.
    ///
    /// In general, the way that this works is that we walk down the pattern,
    /// constructing a `PlaceWithId` that represents the path that will be taken
    /// to reach the value being matched.
    fn cat_pattern<F>(
        &mut self,
        mut place_with_id: PlaceWithOrigin,
        pat: PatId,
        op: &mut F,
    ) -> Result
    where
        F: FnMut(&mut Self, PlaceWithOrigin, PatId) -> Result,
    {
        // If (pattern) adjustments are active for this pattern, adjust the `PlaceWithId` correspondingly.
        // `PlaceWithId`s are constructed differently from patterns. For example, in
        //
        // ```
        // match foo {
        //     &&Some(x, ) => { ... },
        //     _ => { ... },
        // }
        // ```
        //
        // the pattern `&&Some(x,)` is represented as `Ref { Ref { TupleStruct }}`. To build the
        // corresponding `PlaceWithId` we start with the `PlaceWithId` for `foo`, and then, by traversing the
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
        let adjustments_len = self.cx.result.pat_adjustment(pat).map_or(0, |it| it.len());
        for _ in 0..adjustments_len {
            debug!("applying adjustment to place_with_id={:?}", place_with_id);
            // FIXME: We need to adjust this once we implement deref patterns (or pin ergonomics, for that matter).
            place_with_id = self.cat_deref(pat.into(), place_with_id)?;
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
        op(self, place_with_id.clone(), pat)?;

        match self.cx.store[pat] {
            Pat::Tuple { args: ref subpats, ellipsis: dots_pos } => {
                // (p1, ..., pN)
                let total_fields = self.total_fields_in_tuple(pat);

                for (i, &subpat) in subpats.iter().enumerate_and_adjust(total_fields, dots_pos) {
                    let subpat_ty = self.pat_ty_adjusted(subpat)?;
                    let projection_kind =
                        ProjectionKind::Field { field_idx: i as u32, variant_idx: 0 };
                    let sub_place = self.cat_projection(
                        pat.into(),
                        place_with_id.clone(),
                        subpat_ty,
                        projection_kind,
                    );
                    self.cat_pattern(sub_place, subpat, op)?;
                }
            }

            Pat::TupleStruct { args: ref subpats, ellipsis: dots_pos, .. } => {
                // S(p1, ..., pN)
                let (variant_index, variant) = self.variant_index_for_adt(pat)?;
                let total_fields = variant.fields(self.cx.db).len();

                for (i, &subpat) in subpats.iter().enumerate_and_adjust(total_fields, dots_pos) {
                    let subpat_ty = self.pat_ty_adjusted(subpat)?;
                    let projection_kind =
                        ProjectionKind::Field { variant_idx: variant_index, field_idx: i as u32 };
                    let sub_place = self.cat_projection(
                        pat.into(),
                        place_with_id.clone(),
                        subpat_ty,
                        projection_kind,
                    );
                    self.cat_pattern(sub_place, subpat, op)?;
                }
            }

            Pat::Record { args: ref field_pats, .. } => {
                // S { f1: p1, ..., fN: pN }

                let (variant_index, variant) = self.variant_index_for_adt(pat)?;
                let fields = variant.fields(self.cx.db);

                for fp in field_pats {
                    let field_ty = self.pat_ty_adjusted(fp.pat)?;
                    let field_index = fields.field(&fp.name).ok_or(ErrorGuaranteed)?;

                    let field_place = self.cat_projection(
                        pat.into(),
                        place_with_id.clone(),
                        field_ty,
                        ProjectionKind::Field {
                            variant_idx: variant_index,
                            field_idx: field_index.into_raw().into_u32(),
                        },
                    );
                    self.cat_pattern(field_place, fp.pat, op)?;
                }
            }

            Pat::Or(ref pats) => {
                for &pat in pats {
                    self.cat_pattern(place_with_id.clone(), pat, op)?;
                }
            }

            Pat::Bind { subpat: Some(subpat), .. } => {
                self.cat_pattern(place_with_id, subpat, op)?;
            }

            Pat::Box { inner: subpat } | Pat::Ref { pat: subpat, .. } => {
                // box p1, &p1, &mut p1. we can ignore the mutability of
                // PatKind::Ref since that information is already contained
                // in the type.
                let subplace = self.cat_deref(pat.into(), place_with_id)?;
                self.cat_pattern(subplace, subpat, op)?;
            }

            Pat::Slice { prefix: ref before, slice, suffix: ref after } => {
                let Some(element_ty) = self
                    .cx
                    .table
                    .structurally_resolve_type(place_with_id.place.ty())
                    .builtin_index()
                else {
                    debug!("explicit index of non-indexable type {:?}", place_with_id);
                    panic!("explicit index of non-indexable type");
                };
                let elt_place = self.cat_projection(
                    pat.into(),
                    place_with_id.clone(),
                    element_ty,
                    ProjectionKind::Index,
                );
                for &before_pat in before {
                    self.cat_pattern(elt_place.clone(), before_pat, op)?;
                }
                if let Some(slice_pat) = slice {
                    let slice_pat_ty = self.pat_ty_adjusted(slice_pat)?;
                    let slice_place = self.cat_projection(
                        pat.into(),
                        place_with_id,
                        slice_pat_ty,
                        ProjectionKind::Subslice,
                    );
                    self.cat_pattern(slice_place, slice_pat, op)?;
                }
                for &after_pat in after {
                    self.cat_pattern(elt_place.clone(), after_pat, op)?;
                }
            }

            Pat::Bind { subpat: None, .. }
            | Pat::Expr(..)
            | Pat::Path(_)
            | Pat::Lit(..)
            | Pat::ConstBlock(..)
            | Pat::Range { .. }
            | Pat::Missing
            | Pat::Wild => {
                // always ok
            }
        }

        Ok(())
    }

    /// Checks whether a type has multiple variants, and therefore, whether a
    /// read of the discriminant might be necessary. Note that the actual MIR
    /// builder code does a more specific check, filtering out variants that
    /// happen to be uninhabited.
    ///
    /// Here, it is not practical to perform such a check, because inhabitedness
    /// queries require typeck results, and typeck requires closure capture analysis.
    ///
    /// Moreover, the language is moving towards uninhabited variants still semantically
    /// causing a discriminant read, so we *shouldn't* perform any such check.
    ///
    /// FIXME(never_patterns): update this comment once the aforementioned MIR builder
    /// code is changed to be insensitive to inhhabitedness.
    #[instrument(skip(self), level = "debug")]
    fn is_multivariant_adt(&mut self, ty: Ty<'db>) -> bool {
        if let TyKind::Adt(def, _) = self.cx.table.structurally_resolve_type(ty).kind() {
            // Note that if a non-exhaustive SingleVariant is defined in another crate, we need
            // to assume that more cases will be added to the variant in the future. This mean
            // that we should handle non-exhaustive SingleVariant the same way we would handle
            // a MultiVariant.
            match def.def_id().0 {
                AdtId::StructId(_) | AdtId::UnionId(_) => false,
                AdtId::EnumId(did) => {
                    let has_foreign_non_exhaustive = || {
                        AttrFlags::query(self.cx.db, did.into()).contains(AttrFlags::NON_EXHAUSTIVE)
                            && did.krate(self.cx.db) != self.cx.krate()
                    };
                    did.enum_variants(self.cx.db).variants.len() > 1 || has_foreign_non_exhaustive()
                }
            }
        } else {
            false
        }
    }
}
