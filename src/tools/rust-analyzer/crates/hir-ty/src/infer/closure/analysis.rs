//! ### Inferring borrow kinds for upvars
//!
//! Whenever there is a closure expression, we need to determine how each
//! upvar is used. We do this by initially assigning each upvar an
//! immutable "borrow kind" (see `BorrowKind` for details) and then
//! "escalating" the kind as needed. The borrow kind proceeds according to
//! the following lattice:
//! ```ignore (not-rust)
//! ty::ImmBorrow -> ty::UniqueImmBorrow -> ty::MutBorrow
//! ```
//! So, for example, if we see an assignment `x = 5` to an upvar `x`, we
//! will promote its borrow kind to mutable borrow. If we see an `&mut x`
//! we'll do the same. Naturally, this applies not just to the upvar, but
//! to everything owned by `x`, so the result is the same for something
//! like `x.f = 5` and so on (presuming `x` is not a borrowed pointer to a
//! struct). These adjustments are performed in
//! `adjust_for_non_move_closure` (you can trace backwards through the code
//! from there).
//!
//! The fact that we are inferring borrow kinds as we go results in a
//! semi-hacky interaction with the way `ExprUseVisitor` is computing
//! `Place`s. In particular, it will query the current borrow kind as it
//! goes, and we'll return the *current* value, but this may get
//! adjusted later. Therefore, in this module, we generally ignore the
//! borrow kind (and derived mutabilities) that `ExprUseVisitor` returns
//! within `Place`s, since they may be inaccurate. (Another option
//! would be to use a unification scheme, where instead of returning a
//! concrete borrow kind like `ty::ImmBorrow`, we return a
//! `ty::InferBorrow(upvar_id)` or something like that, but this would
//! then mean that all later passes would have to check for these figments
//! and report an error, and it just seems like more mess in the end.)

use std::{iter, mem};

use hir_def::{
    expr_store::ExpressionStore,
    hir::{
        BindingAnnotation, BindingId, CaptureBy, CoroutineSource, Expr, ExprId, ExprOrPatId, Pat,
        PatId, Statement,
    },
    resolver::ValueNs,
};
use macros::{TypeFoldable, TypeVisitable};
use rustc_abi::ExternAbi;
use rustc_ast_ir::Mutability;
use rustc_hash::{FxBuildHasher, FxHashMap};
use rustc_type_ir::{
    BoundVar, ClosureKind,
    inherent::{AdtDef as _, GenericArgs as _, IntoKind as _, Ty as _},
};
use smallvec::{SmallVec, smallvec};
use span::Edition;
use tracing::{debug, instrument};

use crate::{
    Span,
    infer::{
        CaptureInfo, CaptureSourceStack, CapturedPlace, InferenceContext, UpvarCapture,
        closure::analysis::expr_use_visitor::{
            self as euv, FakeReadCause, Place, PlaceBase, PlaceWithOrigin, Projection,
            ProjectionKind,
        },
    },
    next_solver::{
        Binder, BoundRegion, BoundRegionKind, DbInterner, GenericArgs, Region, Ty, TyKind,
        abi::Safety, infer::traits::ObligationCause, normalize,
    },
    upvars::{Upvars, UpvarsRef},
};

pub(crate) mod expr_use_visitor;

#[derive(Debug, Copy, Clone, TypeVisitable, TypeFoldable)]
enum UpvarArgs<'db> {
    Closure(GenericArgs<'db>),
    Coroutine(GenericArgs<'db>),
    CoroutineClosure(GenericArgs<'db>),
}

impl<'db> UpvarArgs<'db> {
    #[inline]
    fn tupled_upvars_ty(self) -> Ty<'db> {
        match self {
            UpvarArgs::Closure(args) => args.as_closure().tupled_upvars_ty(),
            UpvarArgs::Coroutine(args) => args.as_coroutine().tupled_upvars_ty(),
            UpvarArgs::CoroutineClosure(args) => args.as_coroutine_closure().tupled_upvars_ty(),
        }
    }
}

#[derive(Eq, Clone, PartialEq, Debug, Copy, Hash)]
pub enum BorrowKind {
    /// Data must be immutable and is aliasable.
    Immutable,

    /// Data must be immutable but not aliasable. This kind of borrow
    /// cannot currently be expressed by the user and is used only in
    /// implicit closure bindings. It is needed when the closure
    /// is borrowing or mutating a mutable referent, e.g.:
    ///
    /// ```
    /// let mut z = 3;
    /// let x: &mut isize = &mut z;
    /// let y = || *x += 5;
    /// ```
    ///
    /// If we were to try to translate this closure into a more explicit
    /// form, we'd encounter an error with the code as written:
    ///
    /// ```compile_fail,E0594
    /// struct Env<'a> { x: &'a &'a mut isize }
    /// let mut z = 3;
    /// let x: &mut isize = &mut z;
    /// let y = (&mut Env { x: &x }, fn_ptr);  // Closure is pair of env and fn
    /// fn fn_ptr(env: &mut Env) { **env.x += 5; }
    /// ```
    ///
    /// This is then illegal because you cannot mutate a `&mut` found
    /// in an aliasable location. To solve, you'd have to translate with
    /// an `&mut` borrow:
    ///
    /// ```compile_fail,E0596
    /// struct Env<'a> { x: &'a mut &'a mut isize }
    /// let mut z = 3;
    /// let x: &mut isize = &mut z;
    /// let y = (&mut Env { x: &mut x }, fn_ptr); // changed from &x to &mut x
    /// fn fn_ptr(env: &mut Env) { **env.x += 5; }
    /// ```
    ///
    /// Now the assignment to `**env.x` is legal, but creating a
    /// mutable pointer to `x` is not because `x` is not mutable. We
    /// could fix this by declaring `x` as `let mut x`. This is ok in
    /// user code, if awkward, but extra weird for closures, since the
    /// borrow is hidden.
    ///
    /// So we introduce a "unique imm" borrow -- the referent is
    /// immutable, but not aliasable. This solves the problem. For
    /// simplicity, we don't give users the way to express this
    /// borrow, it's just used when translating closures.
    ///
    /// FIXME: Rename this to indicate the borrow is actually not immutable.
    UniqueImmutable,

    /// Data is mutable and not aliasable.
    Mutable,
}

impl BorrowKind {
    pub fn from_hir_mutbl(m: hir_def::hir::type_ref::Mutability) -> BorrowKind {
        match m {
            hir_def::hir::type_ref::Mutability::Mut => BorrowKind::Mutable,
            hir_def::hir::type_ref::Mutability::Shared => BorrowKind::Immutable,
        }
    }

    pub fn from_mutbl(m: Mutability) -> BorrowKind {
        match m {
            Mutability::Mut => BorrowKind::Mutable,
            Mutability::Not => BorrowKind::Immutable,
        }
    }

    /// Returns a mutability `m` such that an `&m T` pointer could be used to obtain this borrow
    /// kind. Because borrow kinds are richer than mutabilities, we sometimes have to pick a
    /// mutability that is stronger than necessary so that it at least *would permit* the borrow in
    /// question.
    pub fn to_mutbl_lossy(self) -> Mutability {
        match self {
            BorrowKind::Mutable => Mutability::Mut,
            BorrowKind::Immutable => Mutability::Not,

            // We have no type corresponding to a unique imm borrow, so
            // use `&mut`. It gives all the capabilities of a `&uniq`
            // and hence is a safe "over approximation".
            BorrowKind::UniqueImmutable => Mutability::Mut,
        }
    }
}

/// Describe the relationship between the paths of two places
/// eg:
/// - `foo` is ancestor of `foo.bar.baz`
/// - `foo.bar.baz` is an descendant of `foo.bar`
/// - `foo.bar` and `foo.baz` are divergent
enum PlaceAncestryRelation {
    Ancestor,
    Descendant,
    SamePlace,
    Divergent,
}

/// Intermediate format to store a captured `Place` and associated `CaptureInfo`
/// during capture analysis. Information in this map feeds into the minimum capture
/// analysis pass.
type InferredCaptureInformation = Vec<(Place, CaptureInfo)>;

impl<'a, 'db> InferenceContext<'a, 'db> {
    pub(crate) fn closure_analyze(&mut self) {
        let upvars = crate::upvars::upvars_mentioned(self.db, self.store_owner)
            .unwrap_or(const { &FxHashMap::with_hasher(FxBuildHasher) });
        for root_expr in self.store.expr_roots() {
            self.analyze_closures_in_expr(root_expr, upvars);
        }

        // it's our job to process these.
        assert!(self.deferred_call_resolutions.is_empty());
    }

    fn analyze_closures_in_expr(&mut self, expr: ExprId, upvars: &'db FxHashMap<ExprId, Upvars>) {
        self.store.walk_child_exprs(expr, |expr| self.analyze_closures_in_expr(expr, upvars));

        match &self.store[expr] {
            Expr::Closure { args, body, closure_kind, capture_by, .. } => {
                self.analyze_closure(
                    expr,
                    args,
                    *body,
                    *capture_by,
                    *closure_kind,
                    upvars.get(&expr).map(|upvars| upvars.as_ref()).unwrap_or_default(),
                );
            }
            _ => {}
        }
    }

    /// Analysis starting point.
    #[instrument(skip(self, body), level = "debug")]
    fn analyze_closure(
        &mut self,
        closure_expr_id: ExprId,
        params: &[PatId],
        body: ExprId,
        mut capture_clause: CaptureBy,
        closure_kind: hir_def::hir::ClosureKind,
        upvars: UpvarsRef<'db>,
    ) {
        // Extract the type of the closure.
        let ty = self.expr_ty(closure_expr_id);
        let (args, infer_kind) = match ty.kind() {
            TyKind::Closure(_def_id, args) => {
                (UpvarArgs::Closure(args), self.infcx().closure_kind(ty).is_none())
            }
            TyKind::CoroutineClosure(_def_id, args) => {
                (UpvarArgs::CoroutineClosure(args), self.infcx().closure_kind(ty).is_none())
            }
            TyKind::Coroutine(_def_id, args) => (UpvarArgs::Coroutine(args), false),
            TyKind::Error(_) => {
                // #51714: skip analysis when we have already encountered type errors
                return;
            }
            _ => {
                panic!("type of closure expr {:?} is not a closure {:?}", closure_expr_id, ty);
            }
        };
        let args = self.infcx().resolve_vars_if_possible(args);

        let mut delegate = InferBorrowKind {
            closure_def_id: closure_expr_id,
            capture_information: Default::default(),
            fake_reads: Default::default(),
        };

        let _ = euv::ExprUseVisitor::new(self, closure_expr_id, upvars, &mut delegate)
            .consume_closure_body(params, body);

        // There are several curious situations with coroutine-closures where
        // analysis is too aggressive with borrows when the coroutine-closure is
        // marked `move`. Specifically:
        //
        // 1. If the coroutine-closure was inferred to be `FnOnce` during signature
        // inference, then it's still possible that we try to borrow upvars from
        // the coroutine-closure because they are not used by the coroutine body
        // in a way that forces a move. See the test:
        // `async-await/async-closures/force-move-due-to-inferred-kind.rs`.
        //
        // 2. If the coroutine-closure is forced to be `FnOnce` due to the way it
        // uses its upvars (e.g. it consumes a non-copy value), but not *all* upvars
        // would force the closure to `FnOnce`.
        // See the test: `async-await/async-closures/force-move-due-to-actually-fnonce.rs`.
        //
        // This would lead to an impossible to satisfy situation, since `AsyncFnOnce`
        // coroutine bodies can't borrow from their parent closure. To fix this,
        // we force the inner coroutine to also be `move`. This only matters for
        // coroutine-closures that are `move` since otherwise they themselves will
        // be borrowing from the outer environment, so there's no self-borrows occurring.
        if let UpvarArgs::Coroutine(..) = args
            && let hir_def::hir::ClosureKind::Coroutine { source: CoroutineSource::Closure, .. } =
                closure_kind
            && let parent_hir_id = ExpressionStore::closure_for_coroutine(closure_expr_id)
            && let parent_ty = self.result.expr_ty(parent_hir_id)
            && let Expr::Closure { capture_by: CaptureBy::Value, .. } = self.store[parent_hir_id]
        {
            // (1.) Closure signature inference forced this closure to `FnOnce`.
            if let Some(ClosureKind::FnOnce) = self.infcx().closure_kind(parent_ty) {
                capture_clause = CaptureBy::Value;
            }
            // (2.) The way that the closure uses its upvars means it's `FnOnce`.
            else if self.coroutine_body_consumes_upvars(closure_expr_id, body, upvars) {
                capture_clause = CaptureBy::Value;
            }
        }

        // As noted in `lower_coroutine_body_with_moved_arguments`, we default the capture mode
        // to `ByRef` for the `async {}` block internal to async fns/closure. This means
        // that we would *not* be moving all of the parameters into the async block in all cases.
        // For example, when one of the arguments is `Copy`, we turn a consuming use into a copy of
        // a reference, so for `async fn x(t: i32) {}`, we'd only take a reference to `t`.
        //
        // We force all of these arguments to be captured by move before we do expr use analysis.
        //
        // FIXME(async_closures): This could be cleaned up. It's a bit janky that we're just
        // moving all of the `LocalSource::AsyncFn` locals here.
        if let hir_def::hir::ClosureKind::Coroutine {
            source: CoroutineSource::Fn | CoroutineSource::Closure,
            ..
        } = closure_kind
        {
            let Expr::Block { statements, .. } = &self.store[body] else {
                panic!();
            };
            for stmt in statements {
                let Statement::Let { pat, initializer: Some(init), .. } = *stmt else {
                    panic!();
                };
                let Pat::Bind { .. } = self.store[pat] else {
                    // Complex pattern, skip the non-upvar local.
                    continue;
                };
                let Expr::Path(path) = &self.store[init] else {
                    panic!();
                };
                let update_guard =
                    self.resolver.update_to_inner_scope(self.db, self.store_owner, init);
                let Some(ValueNs::LocalBinding(local_id)) =
                    self.resolver.resolve_path_in_value_ns_fully(
                        self.db,
                        path,
                        self.store.expr_path_hygiene(init),
                    )
                else {
                    panic!();
                };
                self.resolver.reset_to_guard(update_guard);
                let place = self.place_for_root_variable(closure_expr_id, local_id);
                delegate.capture_information.push((
                    place,
                    CaptureInfo {
                        sources: smallvec![CaptureSourceStack::from_single(init.into())],
                        capture_kind: UpvarCapture::ByValue,
                    },
                ));
            }
        }

        debug!(
            "For closure={:?}, capture_information={:#?}",
            closure_expr_id, delegate.capture_information
        );

        let (capture_information, closure_kind, _origin) = self
            .process_collected_capture_information(capture_clause, &delegate.capture_information);

        self.compute_min_captures(closure_expr_id, capture_information);

        // We now fake capture information for all variables that are mentioned within the closure
        // We do this after handling migrations so that min_captures computes before
        if !enable_precise_capture(self.edition) {
            let mut capture_information: InferredCaptureInformation = Default::default();

            for var_hir_id in upvars.iter() {
                let place = Place {
                    base_ty: self.result.binding_ty(var_hir_id).store(),
                    base: PlaceBase::Upvar { closure: closure_expr_id, var_id: var_hir_id },
                    projections: Vec::new(),
                };

                debug!("seed place {:?}", place);

                let capture_kind = self.init_capture_kind_for_place(&place, capture_clause);
                let fake_info = CaptureInfo { sources: SmallVec::new(), capture_kind };

                capture_information.push((place, fake_info));
            }

            // This will update the min captures based on this new fake information.
            self.compute_min_captures(closure_expr_id, capture_information);
        }

        if infer_kind {
            // Unify the (as yet unbound) type variable in the closure
            // args with the kind we inferred.
            let closure_kind_ty = match args {
                UpvarArgs::Closure(args) => args.as_closure().kind_ty(),
                UpvarArgs::CoroutineClosure(args) => args.as_coroutine_closure().kind_ty(),
                UpvarArgs::Coroutine(_) => unreachable!("coroutines don't have an inferred kind"),
            };
            _ = self.demand_eqtype(
                closure_expr_id.into(),
                Ty::from_closure_kind(self.interner(), closure_kind),
                closure_kind_ty,
            );
        }

        // For coroutine-closures, we additionally must compute the
        // `coroutine_captures_by_ref_ty` type, which is used to generate the by-ref
        // version of the coroutine-closure's output coroutine.
        if let UpvarArgs::CoroutineClosure(args) = args {
            let closure_env_region: Region<'_> = Region::new_bound(
                self.interner(),
                rustc_type_ir::INNERMOST,
                BoundRegion { var: BoundVar::ZERO, kind: BoundRegionKind::ClosureEnv },
            );

            let num_args = args
                .as_coroutine_closure()
                .coroutine_closure_sig()
                .skip_binder()
                .tupled_inputs_ty
                .tuple_fields()
                .len();

            let tupled_upvars_ty_for_borrow = Ty::new_tup_from_iter(
                self.interner(),
                analyze_coroutine_closure_captures(
                    self.closure_min_captures_flattened(closure_expr_id),
                    self.closure_min_captures_flattened(ExpressionStore::coroutine_for_closure(
                        closure_expr_id,
                    ))
                    // Skip the captures that are just moving the closure's args
                    // into the coroutine. These are always by move, and we append
                    // those later in the `CoroutineClosureSignature` helper functions.
                    .skip(num_args),
                    |(_, parent_capture), (_, child_capture)| {
                        // This is subtle. See documentation on function.
                        let needs_ref = should_reborrow_from_env_of_parent_coroutine_closure(
                            parent_capture,
                            child_capture,
                        );

                        let upvar_ty = child_capture.place.ty();
                        let capture = child_capture.info.capture_kind;
                        // Not all upvars are captured by ref, so use
                        // `apply_capture_kind_on_capture_ty` to ensure that we
                        // compute the right captured type.
                        apply_capture_kind_on_capture_ty(
                            self.interner(),
                            upvar_ty,
                            capture,
                            if needs_ref { closure_env_region } else { self.types.regions.erased },
                        )
                    },
                ),
            );
            let coroutine_captures_by_ref_ty = Ty::new_fn_ptr(
                self.interner(),
                Binder::bind_with_vars(
                    self.interner().mk_fn_sig(
                        [],
                        tupled_upvars_ty_for_borrow,
                        false,
                        Safety::Safe,
                        ExternAbi::Rust,
                    ),
                    self.types.coroutine_captures_by_ref_bound_var_kinds,
                ),
            );
            _ = self.demand_eqtype(
                closure_expr_id.into(),
                args.as_coroutine_closure().coroutine_captures_by_ref_ty(),
                coroutine_captures_by_ref_ty,
            );

            // Additionally, we can now constrain the coroutine's kind type.
            //
            // We only do this if `infer_kind`, because if we have constrained
            // the kind from closure signature inference, the kind inferred
            // for the inner coroutine may actually be more restrictive.
            if infer_kind {
                let TyKind::Coroutine(_, coroutine_args) = self.result.expr_ty(body).kind() else {
                    panic!();
                };
                _ = self.demand_eqtype(
                    closure_expr_id.into(),
                    coroutine_args.as_coroutine().kind_ty(),
                    Ty::from_coroutine_closure_kind(self.interner(), closure_kind),
                );
            }
        }

        // Now that we've analyzed the closure, we know how each
        // variable is borrowed, and we know what traits the closure
        // implements (Fn vs FnMut etc). We now have some updates to do
        // with that information.
        //
        // Note that no closure type C may have an upvar of type C
        // (though it may reference itself via a trait object). This
        // results from the desugaring of closures to a struct like
        // `Foo<..., UV0...UVn>`. If one of those upvars referenced
        // C, then the type would have infinite size (and the
        // inference algorithm will reject it).

        // Equate the type variables for the upvars with the actual types.
        let final_upvar_tys = self.final_upvar_tys(closure_expr_id);
        debug!(?closure_expr_id, ?args, ?final_upvar_tys);

        // Build a tuple (U0..Un) of the final upvar types U0..Un
        // and unify the upvar tuple type in the closure with it:
        let final_tupled_upvars_type = Ty::new_tup(self.interner(), &final_upvar_tys);
        _ = self.demand_suptype(
            closure_expr_id.into(),
            args.tupled_upvars_ty(),
            final_tupled_upvars_type,
        );

        let fake_reads = delegate.fake_reads;

        self.result.closures_data.entry(closure_expr_id).or_default().fake_reads =
            fake_reads.into_boxed_slice();

        // If we are also inferred the closure kind here,
        // process any deferred resolutions.
        let deferred_call_resolutions = self.remove_deferred_call_resolutions(closure_expr_id);
        for deferred_call_resolution in deferred_call_resolutions {
            deferred_call_resolution.resolve(self);
        }
    }

    /// Determines whether the body of the coroutine uses its upvars in a way that
    /// consumes (i.e. moves) the value, which would force the coroutine to `FnOnce`.
    /// In a more detailed comment above, we care whether this happens, since if
    /// this happens, we want to force the coroutine to move all of the upvars it
    /// would've borrowed from the parent coroutine-closure.
    ///
    /// This only really makes sense to be called on the child coroutine of a
    /// coroutine-closure.
    fn coroutine_body_consumes_upvars(
        &mut self,
        coroutine_def_id: ExprId,
        body: ExprId,
        upvars: UpvarsRef<'db>,
    ) -> bool {
        let mut delegate = InferBorrowKind {
            closure_def_id: coroutine_def_id,
            capture_information: Default::default(),
            fake_reads: Default::default(),
        };

        let _ = euv::ExprUseVisitor::new(self, coroutine_def_id, upvars, &mut delegate)
            .consume_expr(body);

        let (_, kind, _) = self
            .process_collected_capture_information(CaptureBy::Ref, &delegate.capture_information);

        matches!(kind, ClosureKind::FnOnce)
    }

    // Returns a list of `Ty`s for each upvar.
    fn final_upvar_tys(&self, closure_id: ExprId) -> Vec<Ty<'db>> {
        self.closure_min_captures_flattened(closure_id)
            .map(|captured_place| {
                let upvar_ty = captured_place.place.ty();
                let capture = captured_place.info.capture_kind;

                debug!(?captured_place.place, ?upvar_ty, ?capture, ?captured_place.mutability);

                apply_capture_kind_on_capture_ty(
                    self.interner(),
                    upvar_ty,
                    capture,
                    self.types.regions.erased,
                )
            })
            .collect()
    }

    /// Adjusts the closure capture information to ensure that the operations aren't unsafe,
    /// and that the path can be captured with required capture kind (depending on use in closure,
    /// move closure etc.)
    ///
    /// Returns the set of adjusted information along with the inferred closure kind and span
    /// associated with the closure kind inference.
    ///
    /// Note that we *always* infer a minimal kind, even if
    /// we don't always *use* that in the final result (i.e., sometimes
    /// we've taken the closure kind from the expectations instead, and
    /// for coroutines we don't even implement the closure traits
    /// really).
    ///
    /// If we inferred that the closure needs to be FnMut/FnOnce, last element of the returned tuple
    /// contains a `Some()` with the `Place` that caused us to do so.
    fn process_collected_capture_information(
        &mut self,
        capture_clause: CaptureBy,
        capture_information: &InferredCaptureInformation,
    ) -> (InferredCaptureInformation, ClosureKind, Option<Place>) {
        let mut closure_kind = ClosureKind::LATTICE_BOTTOM;
        let mut origin: Option<Place> = None;

        let processed = capture_information
            .iter()
            .cloned()
            .map(|(place, mut capture_info)| {
                // Apply rules for safety before inferring closure kind
                let place = restrict_capture_precision(place, &mut capture_info);

                let place = truncate_capture_for_optimization(place, &mut capture_info);

                let updated = match capture_info.capture_kind {
                    UpvarCapture::ByValue => match closure_kind {
                        ClosureKind::Fn | ClosureKind::FnMut => {
                            (ClosureKind::FnOnce, Some(place.clone()))
                        }
                        // If closure is already FnOnce, don't update
                        ClosureKind::FnOnce => (closure_kind, origin.take()),
                    },

                    UpvarCapture::ByRef(BorrowKind::Mutable | BorrowKind::UniqueImmutable) => {
                        match closure_kind {
                            ClosureKind::Fn => (ClosureKind::FnMut, Some(place.clone())),
                            // Don't update the origin
                            ClosureKind::FnMut | ClosureKind::FnOnce => {
                                (closure_kind, origin.take())
                            }
                        }
                    }

                    _ => (closure_kind, origin.take()),
                };

                closure_kind = updated.0;
                origin = updated.1;

                let place = match capture_clause {
                    CaptureBy::Value => adjust_for_move_closure(place, &mut capture_info),
                    CaptureBy::Ref => adjust_for_non_move_closure(place, &mut capture_info),
                };

                // This restriction needs to be applied after we have handled adjustments for `move`
                // closures. We want to make sure any adjustment that might make us move the place into
                // the closure gets handled.
                let place = restrict_precision_for_drop_types(self, place, &mut capture_info);

                (place, capture_info)
            })
            .collect();

        (processed, closure_kind, origin)
    }

    /// Analyzes the information collected by `InferBorrowKind` to compute the min number of
    /// Places (and corresponding capture kind) that we need to keep track of to support all
    /// the required captured paths.
    ///
    ///
    /// Note: If this function is called multiple times for the same closure, it will update
    ///       the existing min_capture map that is stored in TypeckResults.
    ///
    /// Eg:
    /// ```
    /// #[derive(Debug)]
    /// struct Point { x: i32, y: i32 }
    ///
    /// let s = String::from("s");  // hir_id_s
    /// let mut p = Point { x: 2, y: -2 }; // his_id_p
    /// let c = || {
    ///        println!("{s:?}");  // L1
    ///        p.x += 10;  // L2
    ///        println!("{}" , p.y); // L3
    ///        println!("{p:?}"); // L4
    ///        drop(s);   // L5
    /// };
    /// ```
    /// and let hir_id_L1..5 be the expressions pointing to use of a captured variable on
    /// the lines L1..5 respectively.
    ///
    /// InferBorrowKind results in a structure like this:
    ///
    /// ```ignore (illustrative)
    /// {
    ///       Place(base: hir_id_s, projections: [], ....) -> {
    ///                                                            capture_kind_expr: hir_id_L5,
    ///                                                            path_expr_id: hir_id_L5,
    ///                                                            capture_kind: ByValue
    ///                                                       },
    ///       Place(base: hir_id_p, projections: [Field(0, 0)], ...) -> {
    ///                                                                     capture_kind_expr: hir_id_L2,
    ///                                                                     path_expr_id: hir_id_L2,
    ///                                                                     capture_kind: ByValue
    ///                                                                 },
    ///       Place(base: hir_id_p, projections: [Field(1, 0)], ...) -> {
    ///                                                                     capture_kind_expr: hir_id_L3,
    ///                                                                     path_expr_id: hir_id_L3,
    ///                                                                     capture_kind: ByValue
    ///                                                                 },
    ///       Place(base: hir_id_p, projections: [], ...) -> {
    ///                                                          capture_kind_expr: hir_id_L4,
    ///                                                          path_expr_id: hir_id_L4,
    ///                                                          capture_kind: ByValue
    ///                                                      },
    /// }
    /// ```
    ///
    /// After the min capture analysis, we get:
    /// ```ignore (illustrative)
    /// {
    ///       hir_id_s -> [
    ///            Place(base: hir_id_s, projections: [], ....) -> {
    ///                                                                capture_kind_expr: hir_id_L5,
    ///                                                                path_expr_id: hir_id_L5,
    ///                                                                capture_kind: ByValue
    ///                                                            },
    ///       ],
    ///       hir_id_p -> [
    ///            Place(base: hir_id_p, projections: [], ...) -> {
    ///                                                               capture_kind_expr: hir_id_L2,
    ///                                                               path_expr_id: hir_id_L4,
    ///                                                               capture_kind: ByValue
    ///                                                           },
    ///       ],
    /// }
    /// ```
    #[instrument(level = "debug", skip(self))]
    fn compute_min_captures(
        &mut self,
        closure_def_id: ExprId,
        capture_information: InferredCaptureInformation,
    ) {
        if capture_information.is_empty() {
            return;
        }

        let mut closure_data =
            self.result.closures_data.remove(&closure_def_id).unwrap_or_default();
        let root_var_min_capture_list = &mut closure_data.min_captures;
        let mut dedup_sources_scratch = FxHashMap::default();

        for (mut place, capture_info) in capture_information.into_iter() {
            let var_hir_id = match place.base {
                PlaceBase::Upvar { var_id, .. } => var_id,
                base => panic!("Expected upvar, found={:?}", base),
            };

            let Some(min_cap_list) = root_var_min_capture_list.get_mut(&var_hir_id) else {
                let mutability = self.determine_capture_mutability(closure_def_id, &place);
                let min_cap_list = vec![CapturedPlace { place, info: capture_info, mutability }];
                root_var_min_capture_list.insert(var_hir_id, min_cap_list);
                continue;
            };

            // Go through each entry in the current list of min_captures
            // - if ancestor is found, update its capture kind to account for current place's
            // capture information.
            //
            // - if descendant is found, remove it from the list, and update the current place's
            // capture information to account for the descendant's capture kind.
            //
            // We can never be in a case where the list contains both an ancestor and a descendant
            // Also there can only be ancestor but in case of descendants there might be
            // multiple.

            let mut descendant_found = false;
            let mut updated_capture_info = capture_info;
            min_cap_list.retain(|possible_descendant| {
                match determine_place_ancestry_relation(&place, &possible_descendant.place) {
                    // current place is ancestor of possible_descendant
                    PlaceAncestryRelation::Ancestor => {
                        descendant_found = true;

                        let mut possible_descendant = possible_descendant.clone();

                        // Truncate the descendant (already in min_captures) to be same as the ancestor to handle any
                        // possible change in capture mode.
                        truncate_place_to_len_and_update_capture_kind(
                            &mut possible_descendant.place,
                            &mut possible_descendant.info,
                            place.projections.len(),
                        );

                        let backup_path_sources = determine_capture_sources(
                            &mut updated_capture_info,
                            &mut possible_descendant.info,
                            &mut dedup_sources_scratch,
                        );
                        determine_capture_info(
                            &mut updated_capture_info,
                            &mut possible_descendant.info,
                        );

                        // we need to keep the ancestor's `path_expr_id`
                        updated_capture_info.sources = backup_path_sources;
                        false
                    }

                    _ => true,
                }
            });

            let mut ancestor_found = false;
            if !descendant_found {
                for possible_ancestor in min_cap_list.iter_mut() {
                    match determine_place_ancestry_relation(&place, &possible_ancestor.place) {
                        PlaceAncestryRelation::SamePlace => {
                            ancestor_found = true;
                            let backup_path_sources = determine_capture_sources(
                                &mut updated_capture_info,
                                &mut possible_ancestor.info,
                                &mut dedup_sources_scratch,
                            );
                            determine_capture_info(
                                &mut possible_ancestor.info,
                                &mut updated_capture_info,
                            );
                            possible_ancestor.info.sources = backup_path_sources;

                            // Only one related place will be in the list.
                            break;
                        }
                        // current place is descendant of possible_ancestor
                        PlaceAncestryRelation::Descendant => {
                            ancestor_found = true;

                            // Truncate the descendant (current place) to be same as the ancestor to handle any
                            // possible change in capture mode.
                            truncate_place_to_len_and_update_capture_kind(
                                &mut place,
                                &mut updated_capture_info,
                                possible_ancestor.place.projections.len(),
                            );

                            let backup_path_sources = determine_capture_sources(
                                &mut updated_capture_info,
                                &mut possible_ancestor.info,
                                &mut dedup_sources_scratch,
                            );
                            determine_capture_info(
                                &mut possible_ancestor.info,
                                &mut updated_capture_info,
                            );

                            // we need to keep the ancestor's `sources`
                            possible_ancestor.info.sources = backup_path_sources;

                            // Only one related place will be in the list.
                            break;
                        }
                        _ => {}
                    }
                }
            }

            // Only need to insert when we don't have an ancestor in the existing min capture list
            if !ancestor_found {
                let mutability = self.determine_capture_mutability(closure_def_id, &place);
                let captured_place =
                    CapturedPlace { place, info: updated_capture_info, mutability };
                min_cap_list.push(captured_place);
            }
        }

        debug!(
            "For closure={:?}, min_captures before sorting={:?}",
            closure_def_id, root_var_min_capture_list
        );

        // Now that we have the minimized list of captures, sort the captures by field id.
        // This causes the closure to capture the upvars in the same order as the fields are
        // declared which is also the drop order. Thus, in situations where we capture all the
        // fields of some type, the observable drop order will remain the same as it previously
        // was even though we're dropping each capture individually.
        // See https://github.com/rust-lang/project-rfc-2229/issues/42 and
        // `tests/ui/closures/2229_closure_analysis/preserve_field_drop_order.rs`.
        for (_, captures) in &mut *root_var_min_capture_list {
            captures.sort_by(|capture1, capture2| {
                fn is_field(p: &&Projection) -> bool {
                    match p.kind {
                        ProjectionKind::Field { .. } => true,
                        ProjectionKind::Deref | ProjectionKind::UnwrapUnsafeBinder => false,
                        p @ (ProjectionKind::Subslice | ProjectionKind::Index) => {
                            panic!("ProjectionKind {:?} was unexpected", p)
                        }
                    }
                }

                // Need to sort only by Field projections, so filter away others.
                // A previous implementation considered other projection types too
                // but that caused ICE #118144
                let capture1_field_projections = capture1.place.projections.iter().filter(is_field);
                let capture2_field_projections = capture2.place.projections.iter().filter(is_field);

                for (p1, p2) in capture1_field_projections.zip(capture2_field_projections) {
                    // We do not need to look at the `Projection.ty` fields here because at each
                    // step of the iteration, the projections will either be the same and therefore
                    // the types must be as well or the current projection will be different and
                    // we will return the result of comparing the field indexes.
                    match (p1.kind, p2.kind) {
                        (
                            ProjectionKind::Field { field_idx: i1, .. },
                            ProjectionKind::Field { field_idx: i2, .. },
                        ) => {
                            // Compare only if paths are different.
                            // Otherwise continue to the next iteration
                            if i1 != i2 {
                                return i1.cmp(&i2);
                            }
                        }
                        // Given the filter above, this arm should never be hit
                        (l, r) => panic!("ProjectionKinds {:?} or {:?} were unexpected", l, r),
                    }
                }

                std::cmp::Ordering::Equal
            });
        }

        debug!(
            "For closure={:?}, min_captures after sorting={:#?}",
            closure_def_id, root_var_min_capture_list
        );
        self.result.closures_data.insert(closure_def_id, closure_data);
    }

    fn normalize_capture_place(&mut self, span: Span, place: Place) -> Place {
        let place = self.infcx().resolve_vars_if_possible(place);

        // In the new solver, types in HIR `Place`s can contain unnormalized aliases,
        // which can ICE later (e.g. when projecting fields for diagnostics).
        let cause = ObligationCause::new(span);
        let at = self.table.at(&cause);
        match normalize::deeply_normalize_with_skipped_universes_and_ambiguous_coroutine_goals(
            at,
            place.clone(),
            vec![],
        ) {
            Ok((normalized, goals)) => {
                if !goals.is_empty() {
                    // FIXME: Insert coroutine stalled predicates, this matters for MIR.
                    // let mut typeck_results = self.typeck_results.borrow_mut();
                    // typeck_results.coroutine_stalled_predicates.extend(
                    //     goals
                    //         .into_iter()
                    //         // FIXME: throwing away the param-env :(
                    //         .map(|goal| (goal.predicate, self.misc(span))),
                    // );
                }
                normalized
            }
            Err(errors) => {
                self.table.trait_errors.extend(errors);
                place
            }
        }
    }

    fn closure_min_captures_flattened(
        &self,
        closure_expr_id: ExprId,
    ) -> impl Iterator<Item = &CapturedPlace> {
        self.result
            .closures_data
            .get(&closure_expr_id)
            .map(|closure_data| closure_data.min_captures.values().flatten())
            .into_iter()
            .flatten()
    }

    fn init_capture_kind_for_place(
        &self,
        place: &Place,
        capture_clause: CaptureBy,
    ) -> UpvarCapture {
        match capture_clause {
            // In case of a move closure if the data is accessed through a reference we
            // want to capture by ref to allow precise capture using reborrows.
            //
            // If the data will be moved out of this place, then the place will be truncated
            // at the first Deref in `adjust_for_move_closure` and then moved into the closure.
            //
            // For example:
            //
            // struct Buffer<'a> {
            //     x: &'a String,
            //     y: Vec<u8>,
            // }
            //
            // fn get<'a>(b: Buffer<'a>) -> impl Sized + 'a {
            //     let c = move || b.x;
            //     drop(b);
            //     c
            // }
            //
            // Even though the closure is declared as move, when we are capturing borrowed data (in
            // this case, *b.x) we prefer to capture by reference.
            // Otherwise you'd get an error in 2021 immediately because you'd be trying to take
            // ownership of the (borrowed) String or else you'd take ownership of b, as in 2018 and
            // before, which is also an error.
            CaptureBy::Value if !place.deref_tys().any(Ty::is_ref) => UpvarCapture::ByValue,
            CaptureBy::Value | CaptureBy::Ref => UpvarCapture::ByRef(BorrowKind::Immutable),
        }
    }

    fn place_for_root_variable(&mut self, closure_def_id: ExprId, var_hir_id: BindingId) -> Place {
        let place = Place {
            base_ty: self.result.binding_ty(var_hir_id).store(),
            base: PlaceBase::Upvar { closure: closure_def_id, var_id: var_hir_id },
            projections: Default::default(),
        };

        // Normalize eagerly when inserting into `capture_information`, so all downstream
        // capture analysis can assume a normalized `Place`.
        self.normalize_capture_place(var_hir_id.into(), place)
    }

    /// A captured place is mutable if
    /// 1. Projections don't include a Deref of an immut-borrow, **and**
    /// 2. PlaceBase is mut or projections include a Deref of a mut-borrow.
    fn determine_capture_mutability(&mut self, closure_expr: ExprId, place: &Place) -> Mutability {
        let var_hir_id = match place.base {
            PlaceBase::Upvar { var_id, .. } => var_id,
            _ => unreachable!(),
        };

        let mut is_mutbl = if self.store[var_hir_id].mode == BindingAnnotation::Mutable {
            Mutability::Mut
        } else {
            Mutability::Not
        };

        for pointer_ty in place.deref_tys() {
            match self.structurally_resolve_type(closure_expr.into(), pointer_ty).kind() {
                // We don't capture derefs of raw ptrs
                TyKind::RawPtr(_, _) => unreachable!(),

                // Dereferencing a mut-ref allows us to mut the Place if we don't deref
                // an immut-ref after on top of this.
                TyKind::Ref(.., Mutability::Mut) => is_mutbl = Mutability::Mut,

                // The place isn't mutable once we dereference an immutable reference.
                TyKind::Ref(.., Mutability::Not) => return Mutability::Not,

                // Dereferencing a box doesn't change mutability
                TyKind::Adt(def, ..) if def.is_box() => {}

                unexpected_ty => panic!("deref of unexpected pointer type {:?}", unexpected_ty),
            }
        }

        is_mutbl
    }
}

/// Determines whether a child capture that is derived from a parent capture
/// should be borrowed with the lifetime of the parent coroutine-closure's env.
///
/// There are two cases when this needs to happen:
///
/// (1.) Are we borrowing data owned by the parent closure? We can determine if
/// that is the case by checking if the parent capture is by move, EXCEPT if we
/// apply a deref projection of an immutable reference, reborrows of immutable
/// references which aren't restricted to the LUB of the lifetimes of the deref
/// chain. This is why `&'short mut &'long T` can be reborrowed as `&'long T`.
///
/// ```rust
/// let x = &1i32; // Let's call this lifetime `'1`.
/// let c = async move || {
///     println!("{:?}", *x);
///     // Even though the inner coroutine borrows by ref, we're only capturing `*x`,
///     // not `x`, so the inner closure is allowed to reborrow the data for `'1`.
/// };
/// ```
///
/// (2.) If a coroutine is mutably borrowing from a parent capture, then that
/// mutable borrow cannot live for longer than either the parent *or* the borrow
/// that we have on the original upvar. Therefore we always need to borrow the
/// child capture with the lifetime of the parent coroutine-closure's env.
///
/// ```rust
/// let mut x = 1i32;
/// let c = async || {
///     x = 1;
///     // The parent borrows `x` for some `&'1 mut i32`.
///     // However, when we call `c()`, we implicitly autoref for the signature of
///     // `AsyncFnMut::async_call_mut`. Let's call that lifetime `'call`. Since
///     // the maximum that `&'call mut &'1 mut i32` can be reborrowed is `&'call mut i32`,
///     // the inner coroutine should capture w/ the lifetime of the coroutine-closure.
/// };
/// ```
///
/// If either of these cases apply, then we should capture the borrow with the
/// lifetime of the parent coroutine-closure's env. Luckily, if this function is
/// not correct, then the program is not unsound, since we still borrowck and validate
/// the choices made from this function -- the only side-effect is that the user
/// may receive unnecessary borrowck errors.
fn should_reborrow_from_env_of_parent_coroutine_closure(
    parent_capture: &CapturedPlace,
    child_capture: &CapturedPlace,
) -> bool {
    // (1.)
    (!parent_capture.is_by_ref()
        // This is just inlined `place.deref_tys()` but truncated to just
        // the child projections. Namely, look for a `&T` deref, since we
        // can always extend `&'short mut &'long T` to `&'long T`.
        && !child_capture
            .place
            .projections
            .iter()
            .enumerate()
            .skip(parent_capture.place.projections.len())
            .any(|(idx, proj)| {
                matches!(proj.kind, ProjectionKind::Deref)
                    && matches!(
                        child_capture.place.ty_before_projection(idx).kind(),
                        TyKind::Ref(.., Mutability::Not)
                    )
            }))
        // (2.)
        || matches!(child_capture.info.capture_kind, UpvarCapture::ByRef(BorrowKind::Mutable))
}

/// Truncate the capture so that the place being borrowed is in accordance with RFC 1240,
/// which states that it's unsafe to take a reference into a struct marked `repr(packed)`.
fn restrict_repr_packed_field_ref_capture(
    mut place: Place,
    capture_info: &mut CaptureInfo,
) -> Place {
    let pos = place.projections.iter().enumerate().position(|(i, p)| {
        let ty = place.ty_before_projection(i);

        // Return true for fields of packed structs.
        match p.kind {
            ProjectionKind::Field { .. } => match ty.kind() {
                TyKind::Adt(def, _) if def.is_packed() => {
                    // We stop here regardless of field alignment. Field alignment can change as
                    // types change, including the types of private fields in other crates, and that
                    // shouldn't affect how we compute our captures.
                    true
                }

                _ => false,
            },
            _ => false,
        }
    });

    if let Some(pos) = pos {
        truncate_place_to_len_and_update_capture_kind(&mut place, capture_info, pos);
    }

    place
}

/// Returns a Ty that applies the specified capture kind on the provided capture Ty
fn apply_capture_kind_on_capture_ty<'db>(
    interner: DbInterner<'db>,
    ty: Ty<'db>,
    capture_kind: UpvarCapture,
    region: Region<'db>,
) -> Ty<'db> {
    match capture_kind {
        UpvarCapture::ByValue | UpvarCapture::ByUse => ty,
        UpvarCapture::ByRef(kind) => Ty::new_ref(interner, region, ty, kind.to_mutbl_lossy()),
    }
}

struct InferBorrowKind {
    // The def-id of the closure whose kind and upvar accesses are being inferred.
    closure_def_id: ExprId,

    /// For each Place that is captured by the closure, we track the minimal kind of
    /// access we need (ref, ref mut, move, etc) and the expression that resulted in such access.
    ///
    /// Consider closure where s.str1 is captured via an ImmutableBorrow and
    /// s.str2 via a MutableBorrow
    ///
    /// ```rust,no_run
    /// struct SomeStruct { str1: String, str2: String };
    ///
    /// // Assume that the HirId for the variable definition is `V1`
    /// let mut s = SomeStruct { str1: format!("s1"), str2: format!("s2") };
    ///
    /// let fix_s = |new_s2| {
    ///     // Assume that the HirId for the expression `s.str1` is `E1`
    ///     println!("Updating SomeStruct with str1={0}", s.str1);
    ///     // Assume that the HirId for the expression `*s.str2` is `E2`
    ///     s.str2 = new_s2;
    /// };
    /// ```
    ///
    /// For closure `fix_s`, (at a high level) the map contains
    ///
    /// ```ignore (illustrative)
    /// Place { V1, [ProjectionKind::Field(Index=0, Variant=0)] } : CaptureKind { E1, ImmutableBorrow }
    /// Place { V1, [ProjectionKind::Field(Index=1, Variant=0)] } : CaptureKind { E2, MutableBorrow }
    /// ```
    capture_information: InferredCaptureInformation,
    fake_reads: Vec<(Place, FakeReadCause, SmallVec<[CaptureSourceStack; 2]>)>,
}

impl<'db> euv::Delegate<'db> for InferBorrowKind {
    #[instrument(skip(self), level = "debug")]
    fn fake_read(
        &mut self,
        place_with_id: PlaceWithOrigin,
        cause: FakeReadCause,
        ctx: &mut InferenceContext<'_, 'db>,
    ) {
        let PlaceBase::Upvar { .. } = place_with_id.place.base else { return };

        // We need to restrict Fake Read precision to avoid fake reading unsafe code,
        // such as deref of a raw pointer.
        let dummy_capture_kind = UpvarCapture::ByRef(BorrowKind::Immutable);
        let mut dummy_capture_info =
            CaptureInfo { sources: SmallVec::new(), capture_kind: dummy_capture_kind };

        let place = ctx.normalize_capture_place(place_with_id.span(), place_with_id.place.clone());

        let place = restrict_capture_precision(place, &mut dummy_capture_info);

        dummy_capture_info.capture_kind = dummy_capture_kind;
        let place = restrict_repr_packed_field_ref_capture(place, &mut dummy_capture_info);
        self.fake_reads.push((place, cause, place_with_id.origins));
    }

    #[instrument(skip(self), level = "debug")]
    fn consume(&mut self, place_with_id: PlaceWithOrigin, ctx: &mut InferenceContext<'_, 'db>) {
        let PlaceBase::Upvar { closure: upvar_closure, .. } = place_with_id.place.base else {
            return;
        };
        assert_eq!(self.closure_def_id, upvar_closure);

        let place = ctx.normalize_capture_place(place_with_id.span(), place_with_id.place.clone());

        self.capture_information.push((
            place,
            CaptureInfo { sources: place_with_id.origins, capture_kind: UpvarCapture::ByValue },
        ));
    }

    #[instrument(skip(self), level = "debug")]
    fn use_cloned(&mut self, place_with_id: PlaceWithOrigin, ctx: &mut InferenceContext<'_, 'db>) {
        let PlaceBase::Upvar { closure: upvar_closure, .. } = place_with_id.place.base else {
            return;
        };
        assert_eq!(self.closure_def_id, upvar_closure);

        let place = ctx.normalize_capture_place(place_with_id.span(), place_with_id.place.clone());

        self.capture_information.push((
            place,
            CaptureInfo { sources: place_with_id.origins, capture_kind: UpvarCapture::ByUse },
        ));
    }

    #[instrument(skip(self), level = "debug")]
    fn borrow(
        &mut self,
        place_with_id: PlaceWithOrigin,
        bk: BorrowKind,
        ctx: &mut InferenceContext<'_, 'db>,
    ) {
        let PlaceBase::Upvar { closure: upvar_closure, .. } = place_with_id.place.base else {
            return;
        };
        assert_eq!(self.closure_def_id, upvar_closure);

        // The region here will get discarded/ignored
        let capture_kind = UpvarCapture::ByRef(bk);
        let mut capture_info =
            CaptureInfo { sources: place_with_id.origins.iter().cloned().collect(), capture_kind };

        let place = ctx.normalize_capture_place(place_with_id.span(), place_with_id.place.clone());

        // We only want repr packed restriction to be applied to reading references into a packed
        // struct, and not when the data is being moved. Therefore we call this method here instead
        // of in `restrict_capture_precision`.
        let place = restrict_repr_packed_field_ref_capture(place, &mut capture_info);

        // Raw pointers don't inherit mutability
        if place.deref_tys().any(Ty::is_raw_ptr) {
            capture_info.capture_kind = UpvarCapture::ByRef(BorrowKind::Immutable);
        }

        self.capture_information.push((place, capture_info));
    }

    #[instrument(skip(self), level = "debug")]
    fn mutate(&mut self, assignee_place: PlaceWithOrigin, ctx: &mut InferenceContext<'_, 'db>) {
        self.borrow(assignee_place, BorrowKind::Mutable, ctx);
    }
}

/// Rust doesn't permit moving fields out of a type that implements drop
#[instrument(skip(fcx), ret, level = "debug")]
fn restrict_precision_for_drop_types<'a, 'db>(
    fcx: &mut InferenceContext<'a, 'db>,
    mut place: Place,
    capture_info: &mut CaptureInfo,
) -> Place {
    let is_copy_type = fcx.infcx().type_is_copy_modulo_regions(fcx.table.param_env, place.ty());

    if let (false, UpvarCapture::ByValue) = (is_copy_type, capture_info.capture_kind) {
        for i in 0..place.projections.len() {
            match place.ty_before_projection(i).kind() {
                TyKind::Adt(def, _) if def.destructor(fcx.interner()).is_some() => {
                    truncate_place_to_len_and_update_capture_kind(&mut place, capture_info, i);
                    break;
                }
                _ => {}
            }
        }
    }

    place
}

/// Truncate `place` so that an `unsafe` block isn't required to capture it.
/// - No projections are applied to raw pointers, since these require unsafe blocks. We capture
///   them completely.
/// - No projections are applied on top of Union ADTs, since these require unsafe blocks.
fn restrict_precision_for_unsafe(mut place: Place, capture_info: &mut CaptureInfo) -> Place {
    if place.base_ty.as_ref().is_raw_ptr() {
        truncate_place_to_len_and_update_capture_kind(&mut place, capture_info, 0);
    }

    if place.base_ty.as_ref().is_union() {
        truncate_place_to_len_and_update_capture_kind(&mut place, capture_info, 0);
    }

    for (i, proj) in place.projections.iter().enumerate() {
        if proj.ty.as_ref().is_raw_ptr() {
            // Don't apply any projections on top of a raw ptr.
            truncate_place_to_len_and_update_capture_kind(&mut place, capture_info, i + 1);
            break;
        }

        if proj.ty.as_ref().is_union() {
            // Don't capture precise fields of a union.
            truncate_place_to_len_and_update_capture_kind(&mut place, capture_info, i + 1);
            break;
        }
    }

    place
}

/// Truncate projections so that the following rules are obeyed by the captured `place`:
/// - No Index projections are captured, since arrays are captured completely.
/// - No unsafe block is required to capture `place`.
///
/// Returns the truncated place and updated capture mode.
#[instrument(ret, level = "debug")]
fn restrict_capture_precision(place: Place, capture_info: &mut CaptureInfo) -> Place {
    let mut place = restrict_precision_for_unsafe(place, capture_info);

    if place.projections.is_empty() {
        // Nothing to do here
        return place;
    }

    for (i, proj) in place.projections.iter().enumerate() {
        match proj.kind {
            ProjectionKind::Index | ProjectionKind::Subslice => {
                // Arrays are completely captured, so we drop Index and Subslice projections
                truncate_place_to_len_and_update_capture_kind(&mut place, capture_info, i);
                return place;
            }
            ProjectionKind::Deref => {}
            ProjectionKind::Field { .. } => {}
            ProjectionKind::UnwrapUnsafeBinder => {}
        }
    }

    place
}

/// Truncate deref of any reference.
#[instrument(ret, level = "debug")]
fn adjust_for_move_closure(mut place: Place, capture_info: &mut CaptureInfo) -> Place {
    let first_deref = place.projections.iter().position(|proj| proj.kind == ProjectionKind::Deref);

    if let Some(idx) = first_deref {
        truncate_place_to_len_and_update_capture_kind(&mut place, capture_info, idx);
    }

    capture_info.capture_kind = UpvarCapture::ByValue;
    place
}

/// Adjust closure capture just that if taking ownership of data, only move data
/// from enclosing stack frame.
#[instrument(ret, level = "debug")]
fn adjust_for_non_move_closure(mut place: Place, capture_info: &mut CaptureInfo) -> Place {
    let contains_deref =
        place.projections.iter().position(|proj| proj.kind == ProjectionKind::Deref);

    match capture_info.capture_kind {
        UpvarCapture::ByValue | UpvarCapture::ByUse => {
            if let Some(idx) = contains_deref {
                truncate_place_to_len_and_update_capture_kind(&mut place, capture_info, idx);
            }
        }

        UpvarCapture::ByRef(..) => {}
    }

    place
}

/// At the end, `capture_info_a` will contain the selected info.
fn determine_capture_info(capture_info_a: &mut CaptureInfo, capture_info_b: &mut CaptureInfo) {
    // If the capture kind is equivalent then, we don't need to escalate and can compare the
    // expressions.
    let eq_capture_kind = match (capture_info_a.capture_kind, capture_info_b.capture_kind) {
        (UpvarCapture::ByValue, UpvarCapture::ByValue) => true,
        (UpvarCapture::ByUse, UpvarCapture::ByUse) => true,
        (UpvarCapture::ByRef(ref_a), UpvarCapture::ByRef(ref_b)) => ref_a == ref_b,
        (UpvarCapture::ByValue, _) | (UpvarCapture::ByUse, _) | (UpvarCapture::ByRef(_), _) => {
            false
        }
    };

    let swap = if eq_capture_kind {
        false
    } else {
        // We select the CaptureKind which ranks higher based the following priority order:
        // (ByUse | ByValue) > MutBorrow > UniqueImmBorrow > ImmBorrow
        match (capture_info_a.capture_kind, capture_info_b.capture_kind) {
            (UpvarCapture::ByUse, UpvarCapture::ByValue)
            | (UpvarCapture::ByValue, UpvarCapture::ByUse) => {
                panic!("Same capture can't be ByUse and ByValue at the same time")
            }
            (UpvarCapture::ByValue, UpvarCapture::ByValue)
            | (UpvarCapture::ByUse, UpvarCapture::ByUse)
            | (UpvarCapture::ByValue | UpvarCapture::ByUse, UpvarCapture::ByRef(_)) => false,
            (UpvarCapture::ByRef(_), UpvarCapture::ByValue | UpvarCapture::ByUse) => true,
            (UpvarCapture::ByRef(ref_a), UpvarCapture::ByRef(ref_b)) => {
                match (ref_a, ref_b) {
                    // Take LHS:
                    (BorrowKind::UniqueImmutable | BorrowKind::Mutable, BorrowKind::Immutable)
                    | (BorrowKind::Mutable, BorrowKind::UniqueImmutable) => false,

                    // Take RHS:
                    (BorrowKind::Immutable, BorrowKind::UniqueImmutable | BorrowKind::Mutable)
                    | (BorrowKind::UniqueImmutable, BorrowKind::Mutable) => true,

                    (BorrowKind::Immutable, BorrowKind::Immutable)
                    | (BorrowKind::UniqueImmutable, BorrowKind::UniqueImmutable)
                    | (BorrowKind::Mutable, BorrowKind::Mutable) => {
                        panic!("Expected unequal capture kinds");
                    }
                }
            }
        }
    };

    if swap {
        mem::swap(capture_info_a, capture_info_b);
    }
}

fn determine_capture_sources(
    capture_info_a: &mut CaptureInfo,
    capture_info_b: &mut CaptureInfo,
    dedup_sources_scratch: &mut FxHashMap<ExprOrPatId, CaptureSourceStack>,
) -> SmallVec<[CaptureSourceStack; 2]> {
    dedup_sources_scratch.clear();
    dedup_sources_scratch.extend(
        mem::take(&mut capture_info_a.sources).into_iter().map(|it| (it.final_source(), it)),
    );
    dedup_sources_scratch.extend(
        mem::take(&mut capture_info_b.sources).into_iter().map(|it| (it.final_source(), it)),
    );

    let mut result = mem::take(&mut capture_info_a.sources);
    result.clear();
    result.extend(dedup_sources_scratch.values().cloned());
    result
}

/// Truncates `place` to have up to `len` projections.
/// `curr_mode` is the current required capture kind for the place.
/// Returns the truncated `place` and the updated required capture kind.
///
/// Note: Capture kind changes from `MutBorrow` to `UniqueImmBorrow` if the truncated part of the `place`
/// contained `Deref` of `&mut`.
fn truncate_place_to_len_and_update_capture_kind(
    place: &mut Place,
    info: &mut CaptureInfo,
    len: usize,
) {
    let is_mut_ref = |ty: Ty<'_>| matches!(ty.kind(), TyKind::Ref(.., Mutability::Mut));

    // If the truncated part of the place contains `Deref` of a `&mut` then convert MutBorrow ->
    // UniqueImmBorrow
    // Note that if the place contained Deref of a raw pointer it would've not been MutBorrow, so
    // we don't need to worry about that case here.
    match info.capture_kind {
        UpvarCapture::ByRef(BorrowKind::Mutable) => {
            for i in len..place.projections.len() {
                if place.projections[i].kind == ProjectionKind::Deref
                    && is_mut_ref(place.ty_before_projection(i))
                {
                    info.capture_kind = UpvarCapture::ByRef(BorrowKind::UniqueImmutable);
                    break;
                }
            }
        }

        UpvarCapture::ByRef(..) => {}
        UpvarCapture::ByValue | UpvarCapture::ByUse => {}
    }

    // Now fix the sources, to point at the smaller place.
    for source in &mut info.sources {
        // +1 because the first place is the base.
        source.truncate(len + 1);
    }

    place.projections.truncate(len);
}

/// Determines the Ancestry relationship of Place A relative to Place B
///
/// `PlaceAncestryRelation::Ancestor` implies Place A is ancestor of Place B
/// `PlaceAncestryRelation::Descendant` implies Place A is descendant of Place B
/// `PlaceAncestryRelation::Divergent` implies neither of them is the ancestor of the other.
fn determine_place_ancestry_relation(place_a: &Place, place_b: &Place) -> PlaceAncestryRelation {
    // If Place A and Place B don't start off from the same root variable, they are divergent.
    if place_a.base != place_b.base {
        return PlaceAncestryRelation::Divergent;
    }

    // Assume of length of projections_a = n
    let projections_a = &place_a.projections;

    // Assume of length of projections_b = m
    let projections_b = &place_b.projections;

    let same_initial_projections =
        iter::zip(projections_a, projections_b).all(|(proj_a, proj_b)| proj_a.kind == proj_b.kind);

    if same_initial_projections {
        use std::cmp::Ordering;

        // First min(n, m) projections are the same
        // Select Ancestor/Descendant
        match projections_b.len().cmp(&projections_a.len()) {
            Ordering::Greater => PlaceAncestryRelation::Ancestor,
            Ordering::Equal => PlaceAncestryRelation::SamePlace,
            Ordering::Less => PlaceAncestryRelation::Descendant,
        }
    } else {
        PlaceAncestryRelation::Divergent
    }
}

/// Reduces the precision of the captured place when the precision doesn't yield any benefit from
/// borrow checking perspective, allowing us to save us on the size of the capture.
///
///
/// Fields that are read through a shared reference will always be read via a shared ref or a copy,
/// and therefore capturing precise paths yields no benefit. This optimization truncates the
/// rightmost deref of the capture if the deref is applied to a shared ref.
///
/// Reason we only drop the last deref is because of the following edge case:
///
/// ```
/// # struct A { field_of_a: Box<i32> }
/// # struct B {}
/// # struct C<'a>(&'a i32);
/// struct MyStruct<'a> {
///    a: &'static A,
///    b: B,
///    c: C<'a>,
/// }
///
/// fn foo<'a, 'b>(m: &'a MyStruct<'b>) -> impl FnMut() + 'static {
///     || drop(&*m.a.field_of_a)
///     // Here we really do want to capture `*m.a` because that outlives `'static`
///
///     // If we capture `m`, then the closure no longer outlives `'static`
///     // it is constrained to `'a`
/// }
/// ```
#[instrument(ret, level = "debug")]
fn truncate_capture_for_optimization(mut place: Place, info: &mut CaptureInfo) -> Place {
    let is_shared_ref = |ty: Ty<'_>| matches!(ty.kind(), TyKind::Ref(.., Mutability::Not));

    // Find the rightmost deref (if any). All the projections that come after this
    // are fields or other "in-place pointer adjustments"; these refer therefore to
    // data owned by whatever pointer is being dereferenced here.
    let idx = place.projections.iter().rposition(|proj| ProjectionKind::Deref == proj.kind);

    match idx {
        // If that pointer is a shared reference, then we don't need those fields.
        Some(idx) if is_shared_ref(place.ty_before_projection(idx)) => {
            truncate_place_to_len_and_update_capture_kind(&mut place, info, idx + 1)
        }
        None | Some(_) => {}
    }

    place
}

/// Precise capture is enabled if user is using Rust Edition 2021 or higher.
/// `span` is the span of the closure.
fn enable_precise_capture(edition: Edition) -> bool {
    // FIXME: We should use the edition from the closure expr.
    edition.at_least_2021()
}

fn analyze_coroutine_closure_captures<'a, T>(
    parent_captures: impl IntoIterator<Item = &'a CapturedPlace>,
    child_captures: impl IntoIterator<Item = &'a CapturedPlace>,
    mut for_each: impl FnMut((usize, &'a CapturedPlace), (usize, &'a CapturedPlace)) -> T,
) -> impl Iterator<Item = T> {
    let mut result = SmallVec::<[_; 10]>::new();

    let mut child_captures = child_captures.into_iter().enumerate().peekable();

    // One parent capture may correspond to several child captures if we end up
    // refining the set of captures via edition-2021 precise captures. We want to
    // match up any number of child captures with one parent capture, so we keep
    // peeking off this `Peekable` until the child doesn't match anymore.
    for (parent_field_idx, parent_capture) in parent_captures.into_iter().enumerate() {
        // Make sure we use every field at least once, b/c why are we capturing something
        // if it's not used in the inner coroutine.
        let mut field_used_at_least_once = false;

        // A parent matches a child if they share the same prefix of projections.
        // The child may have more, if it is capturing sub-fields out of
        // something that is captured by-move in the parent closure.
        while child_captures.peek().is_some_and(|(_, child_capture)| {
            child_prefix_matches_parent_projections(parent_capture, child_capture)
        }) {
            let (child_field_idx, child_capture) = child_captures.next().unwrap();
            // This analysis only makes sense if the parent capture is a
            // prefix of the child capture.
            assert!(
                child_capture.place.projections.len() >= parent_capture.place.projections.len(),
                "parent capture ({parent_capture:#?}) expected to be prefix of \
                    child capture ({child_capture:#?})"
            );

            result.push(for_each(
                (parent_field_idx, parent_capture),
                (child_field_idx, child_capture),
            ));

            field_used_at_least_once = true;
        }

        // Make sure the field was used at least once.
        assert!(
            field_used_at_least_once,
            "we captured {parent_capture:#?} but it was not used in the child coroutine?"
        );
    }
    assert_eq!(child_captures.next(), None, "leftover child captures?");

    result.into_iter()
}

fn child_prefix_matches_parent_projections(
    parent_capture: &CapturedPlace,
    child_capture: &CapturedPlace,
) -> bool {
    let PlaceBase::Upvar { var_id: parent_base, .. } = parent_capture.place.base else {
        panic!("expected capture to be an upvar");
    };
    let PlaceBase::Upvar { var_id: child_base, .. } = child_capture.place.base else {
        panic!("expected capture to be an upvar");
    };

    parent_base == child_base
        && std::iter::zip(&child_capture.place.projections, &parent_capture.place.projections)
            .all(|(child, parent)| child.kind == parent.kind)
}
