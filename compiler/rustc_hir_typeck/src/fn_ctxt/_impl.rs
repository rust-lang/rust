use std::collections::hash_map::Entry;
use std::slice;

use rustc_abi::FieldIdx;
use rustc_data_structures::fx::FxHashSet;
use rustc_errors::{Applicability, Diag, ErrorGuaranteed, MultiSpan};
use rustc_hir::def::{CtorOf, DefKind, Res};
use rustc_hir::def_id::DefId;
use rustc_hir::intravisit::VisitorExt;
use rustc_hir::lang_items::LangItem;
use rustc_hir::{self as hir, AmbigArg, ExprKind, GenericArg, HirId, Node, QPath, intravisit};
use rustc_hir_analysis::hir_ty_lowering::errors::GenericsArgsErrExtend;
use rustc_hir_analysis::hir_ty_lowering::generics::{
    check_generic_arg_count_for_call, lower_generic_args,
};
use rustc_hir_analysis::hir_ty_lowering::{
    ExplicitLateBound, FeedConstTy, GenericArgCountMismatch, GenericArgCountResult,
    GenericArgsLowerer, GenericPathSegment, HirTyLowerer, IsMethodCall, RegionInferReason,
};
use rustc_infer::infer::canonical::{Canonical, OriginalQueryValues, QueryResponse};
use rustc_infer::infer::{DefineOpaqueTypes, InferResult};
use rustc_lint::builtin::SELF_CONSTRUCTOR_FROM_OUTER_ITEM;
use rustc_middle::ty::adjustment::{Adjust, Adjustment, AutoBorrow, AutoBorrowMutability};
use rustc_middle::ty::{
    self, AdtKind, CanonicalUserType, GenericArgsRef, GenericParamDefKind, IsIdentity,
    SizedTraitKind, Ty, TyCtxt, TypeFoldable, TypeVisitable, TypeVisitableExt, UserArgs,
    UserSelfTy,
};
use rustc_middle::{bug, span_bug};
use rustc_session::lint;
use rustc_span::Span;
use rustc_span::def_id::LocalDefId;
use rustc_span::hygiene::DesugaringKind;
use rustc_trait_selection::error_reporting::infer::need_type_info::TypeAnnotationNeeded;
use rustc_trait_selection::traits::{
    self, NormalizeExt, ObligationCauseCode, StructurallyNormalizeExt,
};
use tracing::{debug, instrument};

use crate::callee::{self, DeferredCallResolution};
use crate::errors::{self, CtorIsPrivate};
use crate::method::{self, MethodCallee};
use crate::{BreakableCtxt, Diverges, Expectation, FnCtxt, LoweredTy, rvalue_scopes};

impl<'a, 'tcx> FnCtxt<'a, 'tcx> {
    /// Produces warning on the given node, if the current point in the
    /// function is unreachable, and there hasn't been another warning.
    pub(crate) fn warn_if_unreachable(&self, id: HirId, span: Span, kind: &str) {
        let Diverges::Always { span: orig_span, custom_note } = self.diverges.get() else {
            return;
        };

        match span.desugaring_kind() {
            // If span arose from a desugaring of `if` or `while`, then it is the condition
            // itself, which diverges, that we are about to lint on. This gives suboptimal
            // diagnostics. Instead, stop here so that the `if`- or `while`-expression's
            // block is linted instead.
            Some(DesugaringKind::CondTemporary) => return,

            // Don't lint if the result of an async block or async function is `!`.
            // This does not affect the unreachable lints *within* the body.
            Some(DesugaringKind::Async) => return,

            // Don't lint *within* the `.await` operator, since that's all just desugaring
            // junk. We only want to lint if there is a subsequent expression after the
            // `.await` operator.
            Some(DesugaringKind::Await) => return,

            _ => {}
        }

        // Don't warn twice.
        self.diverges.set(Diverges::WarnedAlways);

        debug!("warn_if_unreachable: id={:?} span={:?} kind={}", id, span, kind);

        let msg = format!("unreachable {kind}");
        self.tcx().node_span_lint(lint::builtin::UNREACHABLE_CODE, id, span, |lint| {
            lint.primary_message(msg.clone());
            lint.span_label(span, msg).span_label(
                orig_span,
                custom_note.unwrap_or("any code following this expression is unreachable"),
            );
        })
    }

    /// Resolves type and const variables in `t` if possible. Unlike the infcx
    /// version (resolve_vars_if_possible), this version will
    /// also select obligations if it seems useful, in an effort
    /// to get more type information.
    // FIXME(-Znext-solver): A lot of the calls to this method should
    // probably be `try_structurally_resolve_type` or `structurally_resolve_type` instead.
    #[instrument(skip(self), level = "debug", ret)]
    pub(crate) fn resolve_vars_with_obligations<T: TypeFoldable<TyCtxt<'tcx>>>(
        &self,
        mut t: T,
    ) -> T {
        // No Infer()? Nothing needs doing.
        if !t.has_non_region_infer() {
            debug!("no inference var, nothing needs doing");
            return t;
        }

        // If `t` is a type variable, see whether we already know what it is.
        t = self.resolve_vars_if_possible(t);
        if !t.has_non_region_infer() {
            debug!(?t);
            return t;
        }

        // If not, try resolving pending obligations as much as
        // possible. This can help substantially when there are
        // indirect dependencies that don't seem worth tracking
        // precisely.
        self.select_obligations_where_possible(|_| {});
        self.resolve_vars_if_possible(t)
    }

    pub(crate) fn record_deferred_call_resolution(
        &self,
        closure_def_id: LocalDefId,
        r: DeferredCallResolution<'tcx>,
    ) {
        let mut deferred_call_resolutions = self.deferred_call_resolutions.borrow_mut();
        deferred_call_resolutions.entry(closure_def_id).or_default().push(r);
    }

    pub(crate) fn remove_deferred_call_resolutions(
        &self,
        closure_def_id: LocalDefId,
    ) -> Vec<DeferredCallResolution<'tcx>> {
        let mut deferred_call_resolutions = self.deferred_call_resolutions.borrow_mut();
        deferred_call_resolutions.remove(&closure_def_id).unwrap_or_default()
    }

    fn tag(&self) -> String {
        format!("{self:p}")
    }

    pub(crate) fn local_ty(&self, span: Span, nid: HirId) -> Ty<'tcx> {
        self.locals.borrow().get(&nid).cloned().unwrap_or_else(|| {
            span_bug!(span, "no type for local variable {}", self.tcx.hir_id_to_string(nid))
        })
    }

    #[inline]
    pub(crate) fn write_ty(&self, id: HirId, ty: Ty<'tcx>) {
        debug!("write_ty({:?}, {:?}) in fcx {}", id, self.resolve_vars_if_possible(ty), self.tag());
        let mut typeck = self.typeck_results.borrow_mut();
        let mut node_ty = typeck.node_types_mut();

        if let Some(prev) = node_ty.insert(id, ty) {
            if prev.references_error() {
                node_ty.insert(id, prev);
            } else if !ty.references_error() {
                // Could change this to a bug, but there's lots of diagnostic code re-lowering
                // or re-typechecking nodes that were already typecked.
                // Lots of that diagnostics code relies on subtle effects of re-lowering, so we'll
                // let it keep doing that and just ensure that compilation won't succeed.
                self.dcx().span_delayed_bug(
                    self.tcx.hir_span(id),
                    format!("`{prev}` overridden by `{ty}` for {id:?} in {:?}", self.body_id),
                );
            }
        }

        if let Err(e) = ty.error_reported() {
            self.set_tainted_by_errors(e);
        }
    }

    pub(crate) fn write_field_index(&self, hir_id: HirId, index: FieldIdx) {
        self.typeck_results.borrow_mut().field_indices_mut().insert(hir_id, index);
    }

    #[instrument(level = "debug", skip(self))]
    pub(crate) fn write_resolution(
        &self,
        hir_id: HirId,
        r: Result<(DefKind, DefId), ErrorGuaranteed>,
    ) {
        self.typeck_results.borrow_mut().type_dependent_defs_mut().insert(hir_id, r);
    }

    #[instrument(level = "debug", skip(self))]
    pub(crate) fn write_method_call_and_enforce_effects(
        &self,
        hir_id: HirId,
        span: Span,
        method: MethodCallee<'tcx>,
    ) {
        self.enforce_context_effects(Some(hir_id), span, method.def_id, method.args);
        self.write_resolution(hir_id, Ok((DefKind::AssocFn, method.def_id)));
        self.write_args(hir_id, method.args);
    }

    fn write_args(&self, node_id: HirId, args: GenericArgsRef<'tcx>) {
        if !args.is_empty() {
            debug!("write_args({:?}, {:?}) in fcx {}", node_id, args, self.tag());

            self.typeck_results.borrow_mut().node_args_mut().insert(node_id, args);
        }
    }

    /// Given the args that we just converted from the HIR, try to
    /// canonicalize them and store them as user-given parameters
    /// (i.e., parameters that must be respected by the NLL check).
    ///
    /// This should be invoked **before any unifications have
    /// occurred**, so that annotations like `Vec<_>` are preserved
    /// properly.
    #[instrument(skip(self), level = "debug")]
    pub(crate) fn write_user_type_annotation_from_args(
        &self,
        hir_id: HirId,
        def_id: DefId,
        args: GenericArgsRef<'tcx>,
        user_self_ty: Option<UserSelfTy<'tcx>>,
    ) {
        debug!("fcx {}", self.tag());

        // Don't write user type annotations for const param types, since we give them
        // identity args just so that we can trivially substitute their `EarlyBinder`.
        // We enforce that they match their type in MIR later on.
        if matches!(self.tcx.def_kind(def_id), DefKind::ConstParam) {
            return;
        }

        if Self::can_contain_user_lifetime_bounds((args, user_self_ty)) {
            let canonicalized = self.canonicalize_user_type_annotation(ty::UserType::new(
                ty::UserTypeKind::TypeOf(def_id, UserArgs { args, user_self_ty }),
            ));
            debug!(?canonicalized);
            self.write_user_type_annotation(hir_id, canonicalized);
        }
    }

    #[instrument(skip(self), level = "debug")]
    pub(crate) fn write_user_type_annotation(
        &self,
        hir_id: HirId,
        canonical_user_type_annotation: CanonicalUserType<'tcx>,
    ) {
        debug!("fcx {}", self.tag());

        // FIXME: is_identity being on `UserType` and not `Canonical<UserType>` is awkward
        if !canonical_user_type_annotation.is_identity() {
            self.typeck_results
                .borrow_mut()
                .user_provided_types_mut()
                .insert(hir_id, canonical_user_type_annotation);
        } else {
            debug!("skipping identity args");
        }
    }

    #[instrument(skip(self, expr), level = "debug")]
    pub(crate) fn apply_adjustments(&self, expr: &hir::Expr<'_>, adj: Vec<Adjustment<'tcx>>) {
        debug!("expr = {:#?}", expr);

        if adj.is_empty() {
            return;
        }

        let mut expr_ty = self.typeck_results.borrow().expr_ty_adjusted(expr);

        for a in &adj {
            match a.kind {
                Adjust::NeverToAny => {
                    if a.target.is_ty_var() {
                        self.diverging_type_vars.borrow_mut().insert(a.target);
                        debug!("apply_adjustments: adding `{:?}` as diverging type var", a.target);
                    }
                }
                Adjust::Deref(Some(overloaded_deref)) => {
                    self.enforce_context_effects(
                        None,
                        expr.span,
                        overloaded_deref.method_call(self.tcx),
                        self.tcx.mk_args(&[expr_ty.into()]),
                    );
                }
                Adjust::Deref(None) => {
                    // FIXME(const_trait_impl): We *could* enforce `&T: ~const Deref` here.
                }
                Adjust::Pointer(_pointer_coercion) => {
                    // FIXME(const_trait_impl): We should probably enforce these.
                }
                Adjust::ReborrowPin(_mutability) => {
                    // FIXME(const_trait_impl): We could enforce these; they correspond to
                    // `&mut T: DerefMut` tho, so it's kinda moot.
                }
                Adjust::Borrow(_) => {
                    // No effects to enforce here.
                }
            }

            expr_ty = a.target;
        }

        let autoborrow_mut = adj.iter().any(|adj| {
            matches!(
                adj,
                &Adjustment {
                    kind: Adjust::Borrow(AutoBorrow::Ref(AutoBorrowMutability::Mut { .. })),
                    ..
                }
            )
        });

        match self.typeck_results.borrow_mut().adjustments_mut().entry(expr.hir_id) {
            Entry::Vacant(entry) => {
                entry.insert(adj);
            }
            Entry::Occupied(mut entry) => {
                debug!(" - composing on top of {:?}", entry.get());
                match (&mut entry.get_mut()[..], &adj[..]) {
                    (
                        [Adjustment { kind: Adjust::NeverToAny, target }],
                        &[.., Adjustment { target: new_target, .. }],
                    ) => {
                        // NeverToAny coercion can target any type, so instead of adding a new
                        // adjustment on top we can change the target.
                        //
                        // This is required for things like `a == a` (where `a: !`) to produce
                        // valid MIR -- we need borrow adjustment from things like `==` to change
                        // the type to `&!` (or `&()` depending on the fallback). This might be
                        // relevant even in unreachable code.
                        *target = new_target;
                    }

                    (
                        &mut [
                            Adjustment { kind: Adjust::Deref(_), .. },
                            Adjustment { kind: Adjust::Borrow(AutoBorrow::Ref(..)), .. },
                        ],
                        &[
                            Adjustment { kind: Adjust::Deref(_), .. },
                            .., // Any following adjustments are allowed.
                        ],
                    ) => {
                        // A reborrow has no effect before a dereference, so we can safely replace adjustments.
                        *entry.get_mut() = adj;
                    }

                    _ => {
                        // FIXME: currently we never try to compose autoderefs
                        // and ReifyFnPointer/UnsafeFnPointer, but we could.
                        self.dcx().span_delayed_bug(
                            expr.span,
                            format!(
                                "while adjusting {:?}, can't compose {:?} and {:?}",
                                expr,
                                entry.get(),
                                adj
                            ),
                        );

                        *entry.get_mut() = adj;
                    }
                }
            }
        }

        // If there is an mutable auto-borrow, it is equivalent to `&mut <expr>`.
        // In this case implicit use of `Deref` and `Index` within `<expr>` should
        // instead be `DerefMut` and `IndexMut`, so fix those up.
        if autoborrow_mut {
            self.convert_place_derefs_to_mutable(expr);
        }
    }

    /// Instantiates and normalizes the bounds for a given item
    pub(crate) fn instantiate_bounds(
        &self,
        span: Span,
        def_id: DefId,
        args: GenericArgsRef<'tcx>,
    ) -> ty::InstantiatedPredicates<'tcx> {
        let bounds = self.tcx.predicates_of(def_id);
        let result = bounds.instantiate(self.tcx, args);
        let result = self.normalize(span, result);
        debug!("instantiate_bounds(bounds={:?}, args={:?}) = {:?}", bounds, args, result);
        result
    }

    pub(crate) fn normalize<T>(&self, span: Span, value: T) -> T
    where
        T: TypeFoldable<TyCtxt<'tcx>>,
    {
        self.register_infer_ok_obligations(
            self.at(&self.misc(span), self.param_env).normalize(value),
        )
    }

    pub(crate) fn require_type_meets(
        &self,
        ty: Ty<'tcx>,
        span: Span,
        code: traits::ObligationCauseCode<'tcx>,
        def_id: DefId,
    ) {
        self.register_bound(ty, def_id, self.cause(span, code));
    }

    pub(crate) fn require_type_is_sized(
        &self,
        ty: Ty<'tcx>,
        span: Span,
        code: traits::ObligationCauseCode<'tcx>,
    ) {
        if !ty.references_error() {
            let lang_item = self.tcx.require_lang_item(LangItem::Sized, span);
            self.require_type_meets(ty, span, code, lang_item);
        }
    }

    pub(crate) fn require_type_is_sized_deferred(
        &self,
        ty: Ty<'tcx>,
        span: Span,
        code: traits::ObligationCauseCode<'tcx>,
    ) {
        if !ty.references_error() {
            self.deferred_sized_obligations.borrow_mut().push((ty, span, code));
        }
    }

    pub(crate) fn require_type_has_static_alignment(&self, ty: Ty<'tcx>, span: Span) {
        if !ty.references_error() {
            let tail = self.tcx.struct_tail_raw(
                ty,
                |ty| {
                    if self.next_trait_solver() {
                        self.try_structurally_resolve_type(span, ty)
                    } else {
                        self.normalize(span, ty)
                    }
                },
                || {},
            );
            // Sized types have static alignment, and so do slices.
            if tail.has_trivial_sizedness(self.tcx, SizedTraitKind::Sized)
                || matches!(tail.kind(), ty::Slice(..))
            {
                // Nothing else is required here.
            } else {
                // We can't be sure, let's required full `Sized`.
                let lang_item = self.tcx.require_lang_item(LangItem::Sized, span);
                self.require_type_meets(ty, span, ObligationCauseCode::Misc, lang_item);
            }
        }
    }

    pub(crate) fn register_bound(
        &self,
        ty: Ty<'tcx>,
        def_id: DefId,
        cause: traits::ObligationCause<'tcx>,
    ) {
        if !ty.references_error() {
            self.fulfillment_cx.borrow_mut().register_bound(
                self,
                self.param_env,
                ty,
                def_id,
                cause,
            );
        }
    }

    pub(crate) fn lower_ty(&self, hir_ty: &hir::Ty<'tcx>) -> LoweredTy<'tcx> {
        let ty = self.lowerer().lower_ty(hir_ty);
        self.register_wf_obligation(ty.into(), hir_ty.span, ObligationCauseCode::WellFormed(None));
        LoweredTy::from_raw(self, hir_ty.span, ty)
    }

    /// Walk a `hir_ty` and collect any clauses that may have come from a type
    /// within the `hir_ty`. These clauses will be canonicalized with a user type
    /// annotation so that we can enforce these bounds in borrowck, too.
    pub(crate) fn collect_impl_trait_clauses_from_hir_ty(
        &self,
        hir_ty: &'tcx hir::Ty<'tcx>,
    ) -> ty::Clauses<'tcx> {
        struct CollectClauses<'a, 'tcx> {
            clauses: Vec<ty::Clause<'tcx>>,
            fcx: &'a FnCtxt<'a, 'tcx>,
        }

        impl<'tcx> intravisit::Visitor<'tcx> for CollectClauses<'_, 'tcx> {
            fn visit_ty(&mut self, ty: &'tcx hir::Ty<'tcx, AmbigArg>) {
                if let Some(clauses) = self.fcx.trait_ascriptions.borrow().get(&ty.hir_id.local_id)
                {
                    self.clauses.extend(clauses.iter().cloned());
                }
                intravisit::walk_ty(self, ty)
            }
        }

        let mut clauses = CollectClauses { clauses: vec![], fcx: self };
        clauses.visit_ty_unambig(hir_ty);
        self.tcx.mk_clauses(&clauses.clauses)
    }

    #[instrument(level = "debug", skip_all)]
    pub(crate) fn lower_ty_saving_user_provided_ty(&self, hir_ty: &'tcx hir::Ty<'tcx>) -> Ty<'tcx> {
        let ty = self.lower_ty(hir_ty);
        debug!(?ty);

        if Self::can_contain_user_lifetime_bounds(ty.raw) {
            let c_ty = self.canonicalize_response(ty::UserType::new(ty::UserTypeKind::Ty(ty.raw)));
            debug!(?c_ty);
            self.typeck_results.borrow_mut().user_provided_types_mut().insert(hir_ty.hir_id, c_ty);
        }

        ty.normalized
    }

    pub(super) fn user_args_for_adt(ty: LoweredTy<'tcx>) -> UserArgs<'tcx> {
        match (ty.raw.kind(), ty.normalized.kind()) {
            (ty::Adt(_, args), _) => UserArgs { args, user_self_ty: None },
            (_, ty::Adt(adt, args)) => UserArgs {
                args,
                user_self_ty: Some(UserSelfTy { impl_def_id: adt.did(), self_ty: ty.raw }),
            },
            _ => bug!("non-adt type {:?}", ty),
        }
    }

    pub(crate) fn lower_const_arg(
        &self,
        const_arg: &'tcx hir::ConstArg<'tcx>,
        feed: FeedConstTy<'_, 'tcx>,
    ) -> ty::Const<'tcx> {
        let ct = self.lowerer().lower_const_arg(const_arg, feed);
        self.register_wf_obligation(
            ct.into(),
            self.tcx.hir_span(const_arg.hir_id),
            ObligationCauseCode::WellFormed(None),
        );
        ct
    }

    // If the type given by the user has free regions, save it for later, since
    // NLL would like to enforce those. Also pass in types that involve
    // projections, since those can resolve to `'static` bounds (modulo #54940,
    // which hopefully will be fixed by the time you see this comment, dear
    // reader, although I have my doubts). Also pass in types with inference
    // types, because they may be repeated. Other sorts of things are already
    // sufficiently enforced with erased regions. =)
    fn can_contain_user_lifetime_bounds<T>(t: T) -> bool
    where
        T: TypeVisitable<TyCtxt<'tcx>>,
    {
        t.has_free_regions() || t.has_aliases() || t.has_infer_types()
    }

    pub(crate) fn node_ty(&self, id: HirId) -> Ty<'tcx> {
        match self.typeck_results.borrow().node_types().get(id) {
            Some(&t) => t,
            None if let Some(e) = self.tainted_by_errors() => Ty::new_error(self.tcx, e),
            None => {
                bug!("no type for node {} in fcx {}", self.tcx.hir_id_to_string(id), self.tag());
            }
        }
    }

    pub(crate) fn node_ty_opt(&self, id: HirId) -> Option<Ty<'tcx>> {
        match self.typeck_results.borrow().node_types().get(id) {
            Some(&t) => Some(t),
            None if let Some(e) = self.tainted_by_errors() => Some(Ty::new_error(self.tcx, e)),
            None => None,
        }
    }

    /// Registers an obligation for checking later, during regionck, that `arg` is well-formed.
    pub(crate) fn register_wf_obligation(
        &self,
        term: ty::Term<'tcx>,
        span: Span,
        code: traits::ObligationCauseCode<'tcx>,
    ) {
        // WF obligations never themselves fail, so no real need to give a detailed cause:
        let cause = self.cause(span, code);
        self.register_predicate(traits::Obligation::new(
            self.tcx,
            cause,
            self.param_env,
            ty::ClauseKind::WellFormed(term),
        ));
    }

    /// Registers obligations that all `args` are well-formed.
    pub(crate) fn add_wf_bounds(&self, args: GenericArgsRef<'tcx>, span: Span) {
        for term in args.iter().filter_map(ty::GenericArg::as_term) {
            self.register_wf_obligation(term, span, ObligationCauseCode::WellFormed(None));
        }
    }

    // FIXME(arielb1): use this instead of field.ty everywhere
    // Only for fields! Returns <none> for methods>
    // Indifferent to privacy flags
    pub(crate) fn field_ty(
        &self,
        span: Span,
        field: &'tcx ty::FieldDef,
        args: GenericArgsRef<'tcx>,
    ) -> Ty<'tcx> {
        self.normalize(span, field.ty(self.tcx, args))
    }

    pub(crate) fn resolve_rvalue_scopes(&self, def_id: DefId) {
        let scope_tree = self.tcx.region_scope_tree(def_id);
        let rvalue_scopes = { rvalue_scopes::resolve_rvalue_scopes(self, scope_tree, def_id) };
        let mut typeck_results = self.typeck_results.borrow_mut();
        typeck_results.rvalue_scopes = rvalue_scopes;
    }

    /// Unify the inference variables corresponding to coroutine witnesses, and save all the
    /// predicates that were stalled on those inference variables.
    ///
    /// This process allows to conservatively save all predicates that do depend on the coroutine
    /// interior types, for later processing by `check_coroutine_obligations`.
    ///
    /// We must not attempt to select obligations after this method has run, or risk query cycle
    /// ICE.
    #[instrument(level = "debug", skip(self))]
    pub(crate) fn resolve_coroutine_interiors(&self) {
        // Try selecting all obligations that are not blocked on inference variables.
        // Once we start unifying coroutine witnesses, trying to select obligations on them will
        // trigger query cycle ICEs, as doing so requires MIR.
        self.select_obligations_where_possible(|_| {});

        let coroutines = std::mem::take(&mut *self.deferred_coroutine_interiors.borrow_mut());
        debug!(?coroutines);

        let mut obligations = vec![];

        if !self.next_trait_solver() {
            for &(coroutine_def_id, interior) in coroutines.iter() {
                debug!(?coroutine_def_id);

                // Create the `CoroutineWitness` type that we will unify with `interior`.
                let args = ty::GenericArgs::identity_for_item(
                    self.tcx,
                    self.tcx.typeck_root_def_id(coroutine_def_id.to_def_id()),
                );
                let witness =
                    Ty::new_coroutine_witness(self.tcx, coroutine_def_id.to_def_id(), args);

                // Unify `interior` with `witness` and collect all the resulting obligations.
                let span = self.tcx.hir_body_owned_by(coroutine_def_id).value.span;
                let ty::Infer(ty::InferTy::TyVar(_)) = interior.kind() else {
                    span_bug!(span, "coroutine interior witness not infer: {:?}", interior.kind())
                };
                let ok = self
                    .at(&self.misc(span), self.param_env)
                    // Will never define opaque types, as all we do is instantiate a type variable.
                    .eq(DefineOpaqueTypes::Yes, interior, witness)
                    .expect("Failed to unify coroutine interior type");

                obligations.extend(ok.obligations);
            }
        }

        if !coroutines.is_empty() {
            obligations.extend(
                self.fulfillment_cx
                    .borrow_mut()
                    .drain_stalled_obligations_for_coroutines(&self.infcx),
            );
        }

        self.typeck_results
            .borrow_mut()
            .coroutine_stalled_predicates
            .extend(obligations.into_iter().map(|o| (o.predicate, o.cause)));
    }

    #[instrument(skip(self), level = "debug")]
    pub(crate) fn report_ambiguity_errors(&self) {
        let mut errors = self.fulfillment_cx.borrow_mut().collect_remaining_errors(self);

        if !errors.is_empty() {
            self.adjust_fulfillment_errors_for_expr_obligation(&mut errors);
            self.err_ctxt().report_fulfillment_errors(errors);
        }
    }

    /// Select as many obligations as we can at present.
    pub(crate) fn select_obligations_where_possible(
        &self,
        mutate_fulfillment_errors: impl Fn(&mut Vec<traits::FulfillmentError<'tcx>>),
    ) {
        let mut result = self.fulfillment_cx.borrow_mut().select_where_possible(self);
        if !result.is_empty() {
            mutate_fulfillment_errors(&mut result);
            self.adjust_fulfillment_errors_for_expr_obligation(&mut result);
            self.err_ctxt().report_fulfillment_errors(result);
        }
    }

    /// For the overloaded place expressions (`*x`, `x[3]`), the trait
    /// returns a type of `&T`, but the actual type we assign to the
    /// *expression* is `T`. So this function just peels off the return
    /// type by one layer to yield `T`.
    pub(crate) fn make_overloaded_place_return_type(&self, method: MethodCallee<'tcx>) -> Ty<'tcx> {
        // extract method return type, which will be &T;
        let ret_ty = method.sig.output();

        // method returns &T, but the type as visible to user is T, so deref
        ret_ty.builtin_deref(true).unwrap()
    }

    pub(crate) fn type_var_is_sized(&self, self_ty: ty::TyVid) -> bool {
        let sized_did = self.tcx.lang_items().sized_trait();
        self.obligations_for_self_ty(self_ty).into_iter().any(|obligation| {
            match obligation.predicate.kind().skip_binder() {
                ty::PredicateKind::Clause(ty::ClauseKind::Trait(data)) => {
                    Some(data.def_id()) == sized_did
                }
                _ => false,
            }
        })
    }

    pub(crate) fn err_args(&self, len: usize, guar: ErrorGuaranteed) -> Vec<Ty<'tcx>> {
        let ty_error = Ty::new_error(self.tcx, guar);
        vec![ty_error; len]
    }

    pub(crate) fn resolve_lang_item_path(
        &self,
        lang_item: hir::LangItem,
        span: Span,
        hir_id: HirId,
    ) -> (Res, Ty<'tcx>) {
        let def_id = self.tcx.require_lang_item(lang_item, span);
        let def_kind = self.tcx.def_kind(def_id);

        let item_ty = if let DefKind::Variant = def_kind {
            self.tcx.type_of(self.tcx.parent(def_id))
        } else {
            self.tcx.type_of(def_id)
        };
        let args = self.fresh_args_for_item(span, def_id);
        let ty = item_ty.instantiate(self.tcx, args);

        self.write_args(hir_id, args);
        self.write_resolution(hir_id, Ok((def_kind, def_id)));

        let code = match lang_item {
            hir::LangItem::IntoFutureIntoFuture => {
                if let hir::Node::Expr(into_future_call) = self.tcx.parent_hir_node(hir_id)
                    && let hir::ExprKind::Call(_, [arg0]) = &into_future_call.kind
                {
                    Some(ObligationCauseCode::AwaitableExpr(arg0.hir_id))
                } else {
                    None
                }
            }
            hir::LangItem::IteratorNext | hir::LangItem::IntoIterIntoIter => {
                Some(ObligationCauseCode::ForLoopIterator)
            }
            hir::LangItem::TryTraitFromOutput
            | hir::LangItem::TryTraitFromResidual
            | hir::LangItem::TryTraitBranch => Some(ObligationCauseCode::QuestionMark),
            _ => None,
        };
        if let Some(code) = code {
            self.add_required_obligations_with_code(span, def_id, args, move |_, _| code.clone());
        } else {
            self.add_required_obligations_for_hir(span, def_id, args, hir_id);
        }

        (Res::Def(def_kind, def_id), ty)
    }

    /// Resolves an associated value path into a base type and associated constant, or method
    /// resolution. The newly resolved definition is written into `type_dependent_defs`.
    #[instrument(level = "trace", skip(self), ret)]
    pub(crate) fn resolve_ty_and_res_fully_qualified_call(
        &self,
        qpath: &'tcx QPath<'tcx>,
        hir_id: HirId,
        span: Span,
    ) -> (Res, Option<LoweredTy<'tcx>>, &'tcx [hir::PathSegment<'tcx>]) {
        let (ty, qself, item_segment) = match *qpath {
            QPath::Resolved(ref opt_qself, path) => {
                return (
                    path.res,
                    opt_qself.as_ref().map(|qself| self.lower_ty(qself)),
                    path.segments,
                );
            }
            QPath::TypeRelative(ref qself, ref segment) => {
                // Don't use `self.lower_ty`, since this will register a WF obligation.
                // If we're trying to call a nonexistent method on a trait
                // (e.g. `MyTrait::missing_method`), then resolution will
                // give us a `QPath::TypeRelative` with a trait object as
                // `qself`. In that case, we want to avoid registering a WF obligation
                // for `dyn MyTrait`, since we don't actually need the trait
                // to be dyn-compatible.
                // We manually call `register_wf_obligation` in the success path
                // below.
                let ty = self.lowerer().lower_ty(qself);
                (LoweredTy::from_raw(self, span, ty), qself, segment)
            }
            QPath::LangItem(..) => {
                bug!("`resolve_ty_and_res_fully_qualified_call` called on `LangItem`")
            }
        };

        self.register_wf_obligation(
            ty.raw.into(),
            qself.span,
            ObligationCauseCode::WellFormed(None),
        );
        self.select_obligations_where_possible(|_| {});

        if let Some(&cached_result) = self.typeck_results.borrow().type_dependent_defs().get(hir_id)
        {
            // Return directly on cache hit. This is useful to avoid doubly reporting
            // errors with default match binding modes. See #44614.
            let def = cached_result.map_or(Res::Err, |(kind, def_id)| Res::Def(kind, def_id));
            return (def, Some(ty), slice::from_ref(&**item_segment));
        }
        let item_name = item_segment.ident;
        let result = self
            .resolve_fully_qualified_call(span, item_name, ty.normalized, qself.span, hir_id)
            .or_else(|error| {
                let guar = self
                    .dcx()
                    .span_delayed_bug(span, "method resolution should've emitted an error");
                let result = match error {
                    method::MethodError::PrivateMatch(kind, def_id, _) => Ok((kind, def_id)),
                    _ => Err(guar),
                };

                let trait_missing_method =
                    matches!(error, method::MethodError::NoMatch(_)) && ty.normalized.is_trait();
                self.report_method_error(
                    hir_id,
                    ty.normalized,
                    error,
                    Expectation::NoExpectation,
                    trait_missing_method && span.edition().at_least_rust_2021(), // emits missing method for trait only after edition 2021
                );

                result
            });

        // Write back the new resolution.
        self.write_resolution(hir_id, result);
        (
            result.map_or(Res::Err, |(kind, def_id)| Res::Def(kind, def_id)),
            Some(ty),
            slice::from_ref(&**item_segment),
        )
    }

    /// Given a `HirId`, return the `HirId` of the enclosing function and its `FnDecl`.
    pub(crate) fn get_fn_decl(
        &self,
        blk_id: HirId,
    ) -> Option<(LocalDefId, &'tcx hir::FnDecl<'tcx>)> {
        // Get enclosing Fn, if it is a function or a trait method, unless there's a `loop` or
        // `while` before reaching it, as block tail returns are not available in them.
        self.tcx.hir_get_fn_id_for_return_block(blk_id).and_then(|item_id| {
            match self.tcx.hir_node(item_id) {
                Node::Item(&hir::Item {
                    kind: hir::ItemKind::Fn { sig, .. }, owner_id, ..
                }) => Some((owner_id.def_id, sig.decl)),
                Node::TraitItem(&hir::TraitItem {
                    kind: hir::TraitItemKind::Fn(ref sig, ..),
                    owner_id,
                    ..
                }) => Some((owner_id.def_id, sig.decl)),
                Node::ImplItem(&hir::ImplItem {
                    kind: hir::ImplItemKind::Fn(ref sig, ..),
                    owner_id,
                    ..
                }) => Some((owner_id.def_id, sig.decl)),
                Node::Expr(&hir::Expr {
                    hir_id,
                    kind: hir::ExprKind::Closure(&hir::Closure { def_id, kind, fn_decl, .. }),
                    ..
                }) => {
                    match kind {
                        hir::ClosureKind::CoroutineClosure(_) => {
                            // FIXME(async_closures): Implement this.
                            return None;
                        }
                        hir::ClosureKind::Closure => Some((def_id, fn_decl)),
                        hir::ClosureKind::Coroutine(hir::CoroutineKind::Desugared(
                            _,
                            hir::CoroutineSource::Fn,
                        )) => {
                            let (sig, owner_id) = match self.tcx.parent_hir_node(hir_id) {
                                Node::Item(&hir::Item {
                                    kind: hir::ItemKind::Fn { ref sig, .. },
                                    owner_id,
                                    ..
                                }) => (sig, owner_id),
                                Node::TraitItem(&hir::TraitItem {
                                    kind: hir::TraitItemKind::Fn(ref sig, ..),
                                    owner_id,
                                    ..
                                }) => (sig, owner_id),
                                Node::ImplItem(&hir::ImplItem {
                                    kind: hir::ImplItemKind::Fn(ref sig, ..),
                                    owner_id,
                                    ..
                                }) => (sig, owner_id),
                                _ => return None,
                            };
                            Some((owner_id.def_id, sig.decl))
                        }
                        _ => None,
                    }
                }
                _ => None,
            }
        })
    }

    pub(crate) fn note_internal_mutation_in_method(
        &self,
        err: &mut Diag<'_>,
        expr: &hir::Expr<'_>,
        expected: Option<Ty<'tcx>>,
        found: Ty<'tcx>,
    ) {
        if found != self.tcx.types.unit {
            return;
        }

        let ExprKind::MethodCall(path_segment, rcvr, ..) = expr.kind else {
            return;
        };

        let rcvr_has_the_expected_type = self
            .typeck_results
            .borrow()
            .expr_ty_adjusted_opt(rcvr)
            .zip(expected)
            .is_some_and(|(ty, expected_ty)| expected_ty.peel_refs() == ty.peel_refs());

        let prev_call_mutates_and_returns_unit = || {
            self.typeck_results
                .borrow()
                .type_dependent_def_id(expr.hir_id)
                .map(|def_id| self.tcx.fn_sig(def_id).skip_binder().skip_binder())
                .and_then(|sig| sig.inputs_and_output.split_last())
                .is_some_and(|(output, inputs)| {
                    output.is_unit()
                        && inputs
                            .get(0)
                            .and_then(|self_ty| self_ty.ref_mutability())
                            .is_some_and(rustc_ast::Mutability::is_mut)
                })
        };

        if !(rcvr_has_the_expected_type || prev_call_mutates_and_returns_unit()) {
            return;
        }

        let mut sp = MultiSpan::from_span(path_segment.ident.span);
        sp.push_span_label(
            path_segment.ident.span,
            format!(
                "this call modifies {} in-place",
                match rcvr.kind {
                    ExprKind::Path(QPath::Resolved(
                        None,
                        hir::Path { segments: [segment], .. },
                    )) => format!("`{}`", segment.ident),
                    _ => "its receiver".to_string(),
                }
            ),
        );

        let modifies_rcvr_note =
            format!("method `{}` modifies its receiver in-place", path_segment.ident);
        if rcvr_has_the_expected_type {
            sp.push_span_label(
                rcvr.span,
                "you probably want to use this value after calling the method...",
            );
            err.span_note(sp, modifies_rcvr_note);
            err.note(format!("...instead of the `()` output of method `{}`", path_segment.ident));
        } else if let ExprKind::MethodCall(..) = rcvr.kind {
            err.span_note(
                sp,
                modifies_rcvr_note + ", it is not meant to be used in method chains.",
            );
        } else {
            err.span_note(sp, modifies_rcvr_note);
        }
    }

    // Instantiates the given path, which must refer to an item with the given
    // number of type parameters and type.
    #[instrument(skip(self, span), level = "debug")]
    pub(crate) fn instantiate_value_path(
        &self,
        segments: &'tcx [hir::PathSegment<'tcx>],
        self_ty: Option<LoweredTy<'tcx>>,
        res: Res,
        span: Span,
        path_span: Span,
        hir_id: HirId,
    ) -> (Ty<'tcx>, Res) {
        let tcx = self.tcx;

        let generic_segments = match res {
            Res::Local(_) | Res::SelfCtor(_) => vec![],
            Res::Def(kind, def_id) => self.lowerer().probe_generic_path_segments(
                segments,
                self_ty.map(|ty| ty.raw),
                kind,
                def_id,
                span,
            ),
            Res::Err => {
                return (
                    Ty::new_error(
                        tcx,
                        tcx.dcx().span_delayed_bug(span, "could not resolve path {:?}"),
                    ),
                    res,
                );
            }
            _ => bug!("instantiate_value_path on {:?}", res),
        };

        let mut user_self_ty = None;
        let mut is_alias_variant_ctor = false;
        let mut err_extend = GenericsArgsErrExtend::None;
        match res {
            Res::Def(DefKind::Ctor(CtorOf::Variant, _), _) if let Some(self_ty) = self_ty => {
                let adt_def = self_ty.normalized.ty_adt_def().unwrap();
                user_self_ty =
                    Some(UserSelfTy { impl_def_id: adt_def.did(), self_ty: self_ty.raw });
                is_alias_variant_ctor = true;
                err_extend = GenericsArgsErrExtend::DefVariant(segments);
            }
            Res::Def(DefKind::Ctor(CtorOf::Variant, _), _) => {
                err_extend = GenericsArgsErrExtend::DefVariant(segments);
            }
            Res::Def(DefKind::AssocFn | DefKind::AssocConst, def_id) => {
                let assoc_item = tcx.associated_item(def_id);
                let container = assoc_item.container;
                let container_id = assoc_item.container_id(tcx);
                debug!(?def_id, ?container, ?container_id);
                match container {
                    ty::AssocItemContainer::Trait => {
                        if let Err(e) = callee::check_legal_trait_for_method_call(
                            tcx,
                            path_span,
                            None,
                            span,
                            container_id,
                            self.body_id.to_def_id(),
                        ) {
                            self.set_tainted_by_errors(e);
                        }
                    }
                    ty::AssocItemContainer::Impl => {
                        if segments.len() == 1 {
                            // `<T>::assoc` will end up here, and so
                            // can `T::assoc`. If this came from an
                            // inherent impl, we need to record the
                            // `T` for posterity (see `UserSelfTy` for
                            // details).
                            let self_ty = self_ty.expect("UFCS sugared assoc missing Self").raw;
                            user_self_ty = Some(UserSelfTy { impl_def_id: container_id, self_ty });
                        }
                    }
                }
            }
            _ => {}
        }

        // Now that we have categorized what space the parameters for each
        // segment belong to, let's sort out the parameters that the user
        // provided (if any) into their appropriate spaces. We'll also report
        // errors if type parameters are provided in an inappropriate place.

        let indices: FxHashSet<_> =
            generic_segments.iter().map(|GenericPathSegment(_, index)| index).collect();
        let generics_err = self.lowerer().prohibit_generic_args(
            segments.iter().enumerate().filter_map(|(index, seg)| {
                if !indices.contains(&index) || is_alias_variant_ctor { Some(seg) } else { None }
            }),
            err_extend,
        );

        if let Res::Local(hid) = res {
            let ty = self.local_ty(span, hid);
            let ty = self.normalize(span, ty);
            return (ty, res);
        }

        if let Err(_) = generics_err {
            // Don't try to infer type parameters when prohibited generic arguments were given.
            user_self_ty = None;
        }

        // Now we have to compare the types that the user *actually*
        // provided against the types that were *expected*. If the user
        // did not provide any types, then we want to instantiate inference
        // variables. If the user provided some types, we may still need
        // to add defaults. If the user provided *too many* types, that's
        // a problem.

        let mut infer_args_for_err = None;

        let mut explicit_late_bound = ExplicitLateBound::No;
        for &GenericPathSegment(def_id, index) in &generic_segments {
            let seg = &segments[index];
            let generics = tcx.generics_of(def_id);

            // Argument-position `impl Trait` is treated as a normal generic
            // parameter internally, but we don't allow users to specify the
            // parameter's value explicitly, so we have to do some error-
            // checking here.
            let arg_count =
                check_generic_arg_count_for_call(self, def_id, generics, seg, IsMethodCall::No);

            if let ExplicitLateBound::Yes = arg_count.explicit_late_bound {
                explicit_late_bound = ExplicitLateBound::Yes;
            }

            if let Err(GenericArgCountMismatch { reported, .. }) = arg_count.correct {
                infer_args_for_err
                    .get_or_insert_with(|| (reported, FxHashSet::default()))
                    .1
                    .insert(index);
                self.set_tainted_by_errors(reported); // See issue #53251.
            }
        }

        let has_self = generic_segments
            .last()
            .is_some_and(|GenericPathSegment(def_id, _)| tcx.generics_of(*def_id).has_self);

        let (res, implicit_args) = if let Res::Def(DefKind::ConstParam, def) = res {
            // types of const parameters are somewhat special as they are part of
            // the same environment as the const parameter itself. this means that
            // unlike most paths `type-of(N)` can return a type naming parameters
            // introduced by the containing item, rather than provided through `N`.
            //
            // for example given `<T, const M: usize, const N: [T; M]>` and some
            // `let a = N;` expression. The path to `N` would wind up with no args
            // (as it has no args), but instantiating the early binder on `typeof(N)`
            // requires providing generic arguments for `[T, M, N]`.
            (res, Some(ty::GenericArgs::identity_for_item(tcx, tcx.parent(def))))
        } else if let Res::SelfCtor(impl_def_id) = res {
            let ty = LoweredTy::from_raw(
                self,
                span,
                tcx.at(span).type_of(impl_def_id).instantiate_identity(),
            );

            // Firstly, check that this SelfCtor even comes from the item we're currently
            // typechecking. This can happen because we never validated the resolution of
            // SelfCtors, and when we started doing so, we noticed regressions. After
            // sufficiently long time, we can remove this check and turn it into a hard
            // error in `validate_res_from_ribs` -- it's just difficult to tell whether the
            // self type has any generic types during rustc_resolve, which is what we use
            // to determine if this is a hard error or warning.
            if std::iter::successors(Some(self.body_id.to_def_id()), |def_id| {
                self.tcx.generics_of(def_id).parent
            })
            .all(|def_id| def_id != impl_def_id)
            {
                let sugg = ty.normalized.ty_adt_def().map(|def| errors::ReplaceWithName {
                    span: path_span,
                    name: self.tcx.item_name(def.did()).to_ident_string(),
                });
                if ty.raw.has_param() {
                    let guar = self.dcx().emit_err(errors::SelfCtorFromOuterItem {
                        span: path_span,
                        impl_span: tcx.def_span(impl_def_id),
                        sugg,
                    });
                    return (Ty::new_error(self.tcx, guar), res);
                } else {
                    self.tcx.emit_node_span_lint(
                        SELF_CONSTRUCTOR_FROM_OUTER_ITEM,
                        hir_id,
                        path_span,
                        errors::SelfCtorFromOuterItemLint {
                            impl_span: tcx.def_span(impl_def_id),
                            sugg,
                        },
                    );
                }
            }

            match ty.normalized.ty_adt_def() {
                Some(adt_def) if adt_def.has_ctor() => {
                    let (ctor_kind, ctor_def_id) = adt_def.non_enum_variant().ctor.unwrap();
                    // Check the visibility of the ctor.
                    let vis = tcx.visibility(ctor_def_id);
                    if !vis.is_accessible_from(tcx.parent_module(hir_id).to_def_id(), tcx) {
                        self.dcx()
                            .emit_err(CtorIsPrivate { span, def: tcx.def_path_str(adt_def.did()) });
                    }
                    let new_res = Res::Def(DefKind::Ctor(CtorOf::Struct, ctor_kind), ctor_def_id);
                    let user_args = Self::user_args_for_adt(ty);
                    user_self_ty = user_args.user_self_ty;
                    (new_res, Some(user_args.args))
                }
                _ => {
                    let mut err = self.dcx().struct_span_err(
                        span,
                        "the `Self` constructor can only be used with tuple or unit structs",
                    );
                    if let Some(adt_def) = ty.normalized.ty_adt_def() {
                        match adt_def.adt_kind() {
                            AdtKind::Enum => {
                                err.help("did you mean to use one of the enum's variants?");
                            }
                            AdtKind::Struct | AdtKind::Union => {
                                err.span_suggestion(
                                    span,
                                    "use curly brackets",
                                    "Self { /* fields */ }",
                                    Applicability::HasPlaceholders,
                                );
                            }
                        }
                    }
                    let reported = err.emit();
                    return (Ty::new_error(tcx, reported), res);
                }
            }
        } else {
            (res, None)
        };
        let def_id = res.def_id();

        let (correct, infer_args_for_err) = match infer_args_for_err {
            Some((reported, args)) => {
                (Err(GenericArgCountMismatch { reported, invalid_args: vec![] }), args)
            }
            None => (Ok(()), Default::default()),
        };

        let arg_count = GenericArgCountResult { explicit_late_bound, correct };

        struct CtorGenericArgsCtxt<'a, 'tcx> {
            fcx: &'a FnCtxt<'a, 'tcx>,
            span: Span,
            generic_segments: &'a [GenericPathSegment],
            infer_args_for_err: &'a FxHashSet<usize>,
            segments: &'tcx [hir::PathSegment<'tcx>],
        }
        impl<'a, 'tcx> GenericArgsLowerer<'a, 'tcx> for CtorGenericArgsCtxt<'a, 'tcx> {
            fn args_for_def_id(
                &mut self,
                def_id: DefId,
            ) -> (Option<&'a hir::GenericArgs<'tcx>>, bool) {
                if let Some(&GenericPathSegment(_, index)) =
                    self.generic_segments.iter().find(|&GenericPathSegment(did, _)| *did == def_id)
                {
                    // If we've encountered an `impl Trait`-related error, we're just
                    // going to infer the arguments for better error messages.
                    if !self.infer_args_for_err.contains(&index) {
                        // Check whether the user has provided generic arguments.
                        if let Some(data) = self.segments[index].args {
                            return (Some(data), self.segments[index].infer_args);
                        }
                    }
                    return (None, self.segments[index].infer_args);
                }

                (None, true)
            }

            fn provided_kind(
                &mut self,
                preceding_args: &[ty::GenericArg<'tcx>],
                param: &ty::GenericParamDef,
                arg: &GenericArg<'tcx>,
            ) -> ty::GenericArg<'tcx> {
                match (&param.kind, arg) {
                    (GenericParamDefKind::Lifetime, GenericArg::Lifetime(lt)) => self
                        .fcx
                        .lowerer()
                        .lower_lifetime(lt, RegionInferReason::Param(param))
                        .into(),
                    (GenericParamDefKind::Type { .. }, GenericArg::Type(ty)) => {
                        // We handle the ambig portions of `Ty` in match arm below
                        self.fcx.lower_ty(ty.as_unambig_ty()).raw.into()
                    }
                    (GenericParamDefKind::Type { .. }, GenericArg::Infer(inf)) => {
                        self.fcx.lower_ty(&inf.to_ty()).raw.into()
                    }
                    (GenericParamDefKind::Const { .. }, GenericArg::Const(ct)) => self
                        .fcx
                        // Ambiguous parts of `ConstArg` are handled in the match arms below
                        .lower_const_arg(
                            ct.as_unambig_ct(),
                            FeedConstTy::Param(param.def_id, preceding_args),
                        )
                        .into(),
                    (&GenericParamDefKind::Const { .. }, GenericArg::Infer(inf)) => {
                        self.fcx.ct_infer(Some(param), inf.span).into()
                    }
                    _ => unreachable!(),
                }
            }

            fn inferred_kind(
                &mut self,
                preceding_args: &[ty::GenericArg<'tcx>],
                param: &ty::GenericParamDef,
                infer_args: bool,
            ) -> ty::GenericArg<'tcx> {
                let tcx = self.fcx.tcx();
                if !infer_args && let Some(default) = param.default_value(tcx) {
                    // If we have a default, then it doesn't matter that we're not inferring
                    // the type/const arguments: We provide the default where any is missing.
                    return default.instantiate(tcx, preceding_args);
                }
                // If no type/const arguments were provided, we have to infer them.
                // This case also occurs as a result of some malformed input, e.g.,
                // a lifetime argument being given instead of a type/const parameter.
                // Using inference instead of `Error` gives better error messages.
                self.fcx.var_for_def(self.span, param)
            }
        }

        let args_raw = implicit_args.unwrap_or_else(|| {
            lower_generic_args(
                self,
                def_id,
                &[],
                has_self,
                self_ty.map(|s| s.raw),
                &arg_count,
                &mut CtorGenericArgsCtxt {
                    fcx: self,
                    span,
                    generic_segments: &generic_segments,
                    infer_args_for_err: &infer_args_for_err,
                    segments,
                },
            )
        });

        // First, store the "user args" for later.
        self.write_user_type_annotation_from_args(hir_id, def_id, args_raw, user_self_ty);

        // Normalize only after registering type annotations.
        let args = self.normalize(span, args_raw);

        self.add_required_obligations_for_hir(span, def_id, args, hir_id);

        // Instantiate the values for the type parameters into the type of
        // the referenced item.
        let ty = tcx.type_of(def_id);
        assert!(!args.has_escaping_bound_vars());
        assert!(!ty.skip_binder().has_escaping_bound_vars());
        let ty_instantiated = self.normalize(span, ty.instantiate(tcx, args));

        if let Some(UserSelfTy { impl_def_id, self_ty }) = user_self_ty {
            // In the case of `Foo<T>::method` and `<Foo<T>>::method`, if `method`
            // is inherent, there is no `Self` parameter; instead, the impl needs
            // type parameters, which we can infer by unifying the provided `Self`
            // with the instantiated impl type.
            // This also occurs for an enum variant on a type alias.
            let impl_ty = self.normalize(span, tcx.type_of(impl_def_id).instantiate(tcx, args));
            let self_ty = self.normalize(span, self_ty);
            match self.at(&self.misc(span), self.param_env).eq(
                DefineOpaqueTypes::Yes,
                impl_ty,
                self_ty,
            ) {
                Ok(ok) => self.register_infer_ok_obligations(ok),
                Err(_) => {
                    self.dcx().span_bug(
                        span,
                        format!(
                            "instantiate_value_path: (UFCS) {self_ty:?} was a subtype of {impl_ty:?} but now is not?",
                        ),
                    );
                }
            }
        }

        debug!("instantiate_value_path: type of {:?} is {:?}", hir_id, ty_instantiated);
        self.write_args(hir_id, args);

        (ty_instantiated, res)
    }

    /// Add all the obligations that are required, instantiated and normalized appropriately.
    pub(crate) fn add_required_obligations_for_hir(
        &self,
        span: Span,
        def_id: DefId,
        args: GenericArgsRef<'tcx>,
        hir_id: HirId,
    ) {
        self.add_required_obligations_with_code(span, def_id, args, |idx, span| {
            ObligationCauseCode::WhereClauseInExpr(def_id, span, hir_id, idx)
        })
    }

    #[instrument(level = "debug", skip(self, code, span, args))]
    fn add_required_obligations_with_code(
        &self,
        span: Span,
        def_id: DefId,
        args: GenericArgsRef<'tcx>,
        code: impl Fn(usize, Span) -> ObligationCauseCode<'tcx>,
    ) {
        let param_env = self.param_env;

        let bounds = self.instantiate_bounds(span, def_id, args);

        for obligation in traits::predicates_for_generics(
            |idx, predicate_span| self.cause(span, code(idx, predicate_span)),
            param_env,
            bounds,
        ) {
            self.register_predicate(obligation);
        }
    }

    /// Try to resolve `ty` to a structural type, normalizing aliases.
    ///
    /// In case there is still ambiguity, the returned type may be an inference
    /// variable. This is different from `structurally_resolve_type` which errors
    /// in this case.
    #[instrument(level = "debug", skip(self, sp), ret)]
    pub(crate) fn try_structurally_resolve_type(&self, sp: Span, ty: Ty<'tcx>) -> Ty<'tcx> {
        if self.next_trait_solver()
            && let ty::Alias(..) = ty.kind()
        {
            // We need to use a separate variable here as otherwise the temporary for
            // `self.fulfillment_cx.borrow_mut()` is alive in the `Err` branch, resulting
            // in a reentrant borrow, causing an ICE.
            let result = self
                .at(&self.misc(sp), self.param_env)
                .structurally_normalize_ty(ty, &mut **self.fulfillment_cx.borrow_mut());
            match result {
                Ok(normalized_ty) => normalized_ty,
                Err(errors) => {
                    let guar = self.err_ctxt().report_fulfillment_errors(errors);
                    return Ty::new_error(self.tcx, guar);
                }
            }
        } else {
            self.resolve_vars_with_obligations(ty)
        }
    }

    #[instrument(level = "debug", skip(self, sp), ret)]
    pub(crate) fn try_structurally_resolve_const(
        &self,
        sp: Span,
        ct: ty::Const<'tcx>,
    ) -> ty::Const<'tcx> {
        let ct = self.resolve_vars_with_obligations(ct);

        if self.next_trait_solver()
            && let ty::ConstKind::Unevaluated(..) = ct.kind()
        {
            // We need to use a separate variable here as otherwise the temporary for
            // `self.fulfillment_cx.borrow_mut()` is alive in the `Err` branch, resulting
            // in a reentrant borrow, causing an ICE.
            let result = self
                .at(&self.misc(sp), self.param_env)
                .structurally_normalize_const(ct, &mut **self.fulfillment_cx.borrow_mut());
            match result {
                Ok(normalized_ct) => normalized_ct,
                Err(errors) => {
                    let guar = self.err_ctxt().report_fulfillment_errors(errors);
                    return ty::Const::new_error(self.tcx, guar);
                }
            }
        } else if self.tcx.features().generic_const_exprs() {
            rustc_trait_selection::traits::evaluate_const(&self.infcx, ct, self.param_env)
        } else {
            ct
        }
    }

    /// Resolves `ty` by a single level if `ty` is a type variable.
    ///
    /// When the new solver is enabled, this will also attempt to normalize
    /// the type if it's a projection (note that it will not deeply normalize
    /// projections within the type, just the outermost layer of the type).
    ///
    /// If no resolution is possible, then an error is reported.
    /// Numeric inference variables may be left unresolved.
    pub(crate) fn structurally_resolve_type(&self, sp: Span, ty: Ty<'tcx>) -> Ty<'tcx> {
        let ty = self.try_structurally_resolve_type(sp, ty);

        if !ty.is_ty_var() {
            ty
        } else {
            let e = self.tainted_by_errors().unwrap_or_else(|| {
                self.err_ctxt()
                    .emit_inference_failure_err(
                        self.body_id,
                        sp,
                        ty.into(),
                        TypeAnnotationNeeded::E0282,
                        true,
                    )
                    .emit()
            });
            let err = Ty::new_error(self.tcx, e);
            self.demand_suptype(sp, err, ty);
            err
        }
    }

    pub(crate) fn structurally_resolve_const(
        &self,
        sp: Span,
        ct: ty::Const<'tcx>,
    ) -> ty::Const<'tcx> {
        let ct = self.try_structurally_resolve_const(sp, ct);

        if !ct.is_ct_infer() {
            ct
        } else {
            let e = self.tainted_by_errors().unwrap_or_else(|| {
                self.err_ctxt()
                    .emit_inference_failure_err(
                        self.body_id,
                        sp,
                        ct.into(),
                        TypeAnnotationNeeded::E0282,
                        true,
                    )
                    .emit()
            });
            // FIXME: Infer `?ct = {const error}`?
            ty::Const::new_error(self.tcx, e)
        }
    }

    pub(crate) fn with_breakable_ctxt<F: FnOnce() -> R, R>(
        &self,
        id: HirId,
        ctxt: BreakableCtxt<'tcx>,
        f: F,
    ) -> (BreakableCtxt<'tcx>, R) {
        let index;
        {
            let mut enclosing_breakables = self.enclosing_breakables.borrow_mut();
            index = enclosing_breakables.stack.len();
            enclosing_breakables.by_id.insert(id, index);
            enclosing_breakables.stack.push(ctxt);
        }
        let result = f();
        let ctxt = {
            let mut enclosing_breakables = self.enclosing_breakables.borrow_mut();
            debug_assert!(enclosing_breakables.stack.len() == index + 1);
            // FIXME(#120456) - is `swap_remove` correct?
            enclosing_breakables.by_id.swap_remove(&id).expect("missing breakable context");
            enclosing_breakables.stack.pop().expect("missing breakable context")
        };
        (ctxt, result)
    }

    /// Instantiate a QueryResponse in a probe context, without a
    /// good ObligationCause.
    pub(crate) fn probe_instantiate_query_response(
        &self,
        span: Span,
        original_values: &OriginalQueryValues<'tcx>,
        query_result: &Canonical<'tcx, QueryResponse<'tcx, Ty<'tcx>>>,
    ) -> InferResult<'tcx, Ty<'tcx>> {
        self.instantiate_query_response_and_region_obligations(
            &self.misc(span),
            self.param_env,
            original_values,
            query_result,
        )
    }

    /// Returns `true` if an expression is contained inside the LHS of an assignment expression.
    pub(crate) fn expr_in_place(&self, mut expr_id: HirId) -> bool {
        let mut contained_in_place = false;

        while let hir::Node::Expr(parent_expr) = self.tcx.parent_hir_node(expr_id) {
            match &parent_expr.kind {
                hir::ExprKind::Assign(lhs, ..) | hir::ExprKind::AssignOp(_, lhs, ..) => {
                    if lhs.hir_id == expr_id {
                        contained_in_place = true;
                        break;
                    }
                }
                _ => (),
            }
            expr_id = parent_expr.hir_id;
        }

        contained_in_place
    }
}
