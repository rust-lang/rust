use crate::callee::{self, DeferredCallResolution};
use crate::errors::CtorIsPrivate;
use crate::method::{self, MethodCallee, SelfSource};
use crate::rvalue_scopes;
use crate::{BreakableCtxt, Diverges, Expectation, FnCtxt, RawTy};
use rustc_data_structures::captures::Captures;
use rustc_data_structures::fx::FxHashSet;
use rustc_errors::{Applicability, Diagnostic, ErrorGuaranteed, MultiSpan, StashKey};
use rustc_hir as hir;
use rustc_hir::def::{CtorOf, DefKind, Res};
use rustc_hir::def_id::DefId;
use rustc_hir::lang_items::LangItem;
use rustc_hir::{ExprKind, GenericArg, Node, QPath};
use rustc_hir_analysis::astconv::generics::{
    check_generic_arg_count_for_call, create_args_for_parent_generic_args,
};
use rustc_hir_analysis::astconv::{
    AstConv, CreateSubstsForGenericArgsCtxt, ExplicitLateBound, GenericArgCountMismatch,
    GenericArgCountResult, IsMethodCall, PathSeg,
};
use rustc_infer::infer::canonical::{Canonical, OriginalQueryValues, QueryResponse};
use rustc_infer::infer::error_reporting::TypeAnnotationNeeded::E0282;
use rustc_infer::infer::{DefineOpaqueTypes, InferResult};
use rustc_middle::ty::adjustment::{Adjust, Adjustment, AutoBorrow, AutoBorrowMutability};
use rustc_middle::ty::error::TypeError;
use rustc_middle::ty::fold::TypeFoldable;
use rustc_middle::ty::visit::{TypeVisitable, TypeVisitableExt};
use rustc_middle::ty::{
    self, AdtKind, CanonicalUserType, GenericParamDefKind, Ty, TyCtxt, UserType,
};
use rustc_middle::ty::{GenericArgKind, GenericArgsRef, UserArgs, UserSelfTy};
use rustc_session::lint;
use rustc_span::def_id::LocalDefId;
use rustc_span::hygiene::DesugaringKind;
use rustc_span::symbol::{kw, sym, Ident};
use rustc_span::Span;
use rustc_target::abi::FieldIdx;
use rustc_trait_selection::traits::error_reporting::TypeErrCtxtExt as _;
use rustc_trait_selection::traits::{
    self, NormalizeExt, ObligationCauseCode, ObligationCtxt, StructurallyNormalizeExt,
};

use std::collections::hash_map::Entry;
use std::slice;

impl<'a, 'tcx> FnCtxt<'a, 'tcx> {
    /// Produces warning on the given node, if the current point in the
    /// function is unreachable, and there hasn't been another warning.
    pub(in super::super) fn warn_if_unreachable(&self, id: hir::HirId, span: Span, kind: &str) {
        // FIXME: Combine these two 'if' expressions into one once
        // let chains are implemented
        if let Diverges::Always { span: orig_span, custom_note } = self.diverges.get() {
            // If span arose from a desugaring of `if` or `while`, then it is the condition itself,
            // which diverges, that we are about to lint on. This gives suboptimal diagnostics.
            // Instead, stop here so that the `if`- or `while`-expression's block is linted instead.
            if !span.is_desugaring(DesugaringKind::CondTemporary)
                && !span.is_desugaring(DesugaringKind::Async)
                && !orig_span.is_desugaring(DesugaringKind::Await)
            {
                self.diverges.set(Diverges::WarnedAlways);

                debug!("warn_if_unreachable: id={:?} span={:?} kind={}", id, span, kind);

                let msg = format!("unreachable {}", kind);
                self.tcx().struct_span_lint_hir(
                    lint::builtin::UNREACHABLE_CODE,
                    id,
                    span,
                    msg.clone(),
                    |lint| {
                        lint.span_label(span, msg).span_label(
                            orig_span,
                            custom_note
                                .unwrap_or("any code following this expression is unreachable"),
                        )
                    },
                )
            }
        }
    }

    /// Resolves type and const variables in `ty` if possible. Unlike the infcx
    /// version (resolve_vars_if_possible), this version will
    /// also select obligations if it seems useful, in an effort
    /// to get more type information.
    // FIXME(-Ztrait-solver=next): A lot of the calls to this method should
    // probably be `try_structurally_resolve_type` or `structurally_resolve_type` instead.
    pub(in super::super) fn resolve_vars_with_obligations(&self, ty: Ty<'tcx>) -> Ty<'tcx> {
        self.resolve_vars_with_obligations_and_mutate_fulfillment(ty, |_| {})
    }

    #[instrument(skip(self, mutate_fulfillment_errors), level = "debug", ret)]
    pub(in super::super) fn resolve_vars_with_obligations_and_mutate_fulfillment(
        &self,
        mut ty: Ty<'tcx>,
        mutate_fulfillment_errors: impl Fn(&mut Vec<traits::FulfillmentError<'tcx>>),
    ) -> Ty<'tcx> {
        // No Infer()? Nothing needs doing.
        if !ty.has_non_region_infer() {
            debug!("no inference var, nothing needs doing");
            return ty;
        }

        // If `ty` is a type variable, see whether we already know what it is.
        ty = self.resolve_vars_if_possible(ty);
        if !ty.has_non_region_infer() {
            debug!(?ty);
            return ty;
        }

        // If not, try resolving pending obligations as much as
        // possible. This can help substantially when there are
        // indirect dependencies that don't seem worth tracking
        // precisely.
        self.select_obligations_where_possible(mutate_fulfillment_errors);
        self.resolve_vars_if_possible(ty)
    }

    pub(in super::super) fn record_deferred_call_resolution(
        &self,
        closure_def_id: LocalDefId,
        r: DeferredCallResolution<'tcx>,
    ) {
        let mut deferred_call_resolutions = self.deferred_call_resolutions.borrow_mut();
        deferred_call_resolutions.entry(closure_def_id).or_default().push(r);
    }

    pub(in super::super) fn remove_deferred_call_resolutions(
        &self,
        closure_def_id: LocalDefId,
    ) -> Vec<DeferredCallResolution<'tcx>> {
        let mut deferred_call_resolutions = self.deferred_call_resolutions.borrow_mut();
        deferred_call_resolutions.remove(&closure_def_id).unwrap_or_default()
    }

    pub fn tag(&self) -> String {
        format!("{:p}", self)
    }

    pub fn local_ty(&self, span: Span, nid: hir::HirId) -> Ty<'tcx> {
        self.locals.borrow().get(&nid).cloned().unwrap_or_else(|| {
            span_bug!(span, "no type for local variable {}", self.tcx.hir().node_to_string(nid))
        })
    }

    #[inline]
    pub fn write_ty(&self, id: hir::HirId, ty: Ty<'tcx>) {
        debug!("write_ty({:?}, {:?}) in fcx {}", id, self.resolve_vars_if_possible(ty), self.tag());
        self.typeck_results.borrow_mut().node_types_mut().insert(id, ty);

        if let Err(e) = ty.error_reported() {
            self.set_tainted_by_errors(e);
        }
    }

    pub fn write_field_index(&self, hir_id: hir::HirId, index: FieldIdx) {
        self.typeck_results.borrow_mut().field_indices_mut().insert(hir_id, index);
    }

    #[instrument(level = "debug", skip(self))]
    pub(in super::super) fn write_resolution(
        &self,
        hir_id: hir::HirId,
        r: Result<(DefKind, DefId), ErrorGuaranteed>,
    ) {
        self.typeck_results.borrow_mut().type_dependent_defs_mut().insert(hir_id, r);
    }

    #[instrument(level = "debug", skip(self))]
    pub fn write_method_call(&self, hir_id: hir::HirId, method: MethodCallee<'tcx>) {
        self.write_resolution(hir_id, Ok((DefKind::AssocFn, method.def_id)));
        self.write_args(hir_id, method.args);
    }

    pub fn write_args(&self, node_id: hir::HirId, args: GenericArgsRef<'tcx>) {
        if !args.is_empty() {
            debug!("write_args({:?}, {:?}) in fcx {}", node_id, args, self.tag());

            self.typeck_results.borrow_mut().node_args_mut().insert(node_id, args);
        }
    }

    /// Given the args that we just converted from the HIR, try to
    /// canonicalize them and store them as user-given substitutions
    /// (i.e., substitutions that must be respected by the NLL check).
    ///
    /// This should be invoked **before any unifications have
    /// occurred**, so that annotations like `Vec<_>` are preserved
    /// properly.
    #[instrument(skip(self), level = "debug")]
    pub fn write_user_type_annotation_from_args(
        &self,
        hir_id: hir::HirId,
        def_id: DefId,
        args: GenericArgsRef<'tcx>,
        user_self_ty: Option<UserSelfTy<'tcx>>,
    ) {
        debug!("fcx {}", self.tag());

        if Self::can_contain_user_lifetime_bounds((args, user_self_ty)) {
            let canonicalized = self.canonicalize_user_type_annotation(UserType::TypeOf(
                def_id,
                UserArgs { args, user_self_ty },
            ));
            debug!(?canonicalized);
            self.write_user_type_annotation(hir_id, canonicalized);
        }
    }

    #[instrument(skip(self), level = "debug")]
    pub fn write_user_type_annotation(
        &self,
        hir_id: hir::HirId,
        canonical_user_type_annotation: CanonicalUserType<'tcx>,
    ) {
        debug!("fcx {}", self.tag());

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
    pub fn apply_adjustments(&self, expr: &hir::Expr<'_>, adj: Vec<Adjustment<'tcx>>) {
        debug!("expr = {:#?}", expr);

        if adj.is_empty() {
            return;
        }

        for a in &adj {
            if let Adjust::NeverToAny = a.kind {
                if a.target.is_ty_var() {
                    self.diverging_type_vars.borrow_mut().insert(a.target);
                    debug!("apply_adjustments: adding `{:?}` as diverging type var", a.target);
                }
            }
        }

        let autoborrow_mut = adj.iter().any(|adj| {
            matches!(
                adj,
                &Adjustment {
                    kind: Adjust::Borrow(AutoBorrow::Ref(_, AutoBorrowMutability::Mut { .. })),
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
                match (&entry.get()[..], &adj[..]) {
                    // Applying any adjustment on top of a NeverToAny
                    // is a valid NeverToAny adjustment, because it can't
                    // be reached.
                    (&[Adjustment { kind: Adjust::NeverToAny, .. }], _) => return,
                    (
                        &[
                            Adjustment { kind: Adjust::Deref(_), .. },
                            Adjustment { kind: Adjust::Borrow(AutoBorrow::Ref(..)), .. },
                        ],
                        &[
                            Adjustment { kind: Adjust::Deref(_), .. },
                            .., // Any following adjustments are allowed.
                        ],
                    ) => {
                        // A reborrow has no effect before a dereference.
                    }
                    // FIXME: currently we never try to compose autoderefs
                    // and ReifyFnPointer/UnsafeFnPointer, but we could.
                    _ => {
                        self.tcx.sess.delay_span_bug(
                            expr.span,
                            format!(
                                "while adjusting {:?}, can't compose {:?} and {:?}",
                                expr,
                                entry.get(),
                                adj
                            ),
                        );
                    }
                }
                *entry.get_mut() = adj;
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
    pub(in super::super) fn instantiate_bounds(
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

    pub(in super::super) fn normalize<T>(&self, span: Span, value: T) -> T
    where
        T: TypeFoldable<TyCtxt<'tcx>>,
    {
        self.register_infer_ok_obligations(
            self.at(&self.misc(span), self.param_env).normalize(value),
        )
    }

    pub fn require_type_meets(
        &self,
        ty: Ty<'tcx>,
        span: Span,
        code: traits::ObligationCauseCode<'tcx>,
        def_id: DefId,
    ) {
        self.register_bound(ty, def_id, traits::ObligationCause::new(span, self.body_id, code));
    }

    pub fn require_type_is_sized(
        &self,
        ty: Ty<'tcx>,
        span: Span,
        code: traits::ObligationCauseCode<'tcx>,
    ) {
        if !ty.references_error() {
            let lang_item = self.tcx.require_lang_item(LangItem::Sized, None);
            self.require_type_meets(ty, span, code, lang_item);
        }
    }

    pub fn require_type_is_sized_deferred(
        &self,
        ty: Ty<'tcx>,
        span: Span,
        code: traits::ObligationCauseCode<'tcx>,
    ) {
        if !ty.references_error() {
            self.deferred_sized_obligations.borrow_mut().push((ty, span, code));
        }
    }

    pub fn register_bound(
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

    pub fn handle_raw_ty(&self, span: Span, ty: Ty<'tcx>) -> RawTy<'tcx> {
        RawTy { raw: ty, normalized: self.normalize(span, ty) }
    }

    pub fn to_ty(&self, ast_t: &hir::Ty<'_>) -> RawTy<'tcx> {
        let t = self.astconv().ast_ty_to_ty(ast_t);
        self.register_wf_obligation(t.into(), ast_t.span, traits::WellFormed(None));
        self.handle_raw_ty(ast_t.span, t)
    }

    pub fn to_ty_saving_user_provided_ty(&self, ast_ty: &hir::Ty<'_>) -> Ty<'tcx> {
        let ty = self.to_ty(ast_ty);
        debug!("to_ty_saving_user_provided_ty: ty={:?}", ty);

        if Self::can_contain_user_lifetime_bounds(ty.raw) {
            let c_ty = self.canonicalize_response(UserType::Ty(ty.raw));
            debug!("to_ty_saving_user_provided_ty: c_ty={:?}", c_ty);
            self.typeck_results.borrow_mut().user_provided_types_mut().insert(ast_ty.hir_id, c_ty);
        }

        ty.normalized
    }

    pub(super) fn user_args_for_adt(ty: RawTy<'tcx>) -> UserArgs<'tcx> {
        match (ty.raw.kind(), ty.normalized.kind()) {
            (ty::Adt(_, args), _) => UserArgs { args, user_self_ty: None },
            (_, ty::Adt(adt, args)) => UserArgs {
                args,
                user_self_ty: Some(UserSelfTy { impl_def_id: adt.did(), self_ty: ty.raw }),
            },
            _ => bug!("non-adt type {:?}", ty),
        }
    }

    pub fn array_length_to_const(&self, length: &hir::ArrayLen) -> ty::Const<'tcx> {
        match length {
            &hir::ArrayLen::Infer(_, span) => self.ct_infer(self.tcx.types.usize, None, span),
            hir::ArrayLen::Body(anon_const) => {
                let span = self.tcx.def_span(anon_const.def_id);
                let c = ty::Const::from_anon_const(self.tcx, anon_const.def_id);
                self.register_wf_obligation(c.into(), span, ObligationCauseCode::WellFormed(None));
                self.normalize(span, c)
            }
        }
    }

    pub fn const_arg_to_const(
        &self,
        ast_c: &hir::AnonConst,
        param_def_id: DefId,
    ) -> ty::Const<'tcx> {
        let did = ast_c.def_id;
        self.tcx.feed_anon_const_type(did, self.tcx.type_of(param_def_id));
        let c = ty::Const::from_anon_const(self.tcx, did);
        self.register_wf_obligation(
            c.into(),
            self.tcx.hir().span(ast_c.hir_id),
            ObligationCauseCode::WellFormed(None),
        );
        c
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
        t.has_free_regions() || t.has_projections() || t.has_infer_types()
    }

    pub fn node_ty(&self, id: hir::HirId) -> Ty<'tcx> {
        match self.typeck_results.borrow().node_types().get(id) {
            Some(&t) => t,
            None if let Some(e) = self.tainted_by_errors() => Ty::new_error(self.tcx,e),
            None => {
                bug!(
                    "no type for node {} in fcx {}",
                    self.tcx.hir().node_to_string(id),
                    self.tag()
                );
            }
        }
    }

    pub fn node_ty_opt(&self, id: hir::HirId) -> Option<Ty<'tcx>> {
        match self.typeck_results.borrow().node_types().get(id) {
            Some(&t) => Some(t),
            None if let Some(e) = self.tainted_by_errors() => Some(Ty::new_error(self.tcx,e)),
            None => None,
        }
    }

    /// Registers an obligation for checking later, during regionck, that `arg` is well-formed.
    pub fn register_wf_obligation(
        &self,
        arg: ty::GenericArg<'tcx>,
        span: Span,
        code: traits::ObligationCauseCode<'tcx>,
    ) {
        // WF obligations never themselves fail, so no real need to give a detailed cause:
        let cause = traits::ObligationCause::new(span, self.body_id, code);
        self.register_predicate(traits::Obligation::new(
            self.tcx,
            cause,
            self.param_env,
            ty::Binder::dummy(ty::PredicateKind::Clause(ty::ClauseKind::WellFormed(arg))),
        ));
    }

    /// Registers obligations that all `args` are well-formed.
    pub fn add_wf_bounds(&self, args: GenericArgsRef<'tcx>, expr: &hir::Expr<'_>) {
        for arg in args.iter().filter(|arg| {
            matches!(arg.unpack(), GenericArgKind::Type(..) | GenericArgKind::Const(..))
        }) {
            self.register_wf_obligation(arg, expr.span, traits::WellFormed(None));
        }
    }

    // FIXME(arielb1): use this instead of field.ty everywhere
    // Only for fields! Returns <none> for methods>
    // Indifferent to privacy flags
    pub fn field_ty(
        &self,
        span: Span,
        field: &'tcx ty::FieldDef,
        args: GenericArgsRef<'tcx>,
    ) -> Ty<'tcx> {
        self.normalize(span, field.ty(self.tcx, args))
    }

    pub(in super::super) fn resolve_rvalue_scopes(&self, def_id: DefId) {
        let scope_tree = self.tcx.region_scope_tree(def_id);
        let rvalue_scopes = { rvalue_scopes::resolve_rvalue_scopes(self, &scope_tree, def_id) };
        let mut typeck_results = self.inh.typeck_results.borrow_mut();
        typeck_results.rvalue_scopes = rvalue_scopes;
    }

    pub(in super::super) fn resolve_generator_interiors(&self, def_id: DefId) {
        if self.tcx.sess.opts.unstable_opts.drop_tracking_mir {
            self.save_generator_interior_predicates(def_id);
            return;
        }

        self.select_obligations_where_possible(|_| {});

        let mut generators = self.deferred_generator_interiors.borrow_mut();
        for (_, body_id, interior, kind) in generators.drain(..) {
            crate::generator_interior::resolve_interior(self, def_id, body_id, interior, kind);
            self.select_obligations_where_possible(|_| {});
        }
    }

    /// Unify the inference variables corresponding to generator witnesses, and save all the
    /// predicates that were stalled on those inference variables.
    ///
    /// This process allows to conservatively save all predicates that do depend on the generator
    /// interior types, for later processing by `check_generator_obligations`.
    ///
    /// We must not attempt to select obligations after this method has run, or risk query cycle
    /// ICE.
    #[instrument(level = "debug", skip(self))]
    fn save_generator_interior_predicates(&self, def_id: DefId) {
        // Try selecting all obligations that are not blocked on inference variables.
        // Once we start unifying generator witnesses, trying to select obligations on them will
        // trigger query cycle ICEs, as doing so requires MIR.
        self.select_obligations_where_possible(|_| {});

        let generators = std::mem::take(&mut *self.deferred_generator_interiors.borrow_mut());
        debug!(?generators);

        for &(expr_def_id, body_id, interior, _) in generators.iter() {
            debug!(?expr_def_id);

            // Create the `GeneratorWitness` type that we will unify with `interior`.
            let args = ty::GenericArgs::identity_for_item(
                self.tcx,
                self.tcx.typeck_root_def_id(expr_def_id.to_def_id()),
            );
            let witness = Ty::new_generator_witness_mir(self.tcx, expr_def_id.to_def_id(), args);

            // Unify `interior` with `witness` and collect all the resulting obligations.
            let span = self.tcx.hir().body(body_id).value.span;
            let ok = self
                .at(&self.misc(span), self.param_env)
                .eq(DefineOpaqueTypes::No, interior, witness)
                .expect("Failed to unify generator interior type");
            let mut obligations = ok.obligations;

            // Also collect the obligations that were unstalled by this unification.
            obligations
                .extend(self.fulfillment_cx.borrow_mut().drain_unstalled_obligations(&self.infcx));

            let obligations = obligations.into_iter().map(|o| (o.predicate, o.cause)).collect();
            debug!(?obligations);
            self.typeck_results
                .borrow_mut()
                .generator_interior_predicates
                .insert(expr_def_id, obligations);
        }
    }

    #[instrument(skip(self), level = "debug")]
    pub(in super::super) fn report_ambiguity_errors(&self) {
        let mut errors = self.fulfillment_cx.borrow_mut().collect_remaining_errors(self);

        if !errors.is_empty() {
            self.adjust_fulfillment_errors_for_expr_obligation(&mut errors);
            self.err_ctxt().report_fulfillment_errors(&errors);
        }
    }

    /// Select as many obligations as we can at present.
    pub(in super::super) fn select_obligations_where_possible(
        &self,
        mutate_fulfillment_errors: impl Fn(&mut Vec<traits::FulfillmentError<'tcx>>),
    ) {
        let mut result = self.fulfillment_cx.borrow_mut().select_where_possible(self);
        if !result.is_empty() {
            mutate_fulfillment_errors(&mut result);
            self.adjust_fulfillment_errors_for_expr_obligation(&mut result);
            self.err_ctxt().report_fulfillment_errors(&result);
        }
    }

    /// For the overloaded place expressions (`*x`, `x[3]`), the trait
    /// returns a type of `&T`, but the actual type we assign to the
    /// *expression* is `T`. So this function just peels off the return
    /// type by one layer to yield `T`.
    pub(in super::super) fn make_overloaded_place_return_type(
        &self,
        method: MethodCallee<'tcx>,
    ) -> ty::TypeAndMut<'tcx> {
        // extract method return type, which will be &T;
        let ret_ty = method.sig.output();

        // method returns &T, but the type as visible to user is T, so deref
        ret_ty.builtin_deref(true).unwrap()
    }

    #[instrument(skip(self), level = "debug")]
    fn self_type_matches_expected_vid(&self, self_ty: Ty<'tcx>, expected_vid: ty::TyVid) -> bool {
        let self_ty = self.shallow_resolve(self_ty);
        debug!(?self_ty);

        match *self_ty.kind() {
            ty::Infer(ty::TyVar(found_vid)) => {
                // FIXME: consider using `sub_root_var` here so we
                // can see through subtyping.
                let found_vid = self.root_var(found_vid);
                debug!("self_type_matches_expected_vid - found_vid={:?}", found_vid);
                expected_vid == found_vid
            }
            _ => false,
        }
    }

    #[instrument(skip(self), level = "debug")]
    pub(in super::super) fn obligations_for_self_ty<'b>(
        &'b self,
        self_ty: ty::TyVid,
    ) -> impl DoubleEndedIterator<Item = traits::PredicateObligation<'tcx>> + Captures<'tcx> + 'b
    {
        // FIXME: consider using `sub_root_var` here so we
        // can see through subtyping.
        let ty_var_root = self.root_var(self_ty);
        trace!("pending_obligations = {:#?}", self.fulfillment_cx.borrow().pending_obligations());

        self.fulfillment_cx.borrow().pending_obligations().into_iter().filter_map(
            move |obligation| match &obligation.predicate.kind().skip_binder() {
                ty::PredicateKind::Clause(ty::ClauseKind::Projection(data))
                    if self.self_type_matches_expected_vid(
                        data.projection_ty.self_ty(),
                        ty_var_root,
                    ) =>
                {
                    Some(obligation)
                }
                ty::PredicateKind::Clause(ty::ClauseKind::Trait(data))
                    if self.self_type_matches_expected_vid(data.self_ty(), ty_var_root) =>
                {
                    Some(obligation)
                }

                ty::PredicateKind::Clause(ty::ClauseKind::Trait(..))
                | ty::PredicateKind::Clause(ty::ClauseKind::Projection(..))
                | ty::PredicateKind::Clause(ty::ClauseKind::ConstArgHasType(..))
                | ty::PredicateKind::Subtype(..)
                | ty::PredicateKind::Coerce(..)
                | ty::PredicateKind::Clause(ty::ClauseKind::RegionOutlives(..))
                | ty::PredicateKind::Clause(ty::ClauseKind::TypeOutlives(..))
                | ty::PredicateKind::Clause(ty::ClauseKind::WellFormed(..))
                | ty::PredicateKind::ObjectSafe(..)
                | ty::PredicateKind::AliasRelate(..)
                | ty::PredicateKind::Clause(ty::ClauseKind::ConstEvaluatable(..))
                | ty::PredicateKind::ConstEquate(..)
                // N.B., this predicate is created by breaking down a
                // `ClosureType: FnFoo()` predicate, where
                // `ClosureType` represents some `Closure`. It can't
                // possibly be referring to the current closure,
                // because we haven't produced the `Closure` for
                // this closure yet; this is exactly why the other
                // code is looking for a self type of an unresolved
                // inference variable.
                | ty::PredicateKind::ClosureKind(..)
                | ty::PredicateKind::Ambiguous
                 => None,
            },
        )
    }

    pub(in super::super) fn type_var_is_sized(&self, self_ty: ty::TyVid) -> bool {
        let sized_did = self.tcx.lang_items().sized_trait();
        self.obligations_for_self_ty(self_ty).any(|obligation| {
            match obligation.predicate.kind().skip_binder() {
                ty::PredicateKind::Clause(ty::ClauseKind::Trait(data)) => {
                    Some(data.def_id()) == sized_did
                }
                _ => false,
            }
        })
    }

    pub(in super::super) fn err_args(&self, len: usize) -> Vec<Ty<'tcx>> {
        let ty_error = Ty::new_misc_error(self.tcx);
        vec![ty_error; len]
    }

    /// Unifies the output type with the expected type early, for more coercions
    /// and forward type information on the input expressions.
    #[instrument(skip(self, call_span), level = "debug")]
    pub(in super::super) fn expected_inputs_for_expected_output(
        &self,
        call_span: Span,
        expected_ret: Expectation<'tcx>,
        formal_ret: Ty<'tcx>,
        formal_args: &[Ty<'tcx>],
    ) -> Option<Vec<Ty<'tcx>>> {
        let formal_ret = self.resolve_vars_with_obligations(formal_ret);
        let ret_ty = expected_ret.only_has_type(self)?;

        // HACK(oli-obk): This is a hack to keep RPIT and TAIT in sync wrt their behaviour.
        // Without it, the inference
        // variable will get instantiated with the opaque type. The inference variable often
        // has various helpful obligations registered for it that help closures figure out their
        // signature. If we infer the inference var to the opaque type, the closure won't be able
        // to find those obligations anymore, and it can't necessarily find them from the opaque
        // type itself. We could be more powerful with inference if we *combined* the obligations
        // so that we got both the obligations from the opaque type and the ones from the inference
        // variable. That will accept more code than we do right now, so we need to carefully consider
        // the implications.
        // Note: this check is pessimistic, as the inference type could be matched with something other
        // than the opaque type, but then we need a new `TypeRelation` just for this specific case and
        // can't re-use `sup` below.
        // See tests/ui/impl-trait/hidden-type-is-opaque.rs and
        // tests/ui/impl-trait/hidden-type-is-opaque-2.rs for examples that hit this path.
        if formal_ret.has_infer_types() {
            for ty in ret_ty.walk() {
                if let ty::GenericArgKind::Type(ty) = ty.unpack()
                    && let ty::Alias(ty::Opaque, ty::AliasTy { def_id, .. }) = *ty.kind()
                    && let Some(def_id) = def_id.as_local()
                    && self.opaque_type_origin(def_id).is_some() {
                    return None;
                }
            }
        }

        let expect_args = self
            .fudge_inference_if_ok(|| {
                let ocx = ObligationCtxt::new(self);

                // Attempt to apply a subtyping relationship between the formal
                // return type (likely containing type variables if the function
                // is polymorphic) and the expected return type.
                // No argument expectations are produced if unification fails.
                let origin = self.misc(call_span);
                ocx.sup(&origin, self.param_env, ret_ty, formal_ret)?;
                if !ocx.select_where_possible().is_empty() {
                    return Err(TypeError::Mismatch);
                }

                // Record all the argument types, with the substitutions
                // produced from the above subtyping unification.
                Ok(Some(formal_args.iter().map(|&ty| self.resolve_vars_if_possible(ty)).collect()))
            })
            .unwrap_or_default();
        debug!(?formal_args, ?formal_ret, ?expect_args, ?expected_ret);
        expect_args
    }

    pub(in super::super) fn resolve_lang_item_path(
        &self,
        lang_item: hir::LangItem,
        span: Span,
        hir_id: hir::HirId,
        expr_hir_id: Option<hir::HirId>,
    ) -> (Res, Ty<'tcx>) {
        let def_id = self.tcx.require_lang_item(lang_item, Some(span));
        let def_kind = self.tcx.def_kind(def_id);

        let item_ty = if let DefKind::Variant = def_kind {
            self.tcx.type_of(self.tcx.parent(def_id))
        } else {
            self.tcx.type_of(def_id)
        };
        let args = self.fresh_args_for_item(span, def_id);
        let ty = item_ty.instantiate(self.tcx, args);

        self.write_resolution(hir_id, Ok((def_kind, def_id)));

        let code = match lang_item {
            hir::LangItem::IntoFutureIntoFuture => {
                Some(ObligationCauseCode::AwaitableExpr(expr_hir_id))
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
    pub fn resolve_ty_and_res_fully_qualified_call(
        &self,
        qpath: &'tcx QPath<'tcx>,
        hir_id: hir::HirId,
        span: Span,
    ) -> (Res, Option<RawTy<'tcx>>, &'tcx [hir::PathSegment<'tcx>]) {
        debug!(
            "resolve_ty_and_res_fully_qualified_call: qpath={:?} hir_id={:?} span={:?}",
            qpath, hir_id, span
        );
        let (ty, qself, item_segment) = match *qpath {
            QPath::Resolved(ref opt_qself, ref path) => {
                return (
                    path.res,
                    opt_qself.as_ref().map(|qself| self.to_ty(qself)),
                    path.segments,
                );
            }
            QPath::TypeRelative(ref qself, ref segment) => {
                // Don't use `self.to_ty`, since this will register a WF obligation.
                // If we're trying to call a nonexistent method on a trait
                // (e.g. `MyTrait::missing_method`), then resolution will
                // give us a `QPath::TypeRelative` with a trait object as
                // `qself`. In that case, we want to avoid registering a WF obligation
                // for `dyn MyTrait`, since we don't actually need the trait
                // to be object-safe.
                // We manually call `register_wf_obligation` in the success path
                // below.
                let ty = self.astconv().ast_ty_to_ty_in_path(qself);
                (self.handle_raw_ty(span, ty), qself, segment)
            }
            QPath::LangItem(..) => {
                bug!("`resolve_ty_and_res_fully_qualified_call` called on `LangItem`")
            }
        };
        if let Some(&cached_result) = self.typeck_results.borrow().type_dependent_defs().get(hir_id)
        {
            self.register_wf_obligation(ty.raw.into(), qself.span, traits::WellFormed(None));
            // Return directly on cache hit. This is useful to avoid doubly reporting
            // errors with default match binding modes. See #44614.
            let def = cached_result.map_or(Res::Err, |(kind, def_id)| Res::Def(kind, def_id));
            return (def, Some(ty), slice::from_ref(&**item_segment));
        }
        let item_name = item_segment.ident;
        let result = self
            .resolve_fully_qualified_call(span, item_name, ty.normalized, qself.span, hir_id)
            .and_then(|r| {
                // lint bare trait if the method is found in the trait
                if span.edition().rust_2021() && let Some(mut diag) = self.tcx.sess.diagnostic().steal_diagnostic(qself.span, StashKey::TraitMissingMethod) {
                    diag.emit();
                }
                Ok(r)
            })
            .or_else(|error| {
                let guar = self
                    .tcx
                    .sess
                    .delay_span_bug(span, "method resolution should've emitted an error");
                let result = match error {
                    method::MethodError::PrivateMatch(kind, def_id, _) => Ok((kind, def_id)),
                    _ => Err(guar),
                };

                let trait_missing_method =
                    matches!(error, method::MethodError::NoMatch(_)) && ty.normalized.is_trait();
                // If we have a path like `MyTrait::missing_method`, then don't register
                // a WF obligation for `dyn MyTrait` when method lookup fails. Otherwise,
                // register a WF obligation so that we can detect any additional
                // errors in the self type.
                if !trait_missing_method {
                    self.register_wf_obligation(
                        ty.raw.into(),
                        qself.span,
                        traits::WellFormed(None),
                    );
                }

                // emit or cancel the diagnostic for bare traits
                if span.edition().rust_2021() && let Some(mut diag) = self.tcx.sess.diagnostic().steal_diagnostic(qself.span, StashKey::TraitMissingMethod) {
                    if trait_missing_method {
                        // cancel the diag for bare traits when meeting `MyTrait::missing_method`
                        diag.cancel();
                    } else {
                        diag.emit();
                    }
                }

                if item_name.name != kw::Empty {
                    if let Some(mut e) = self.report_method_error(
                        span,
                        ty.normalized,
                        item_name,
                        SelfSource::QPath(qself),
                        error,
                        None,
                        Expectation::NoExpectation,
                        trait_missing_method && span.edition().rust_2021(), // emits missing method for trait only after edition 2021
                    ) {
                        e.emit();
                    }
                }

                result
            });

        if result.is_ok() {
            self.register_wf_obligation(ty.raw.into(), qself.span, traits::WellFormed(None));
        }

        // Write back the new resolution.
        self.write_resolution(hir_id, result);
        (
            result.map_or(Res::Err, |(kind, def_id)| Res::Def(kind, def_id)),
            Some(ty),
            slice::from_ref(&**item_segment),
        )
    }

    /// Given a function `Node`, return its  `HirId` and `FnDecl` if it exists. Given a closure
    /// that is the child of a function, return that function's `HirId` and `FnDecl` instead.
    /// This may seem confusing at first, but this is used in diagnostics for `async fn`,
    /// for example, where most of the type checking actually happens within a nested closure,
    /// but we often want access to the parent function's signature.
    ///
    /// Otherwise, return false.
    pub(in super::super) fn get_node_fn_decl(
        &self,
        node: Node<'tcx>,
    ) -> Option<(hir::HirId, &'tcx hir::FnDecl<'tcx>, Ident, bool)> {
        match node {
            Node::Item(&hir::Item {
                ident,
                kind: hir::ItemKind::Fn(ref sig, ..),
                owner_id,
                ..
            }) => {
                // This is less than ideal, it will not suggest a return type span on any
                // method called `main`, regardless of whether it is actually the entry point,
                // but it will still present it as the reason for the expected type.
                Some((
                    hir::HirId::make_owner(owner_id.def_id),
                    &sig.decl,
                    ident,
                    ident.name != sym::main,
                ))
            }
            Node::TraitItem(&hir::TraitItem {
                ident,
                kind: hir::TraitItemKind::Fn(ref sig, ..),
                owner_id,
                ..
            }) => Some((hir::HirId::make_owner(owner_id.def_id), &sig.decl, ident, true)),
            Node::ImplItem(&hir::ImplItem {
                ident,
                kind: hir::ImplItemKind::Fn(ref sig, ..),
                owner_id,
                ..
            }) => Some((hir::HirId::make_owner(owner_id.def_id), &sig.decl, ident, false)),
            Node::Expr(&hir::Expr { hir_id, kind: hir::ExprKind::Closure(..), .. })
                if let Some(Node::Item(&hir::Item {
                    ident,
                    kind: hir::ItemKind::Fn(ref sig, ..),
                    owner_id,
                    ..
                })) = self.tcx.hir().find_parent(hir_id) => Some((
                hir::HirId::make_owner(owner_id.def_id),
                &sig.decl,
                ident,
                ident.name != sym::main,
            )),
            _ => None,
        }
    }

    /// Given a `HirId`, return the `HirId` of the enclosing function, its `FnDecl`, and whether a
    /// suggestion can be made, `None` otherwise.
    pub fn get_fn_decl(
        &self,
        blk_id: hir::HirId,
    ) -> Option<(hir::HirId, &'tcx hir::FnDecl<'tcx>, bool)> {
        // Get enclosing Fn, if it is a function or a trait method, unless there's a `loop` or
        // `while` before reaching it, as block tail returns are not available in them.
        self.tcx.hir().get_return_block(blk_id).and_then(|blk_id| {
            let parent = self.tcx.hir().get(blk_id);
            self.get_node_fn_decl(parent)
                .map(|(fn_id, fn_decl, _, is_main)| (fn_id, fn_decl, is_main))
        })
    }

    pub(in super::super) fn note_internal_mutation_in_method(
        &self,
        err: &mut Diagnostic,
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
                modifies_rcvr_note.clone() + ", it is not meant to be used in method chains.",
            );
        } else {
            err.span_note(sp, modifies_rcvr_note);
        }
    }

    // Instantiates the given path, which must refer to an item with the given
    // number of type parameters and type.
    #[instrument(skip(self, span), level = "debug")]
    pub fn instantiate_value_path(
        &self,
        segments: &[hir::PathSegment<'_>],
        self_ty: Option<RawTy<'tcx>>,
        res: Res,
        span: Span,
        hir_id: hir::HirId,
    ) -> (Ty<'tcx>, Res) {
        let tcx = self.tcx;

        let path_segs = match res {
            Res::Local(_) | Res::SelfCtor(_) => vec![],
            Res::Def(kind, def_id) => self.astconv().def_ids_for_value_path_segments(
                segments,
                self_ty.map(|ty| ty.raw),
                kind,
                def_id,
                span,
            ),
            _ => bug!("instantiate_value_path on {:?}", res),
        };

        let mut user_self_ty = None;
        let mut is_alias_variant_ctor = false;
        match res {
            Res::Def(DefKind::Ctor(CtorOf::Variant, _), _)
                if let Some(self_ty) = self_ty =>
            {
                let adt_def = self_ty.normalized.ty_adt_def().unwrap();
                user_self_ty = Some(UserSelfTy { impl_def_id: adt_def.did(), self_ty: self_ty.raw });
                is_alias_variant_ctor = true;
            }
            Res::Def(DefKind::AssocFn | DefKind::AssocConst, def_id) => {
                let assoc_item = tcx.associated_item(def_id);
                let container = assoc_item.container;
                let container_id = assoc_item.container_id(tcx);
                debug!(?def_id, ?container, ?container_id);
                match container {
                    ty::TraitContainer => {
                        callee::check_legal_trait_for_method_call(tcx, span, None, span, container_id)
                    }
                    ty::ImplContainer => {
                        if segments.len() == 1 {
                            // `<T>::assoc` will end up here, and so
                            // can `T::assoc`. It this came from an
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

        let generic_segs: FxHashSet<_> = path_segs.iter().map(|PathSeg(_, index)| index).collect();
        let generics_has_err = self.astconv().prohibit_generics(
            segments.iter().enumerate().filter_map(|(index, seg)| {
                if !generic_segs.contains(&index) || is_alias_variant_ctor {
                    Some(seg)
                } else {
                    None
                }
            }),
            |_| {},
        );

        if let Res::Local(hid) = res {
            let ty = self.local_ty(span, hid);
            let ty = self.normalize(span, ty);
            self.write_ty(hir_id, ty);
            return (ty, res);
        }

        if generics_has_err {
            // Don't try to infer type parameters when prohibited generic arguments were given.
            user_self_ty = None;
        }

        // Now we have to compare the types that the user *actually*
        // provided against the types that were *expected*. If the user
        // did not provide any types, then we want to substitute inference
        // variables. If the user provided some types, we may still need
        // to add defaults. If the user provided *too many* types, that's
        // a problem.

        let mut infer_args_for_err = FxHashSet::default();

        let mut explicit_late_bound = ExplicitLateBound::No;
        for &PathSeg(def_id, index) in &path_segs {
            let seg = &segments[index];
            let generics = tcx.generics_of(def_id);

            // Argument-position `impl Trait` is treated as a normal generic
            // parameter internally, but we don't allow users to specify the
            // parameter's value explicitly, so we have to do some error-
            // checking here.
            let arg_count = check_generic_arg_count_for_call(
                tcx,
                span,
                def_id,
                &generics,
                seg,
                IsMethodCall::No,
            );

            if let ExplicitLateBound::Yes = arg_count.explicit_late_bound {
                explicit_late_bound = ExplicitLateBound::Yes;
            }

            if let Err(GenericArgCountMismatch { reported: Some(e), .. }) = arg_count.correct {
                infer_args_for_err.insert(index);
                self.set_tainted_by_errors(e); // See issue #53251.
            }
        }

        let has_self =
            path_segs.last().is_some_and(|PathSeg(def_id, _)| tcx.generics_of(*def_id).has_self);

        let (res, self_ctor_args) = if let Res::SelfCtor(impl_def_id) = res {
            let ty =
                self.handle_raw_ty(span, tcx.at(span).type_of(impl_def_id).instantiate_identity());
            match ty.normalized.ty_adt_def() {
                Some(adt_def) if adt_def.has_ctor() => {
                    let (ctor_kind, ctor_def_id) = adt_def.non_enum_variant().ctor.unwrap();
                    // Check the visibility of the ctor.
                    let vis = tcx.visibility(ctor_def_id);
                    if !vis.is_accessible_from(tcx.parent_module(hir_id).to_def_id(), tcx) {
                        tcx.sess
                            .emit_err(CtorIsPrivate { span, def: tcx.def_path_str(adt_def.did()) });
                    }
                    let new_res = Res::Def(DefKind::Ctor(CtorOf::Struct, ctor_kind), ctor_def_id);
                    let user_args = Self::user_args_for_adt(ty);
                    user_self_ty = user_args.user_self_ty;
                    (new_res, Some(user_args.args))
                }
                _ => {
                    let mut err = tcx.sess.struct_span_err(
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

        let arg_count = GenericArgCountResult {
            explicit_late_bound,
            correct: if infer_args_for_err.is_empty() {
                Ok(())
            } else {
                Err(GenericArgCountMismatch::default())
            },
        };

        struct CreateCtorSubstsContext<'a, 'tcx> {
            fcx: &'a FnCtxt<'a, 'tcx>,
            span: Span,
            path_segs: &'a [PathSeg],
            infer_args_for_err: &'a FxHashSet<usize>,
            segments: &'a [hir::PathSegment<'a>],
        }
        impl<'tcx, 'a> CreateSubstsForGenericArgsCtxt<'a, 'tcx> for CreateCtorSubstsContext<'a, 'tcx> {
            fn args_for_def_id(
                &mut self,
                def_id: DefId,
            ) -> (Option<&'a hir::GenericArgs<'a>>, bool) {
                if let Some(&PathSeg(_, index)) =
                    self.path_segs.iter().find(|&PathSeg(did, _)| *did == def_id)
                {
                    // If we've encountered an `impl Trait`-related error, we're just
                    // going to infer the arguments for better error messages.
                    if !self.infer_args_for_err.contains(&index) {
                        // Check whether the user has provided generic arguments.
                        if let Some(ref data) = self.segments[index].args {
                            return (Some(data), self.segments[index].infer_args);
                        }
                    }
                    return (None, self.segments[index].infer_args);
                }

                (None, true)
            }

            fn provided_kind(
                &mut self,
                param: &ty::GenericParamDef,
                arg: &GenericArg<'_>,
            ) -> ty::GenericArg<'tcx> {
                match (&param.kind, arg) {
                    (GenericParamDefKind::Lifetime, GenericArg::Lifetime(lt)) => {
                        self.fcx.astconv().ast_region_to_region(lt, Some(param)).into()
                    }
                    (GenericParamDefKind::Type { .. }, GenericArg::Type(ty)) => {
                        self.fcx.to_ty(ty).raw.into()
                    }
                    (GenericParamDefKind::Const { .. }, GenericArg::Const(ct)) => {
                        self.fcx.const_arg_to_const(&ct.value, param.def_id).into()
                    }
                    (GenericParamDefKind::Type { .. }, GenericArg::Infer(inf)) => {
                        self.fcx.ty_infer(Some(param), inf.span).into()
                    }
                    (GenericParamDefKind::Const { .. }, GenericArg::Infer(inf)) => {
                        let tcx = self.fcx.tcx();
                        self.fcx
                            .ct_infer(
                                tcx.type_of(param.def_id)
                                    .no_bound_vars()
                                    .expect("const parameter types cannot be generic"),
                                Some(param),
                                inf.span,
                            )
                            .into()
                    }
                    _ => unreachable!(),
                }
            }

            fn inferred_kind(
                &mut self,
                args: Option<&[ty::GenericArg<'tcx>]>,
                param: &ty::GenericParamDef,
                infer_args: bool,
            ) -> ty::GenericArg<'tcx> {
                let tcx = self.fcx.tcx();
                match param.kind {
                    GenericParamDefKind::Lifetime => {
                        self.fcx.re_infer(Some(param), self.span).unwrap().into()
                    }
                    GenericParamDefKind::Type { has_default, .. } => {
                        if !infer_args && has_default {
                            // If we have a default, then we it doesn't matter that we're not
                            // inferring the type arguments: we provide the default where any
                            // is missing.
                            tcx.type_of(param.def_id).instantiate(tcx, args.unwrap()).into()
                        } else {
                            // If no type arguments were provided, we have to infer them.
                            // This case also occurs as a result of some malformed input, e.g.
                            // a lifetime argument being given instead of a type parameter.
                            // Using inference instead of `Error` gives better error messages.
                            self.fcx.var_for_def(self.span, param)
                        }
                    }
                    GenericParamDefKind::Const { has_default } => {
                        if !infer_args && has_default {
                            tcx.const_param_default(param.def_id)
                                .instantiate(tcx, args.unwrap())
                                .into()
                        } else {
                            self.fcx.var_for_def(self.span, param)
                        }
                    }
                }
            }
        }

        let args_raw = self_ctor_args.unwrap_or_else(|| {
            create_args_for_parent_generic_args(
                tcx,
                def_id,
                &[],
                has_self,
                self_ty.map(|s| s.raw),
                &arg_count,
                &mut CreateCtorSubstsContext {
                    fcx: self,
                    span,
                    path_segs: &path_segs,
                    infer_args_for_err: &infer_args_for_err,
                    segments,
                },
            )
        });

        // First, store the "user args" for later.
        self.write_user_type_annotation_from_args(hir_id, def_id, args_raw, user_self_ty);

        // Normalize only after registering type annotations.
        let args = self.normalize(span, args_raw);

        self.add_required_obligations_for_hir(span, def_id, &args, hir_id);

        // Substitute the values for the type parameters into the type of
        // the referenced item.
        let ty = tcx.type_of(def_id);
        assert!(!args.has_escaping_bound_vars());
        assert!(!ty.skip_binder().has_escaping_bound_vars());
        let ty_substituted = self.normalize(span, ty.instantiate(tcx, args));

        if let Some(UserSelfTy { impl_def_id, self_ty }) = user_self_ty {
            // In the case of `Foo<T>::method` and `<Foo<T>>::method`, if `method`
            // is inherent, there is no `Self` parameter; instead, the impl needs
            // type parameters, which we can infer by unifying the provided `Self`
            // with the substituted impl type.
            // This also occurs for an enum variant on a type alias.
            let impl_ty = self.normalize(span, tcx.type_of(impl_def_id).instantiate(tcx, args));
            let self_ty = self.normalize(span, self_ty);
            match self.at(&self.misc(span), self.param_env).eq(
                DefineOpaqueTypes::No,
                impl_ty,
                self_ty,
            ) {
                Ok(ok) => self.register_infer_ok_obligations(ok),
                Err(_) => {
                    self.tcx.sess.delay_span_bug(
                        span,
                        format!(
                        "instantiate_value_path: (UFCS) {:?} was a subtype of {:?} but now is not?",
                        self_ty,
                        impl_ty,
                    ),
                    );
                }
            }
        }

        debug!("instantiate_value_path: type of {:?} is {:?}", hir_id, ty_substituted);
        self.write_args(hir_id, args);

        (ty_substituted, res)
    }

    /// Add all the obligations that are required, substituting and normalized appropriately.
    pub(crate) fn add_required_obligations_for_hir(
        &self,
        span: Span,
        def_id: DefId,
        args: GenericArgsRef<'tcx>,
        hir_id: hir::HirId,
    ) {
        self.add_required_obligations_with_code(span, def_id, args, |idx, span| {
            if span.is_dummy() {
                ObligationCauseCode::ExprItemObligation(def_id, hir_id, idx)
            } else {
                ObligationCauseCode::ExprBindingObligation(def_id, span, hir_id, idx)
            }
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

        let bounds = self.instantiate_bounds(span, def_id, &args);

        for obligation in traits::predicates_for_generics(
            |idx, predicate_span| {
                traits::ObligationCause::new(span, self.body_id, code(idx, predicate_span))
            },
            param_env,
            bounds,
        ) {
            // N.B. We are remapping all predicates to non-const since we don't know if we just
            // want them as function pointers or we are calling them from a const-context. The
            // actual checking will occur in `rustc_const_eval::transform::check_consts`.
            self.register_predicate(obligation.without_const(self.tcx));
        }
    }

    /// Try to resolve `ty` to a structural type, normalizing aliases.
    ///
    /// In case there is still ambiguity, the returned type may be an inference
    /// variable. This is different from `structurally_resolve_type` which errors
    /// in this case.
    pub fn try_structurally_resolve_type(&self, sp: Span, ty: Ty<'tcx>) -> Ty<'tcx> {
        let ty = self.resolve_vars_with_obligations(ty);

        if self.next_trait_solver()
            && let ty::Alias(ty::Projection, _) = ty.kind()
        {
            match self
                .at(&self.misc(sp), self.param_env)
                .structurally_normalize(ty, &mut **self.fulfillment_cx.borrow_mut())
            {
                Ok(normalized_ty) => normalized_ty,
                Err(errors) => {
                    let guar = self.err_ctxt().report_fulfillment_errors(&errors);
                    return Ty::new_error(self.tcx,guar);
                }
            }
        } else {
            ty
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
    pub fn structurally_resolve_type(&self, sp: Span, ty: Ty<'tcx>) -> Ty<'tcx> {
        let ty = self.try_structurally_resolve_type(sp, ty);

        if !ty.is_ty_var() {
            ty
        } else {
            let e = self.tainted_by_errors().unwrap_or_else(|| {
                self.err_ctxt()
                    .emit_inference_failure_err(self.body_id, sp, ty.into(), E0282, true)
                    .emit()
            });
            let err = Ty::new_error(self.tcx, e);
            self.demand_suptype(sp, err, ty);
            err
        }
    }

    pub(in super::super) fn with_breakable_ctxt<F: FnOnce() -> R, R>(
        &self,
        id: hir::HirId,
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
            enclosing_breakables.by_id.remove(&id).expect("missing breakable context");
            enclosing_breakables.stack.pop().expect("missing breakable context")
        };
        (ctxt, result)
    }

    /// Instantiate a QueryResponse in a probe context, without a
    /// good ObligationCause.
    pub(in super::super) fn probe_instantiate_query_response(
        &self,
        span: Span,
        original_values: &OriginalQueryValues<'tcx>,
        query_result: &Canonical<'tcx, QueryResponse<'tcx, Ty<'tcx>>>,
    ) -> InferResult<'tcx, Ty<'tcx>> {
        self.instantiate_query_response_and_region_obligations(
            &traits::ObligationCause::misc(span, self.body_id),
            self.param_env,
            original_values,
            query_result,
        )
    }

    /// Returns `true` if an expression is contained inside the LHS of an assignment expression.
    pub(in super::super) fn expr_in_place(&self, mut expr_id: hir::HirId) -> bool {
        let mut contained_in_place = false;

        while let hir::Node::Expr(parent_expr) = self.tcx.hir().get_parent(expr_id) {
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
