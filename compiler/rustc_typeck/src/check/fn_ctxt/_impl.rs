use crate::astconv::{
    AstConv, CreateSubstsForGenericArgsCtxt, ExplicitLateBound, GenericArgCountMismatch,
    GenericArgCountResult, IsMethodCall, PathSeg,
};
use crate::check::callee::{self, DeferredCallResolution};
use crate::check::method::{self, MethodCallee, SelfSource};
use crate::check::rvalue_scopes;
use crate::check::{BreakableCtxt, Diverges, Expectation, FnCtxt, LocalTy};

use rustc_data_structures::captures::Captures;
use rustc_data_structures::fx::FxHashSet;
use rustc_errors::{Applicability, Diagnostic, ErrorGuaranteed, MultiSpan};
use rustc_hir as hir;
use rustc_hir::def::{CtorOf, DefKind, Res};
use rustc_hir::def_id::DefId;
use rustc_hir::lang_items::LangItem;
use rustc_hir::{ExprKind, GenericArg, Node, QPath};
use rustc_infer::infer::canonical::{Canonical, OriginalQueryValues, QueryResponse};
use rustc_infer::infer::error_reporting::TypeAnnotationNeeded::E0282;
use rustc_infer::infer::{InferOk, InferResult};
use rustc_middle::ty::adjustment::{Adjust, Adjustment, AutoBorrow, AutoBorrowMutability};
use rustc_middle::ty::fold::TypeFoldable;
use rustc_middle::ty::subst::{
    self, GenericArgKind, InternalSubsts, Subst, SubstsRef, UserSelfTy, UserSubsts,
};
use rustc_middle::ty::visit::TypeVisitable;
use rustc_middle::ty::{
    self, AdtKind, CanonicalUserType, DefIdTree, EarlyBinder, GenericParamDefKind, ToPolyTraitRef,
    ToPredicate, Ty, UserType,
};
use rustc_session::lint;
use rustc_span::def_id::LocalDefId;
use rustc_span::hygiene::DesugaringKind;
use rustc_span::symbol::{kw, sym, Ident};
use rustc_span::{Span, DUMMY_SP};
use rustc_trait_selection::infer::InferCtxtExt as _;
use rustc_trait_selection::traits::error_reporting::InferCtxtExt as _;
use rustc_trait_selection::traits::{
    self, ObligationCause, ObligationCauseCode, TraitEngine, TraitEngineExt,
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

                self.tcx().struct_span_lint_hir(lint::builtin::UNREACHABLE_CODE, id, span, |lint| {
                    let msg = format!("unreachable {}", kind);
                    lint.build(&msg)
                        .span_label(span, &msg)
                        .span_label(
                            orig_span,
                            custom_note
                                .unwrap_or("any code following this expression is unreachable"),
                        )
                        .emit();
                })
            }
        }
    }

    /// Resolves type and const variables in `ty` if possible. Unlike the infcx
    /// version (resolve_vars_if_possible), this version will
    /// also select obligations if it seems useful, in an effort
    /// to get more type information.
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
        if !ty.has_infer_types_or_consts() {
            debug!("no inference var, nothing needs doing");
            return ty;
        }

        // If `ty` is a type variable, see whether we already know what it is.
        ty = self.resolve_vars_if_possible(ty);
        if !ty.has_infer_types_or_consts() {
            debug!(?ty);
            return ty;
        }

        // If not, try resolving pending obligations as much as
        // possible. This can help substantially when there are
        // indirect dependencies that don't seem worth tracking
        // precisely.
        self.select_obligations_where_possible(false, mutate_fulfillment_errors);
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

    pub fn local_ty(&self, span: Span, nid: hir::HirId) -> LocalTy<'tcx> {
        self.locals.borrow().get(&nid).cloned().unwrap_or_else(|| {
            span_bug!(span, "no type for local variable {}", self.tcx.hir().node_to_string(nid))
        })
    }

    #[inline]
    pub fn write_ty(&self, id: hir::HirId, ty: Ty<'tcx>) {
        debug!("write_ty({:?}, {:?}) in fcx {}", id, self.resolve_vars_if_possible(ty), self.tag());
        self.typeck_results.borrow_mut().node_types_mut().insert(id, ty);

        if ty.references_error() {
            self.has_errors.set(true);
            self.set_tainted_by_errors();
        }
    }

    pub fn write_field_index(&self, hir_id: hir::HirId, index: usize) {
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
        self.write_substs(hir_id, method.substs);

        // When the method is confirmed, the `method.substs` includes
        // parameters from not just the method, but also the impl of
        // the method -- in particular, the `Self` type will be fully
        // resolved. However, those are not something that the "user
        // specified" -- i.e., those types come from the inferred type
        // of the receiver, not something the user wrote. So when we
        // create the user-substs, we want to replace those earlier
        // types with just the types that the user actually wrote --
        // that is, those that appear on the *method itself*.
        //
        // As an example, if the user wrote something like
        // `foo.bar::<u32>(...)` -- the `Self` type here will be the
        // type of `foo` (possibly adjusted), but we don't want to
        // include that. We want just the `[_, u32]` part.
        if !method.substs.is_empty() {
            let method_generics = self.tcx.generics_of(method.def_id);
            if !method_generics.params.is_empty() {
                let user_type_annotation = self.probe(|_| {
                    let user_substs = UserSubsts {
                        substs: InternalSubsts::for_item(self.tcx, method.def_id, |param, _| {
                            let i = param.index as usize;
                            if i < method_generics.parent_count {
                                self.var_for_def(DUMMY_SP, param)
                            } else {
                                method.substs[i]
                            }
                        }),
                        user_self_ty: None, // not relevant here
                    };

                    self.canonicalize_user_type_annotation(UserType::TypeOf(
                        method.def_id,
                        user_substs,
                    ))
                });

                debug!("write_method_call: user_type_annotation={:?}", user_type_annotation);
                self.write_user_type_annotation(hir_id, user_type_annotation);
            }
        }
    }

    pub fn write_substs(&self, node_id: hir::HirId, substs: SubstsRef<'tcx>) {
        if !substs.is_empty() {
            debug!("write_substs({:?}, {:?}) in fcx {}", node_id, substs, self.tag());

            self.typeck_results.borrow_mut().node_substs_mut().insert(node_id, substs);
        }
    }

    /// Given the substs that we just converted from the HIR, try to
    /// canonicalize them and store them as user-given substitutions
    /// (i.e., substitutions that must be respected by the NLL check).
    ///
    /// This should be invoked **before any unifications have
    /// occurred**, so that annotations like `Vec<_>` are preserved
    /// properly.
    #[instrument(skip(self), level = "debug")]
    pub fn write_user_type_annotation_from_substs(
        &self,
        hir_id: hir::HirId,
        def_id: DefId,
        substs: SubstsRef<'tcx>,
        user_self_ty: Option<UserSelfTy<'tcx>>,
    ) {
        debug!("fcx {}", self.tag());

        if Self::can_contain_user_lifetime_bounds((substs, user_self_ty)) {
            let canonicalized = self.canonicalize_user_type_annotation(UserType::TypeOf(
                def_id,
                UserSubsts { substs, user_self_ty },
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
            debug!("skipping identity substs");
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
                            &format!(
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

    /// Basically whenever we are converting from a type scheme into
    /// the fn body space, we always want to normalize associated
    /// types as well. This function combines the two.
    fn instantiate_type_scheme<T>(&self, span: Span, substs: SubstsRef<'tcx>, value: T) -> T
    where
        T: TypeFoldable<'tcx>,
    {
        debug!("instantiate_type_scheme(value={:?}, substs={:?})", value, substs);
        let value = EarlyBinder(value).subst(self.tcx, substs);
        let result = self.normalize_associated_types_in(span, value);
        debug!("instantiate_type_scheme = {:?}", result);
        result
    }

    /// As `instantiate_type_scheme`, but for the bounds found in a
    /// generic type scheme.
    pub(in super::super) fn instantiate_bounds(
        &self,
        span: Span,
        def_id: DefId,
        substs: SubstsRef<'tcx>,
    ) -> (ty::InstantiatedPredicates<'tcx>, Vec<Span>) {
        let bounds = self.tcx.predicates_of(def_id);
        let spans: Vec<Span> = bounds.predicates.iter().map(|(_, span)| *span).collect();
        let result = bounds.instantiate(self.tcx, substs);
        let result = self.normalize_associated_types_in(span, result);
        debug!(
            "instantiate_bounds(bounds={:?}, substs={:?}) = {:?}, {:?}",
            bounds, substs, result, spans,
        );
        (result, spans)
    }

    pub(in super::super) fn normalize_associated_types_in<T>(&self, span: Span, value: T) -> T
    where
        T: TypeFoldable<'tcx>,
    {
        self.inh.normalize_associated_types_in(span, self.body_id, self.param_env, value)
    }

    pub(in super::super) fn normalize_associated_types_in_as_infer_ok<T>(
        &self,
        span: Span,
        value: T,
    ) -> InferOk<'tcx, T>
    where
        T: TypeFoldable<'tcx>,
    {
        self.inh.partially_normalize_associated_types_in(
            ObligationCause::misc(span, self.body_id),
            self.param_env,
            value,
        )
    }

    pub(in super::super) fn normalize_op_associated_types_in_as_infer_ok<T>(
        &self,
        span: Span,
        value: T,
        opt_input_expr: Option<&hir::Expr<'_>>,
    ) -> InferOk<'tcx, T>
    where
        T: TypeFoldable<'tcx>,
    {
        self.inh.partially_normalize_associated_types_in(
            ObligationCause::new(
                span,
                self.body_id,
                traits::BinOp {
                    rhs_span: opt_input_expr.map(|expr| expr.span),
                    is_lit: opt_input_expr
                        .map_or(false, |expr| matches!(expr.kind, ExprKind::Lit(_))),
                    output_pred: None,
                },
            ),
            self.param_env,
            value,
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

    pub fn to_ty(&self, ast_t: &hir::Ty<'_>) -> Ty<'tcx> {
        let t = <dyn AstConv<'_>>::ast_ty_to_ty(self, ast_t);
        self.register_wf_obligation(t.into(), ast_t.span, traits::WellFormed(None));
        t
    }

    pub fn to_ty_saving_user_provided_ty(&self, ast_ty: &hir::Ty<'_>) -> Ty<'tcx> {
        let ty = self.to_ty(ast_ty);
        debug!("to_ty_saving_user_provided_ty: ty={:?}", ty);

        if Self::can_contain_user_lifetime_bounds(ty) {
            let c_ty = self.canonicalize_response(UserType::Ty(ty));
            debug!("to_ty_saving_user_provided_ty: c_ty={:?}", c_ty);
            self.typeck_results.borrow_mut().user_provided_types_mut().insert(ast_ty.hir_id, c_ty);
        }

        ty
    }

    pub fn array_length_to_const(&self, length: &hir::ArrayLen) -> ty::Const<'tcx> {
        match length {
            &hir::ArrayLen::Infer(_, span) => self.ct_infer(self.tcx.types.usize, None, span),
            hir::ArrayLen::Body(anon_const) => self.to_const(anon_const),
        }
    }

    pub fn to_const(&self, ast_c: &hir::AnonConst) -> ty::Const<'tcx> {
        let const_def_id = self.tcx.hir().local_def_id(ast_c.hir_id);
        let c = ty::Const::from_anon_const(self.tcx, const_def_id);
        self.register_wf_obligation(
            c.into(),
            self.tcx.hir().span(ast_c.hir_id),
            ObligationCauseCode::WellFormed(None),
        );
        c
    }

    pub fn const_arg_to_const(
        &self,
        ast_c: &hir::AnonConst,
        param_def_id: DefId,
    ) -> ty::Const<'tcx> {
        let const_def = ty::WithOptConstParam {
            did: self.tcx.hir().local_def_id(ast_c.hir_id),
            const_param_did: Some(param_def_id),
        };
        let c = ty::Const::from_opt_const_arg_anon_const(self.tcx, const_def);
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
        T: TypeVisitable<'tcx>,
    {
        t.has_free_regions() || t.has_projections() || t.has_infer_types()
    }

    pub fn node_ty(&self, id: hir::HirId) -> Ty<'tcx> {
        match self.typeck_results.borrow().node_types().get(id) {
            Some(&t) => t,
            None if self.is_tainted_by_errors() => self.tcx.ty_error(),
            None => {
                bug!(
                    "no type for node {}: {} in fcx {}",
                    id,
                    self.tcx.hir().node_to_string(id),
                    self.tag()
                );
            }
        }
    }

    pub fn node_ty_opt(&self, id: hir::HirId) -> Option<Ty<'tcx>> {
        match self.typeck_results.borrow().node_types().get(id) {
            Some(&t) => Some(t),
            None if self.is_tainted_by_errors() => Some(self.tcx.ty_error()),
            None => None,
        }
    }

    /// Registers an obligation for checking later, during regionck, that `arg` is well-formed.
    pub fn register_wf_obligation(
        &self,
        arg: subst::GenericArg<'tcx>,
        span: Span,
        code: traits::ObligationCauseCode<'tcx>,
    ) {
        // WF obligations never themselves fail, so no real need to give a detailed cause:
        let cause = traits::ObligationCause::new(span, self.body_id, code);
        self.register_predicate(traits::Obligation::new(
            cause,
            self.param_env,
            ty::Binder::dummy(ty::PredicateKind::WellFormed(arg)).to_predicate(self.tcx),
        ));
    }

    /// Registers obligations that all `substs` are well-formed.
    pub fn add_wf_bounds(&self, substs: SubstsRef<'tcx>, expr: &hir::Expr<'_>) {
        for arg in substs.iter().filter(|arg| {
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
        substs: SubstsRef<'tcx>,
    ) -> Ty<'tcx> {
        self.normalize_associated_types_in(span, field.ty(self.tcx, substs))
    }

    pub(in super::super) fn resolve_rvalue_scopes(&self, def_id: DefId) {
        let scope_tree = self.tcx.region_scope_tree(def_id);
        let rvalue_scopes = { rvalue_scopes::resolve_rvalue_scopes(self, &scope_tree, def_id) };
        let mut typeck_results = self.inh.typeck_results.borrow_mut();
        typeck_results.rvalue_scopes = rvalue_scopes;
    }

    pub(in super::super) fn resolve_generator_interiors(&self, def_id: DefId) {
        let mut generators = self.deferred_generator_interiors.borrow_mut();
        for (body_id, interior, kind) in generators.drain(..) {
            self.select_obligations_where_possible(false, |_| {});
            crate::check::generator_interior::resolve_interior(
                self, def_id, body_id, interior, kind,
            );
        }
    }

    #[instrument(skip(self), level = "debug")]
    pub(in super::super) fn select_all_obligations_or_error(&self) {
        let mut errors = self.fulfillment_cx.borrow_mut().select_all_or_error(&self);

        if !errors.is_empty() {
            self.adjust_fulfillment_errors_for_expr_obligation(&mut errors);
            self.report_fulfillment_errors(&errors, self.inh.body_id, false);
        }
    }

    /// Select as many obligations as we can at present.
    pub(in super::super) fn select_obligations_where_possible(
        &self,
        fallback_has_occurred: bool,
        mutate_fulfillment_errors: impl Fn(&mut Vec<traits::FulfillmentError<'tcx>>),
    ) {
        let mut result = self.fulfillment_cx.borrow_mut().select_where_possible(self);
        if !result.is_empty() {
            mutate_fulfillment_errors(&mut result);
            self.adjust_fulfillment_errors_for_expr_obligation(&mut result);
            self.report_fulfillment_errors(&result, self.inh.body_id, fallback_has_occurred);
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
    fn self_type_matches_expected_vid(
        &self,
        trait_ref: ty::PolyTraitRef<'tcx>,
        expected_vid: ty::TyVid,
    ) -> bool {
        let self_ty = self.shallow_resolve(trait_ref.skip_binder().self_ty());
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
    ) -> impl Iterator<Item = (ty::PolyTraitRef<'tcx>, traits::PredicateObligation<'tcx>)>
    + Captures<'tcx>
    + 'b {
        // FIXME: consider using `sub_root_var` here so we
        // can see through subtyping.
        let ty_var_root = self.root_var(self_ty);
        trace!("pending_obligations = {:#?}", self.fulfillment_cx.borrow().pending_obligations());

        self.fulfillment_cx
            .borrow()
            .pending_obligations()
            .into_iter()
            .filter_map(move |obligation| {
                let bound_predicate = obligation.predicate.kind();
                match bound_predicate.skip_binder() {
                    ty::PredicateKind::Projection(data) => Some((
                        bound_predicate.rebind(data).required_poly_trait_ref(self.tcx),
                        obligation,
                    )),
                    ty::PredicateKind::Trait(data) => {
                        Some((bound_predicate.rebind(data).to_poly_trait_ref(), obligation))
                    }
                    ty::PredicateKind::Subtype(..) => None,
                    ty::PredicateKind::Coerce(..) => None,
                    ty::PredicateKind::RegionOutlives(..) => None,
                    ty::PredicateKind::TypeOutlives(..) => None,
                    ty::PredicateKind::WellFormed(..) => None,
                    ty::PredicateKind::ObjectSafe(..) => None,
                    ty::PredicateKind::ConstEvaluatable(..) => None,
                    ty::PredicateKind::ConstEquate(..) => None,
                    // N.B., this predicate is created by breaking down a
                    // `ClosureType: FnFoo()` predicate, where
                    // `ClosureType` represents some `Closure`. It can't
                    // possibly be referring to the current closure,
                    // because we haven't produced the `Closure` for
                    // this closure yet; this is exactly why the other
                    // code is looking for a self type of an unresolved
                    // inference variable.
                    ty::PredicateKind::ClosureKind(..) => None,
                    ty::PredicateKind::TypeWellFormedFromEnv(..) => None,
                }
            })
            .filter(move |(tr, _)| self.self_type_matches_expected_vid(*tr, ty_var_root))
    }

    pub(in super::super) fn type_var_is_sized(&self, self_ty: ty::TyVid) -> bool {
        self.obligations_for_self_ty(self_ty)
            .any(|(tr, _)| Some(tr.def_id()) == self.tcx.lang_items().sized_trait())
    }

    pub(in super::super) fn err_args(&self, len: usize) -> Vec<Ty<'tcx>> {
        vec![self.tcx.ty_error(); len]
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
        // See src/test/ui/impl-trait/hidden-type-is-opaque.rs and
        // src/test/ui/impl-trait/hidden-type-is-opaque-2.rs for examples that hit this path.
        if formal_ret.has_infer_types() {
            for ty in ret_ty.walk() {
                if let ty::subst::GenericArgKind::Type(ty) = ty.unpack()
                    && let ty::Opaque(def_id, _) = *ty.kind()
                    && let Some(def_id) = def_id.as_local()
                    && self.opaque_type_origin(def_id, DUMMY_SP).is_some() {
                    return None;
                }
            }
        }

        let expect_args = self
            .fudge_inference_if_ok(|| {
                // Attempt to apply a subtyping relationship between the formal
                // return type (likely containing type variables if the function
                // is polymorphic) and the expected return type.
                // No argument expectations are produced if unification fails.
                let origin = self.misc(call_span);
                let ures = self.at(&origin, self.param_env).sup(ret_ty, formal_ret);

                // FIXME(#27336) can't use ? here, Try::from_error doesn't default
                // to identity so the resulting type is not constrained.
                match ures {
                    Ok(ok) => {
                        // Process any obligations locally as much as
                        // we can.  We don't care if some things turn
                        // out unconstrained or ambiguous, as we're
                        // just trying to get hints here.
                        let errors = self.save_and_restore_in_snapshot_flag(|_| {
                            let mut fulfill = <dyn TraitEngine<'_>>::new(self.tcx);
                            for obligation in ok.obligations {
                                fulfill.register_predicate_obligation(self, obligation);
                            }
                            fulfill.select_where_possible(self)
                        });

                        if !errors.is_empty() {
                            return Err(());
                        }
                    }
                    Err(_) => return Err(()),
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
            self.tcx.bound_type_of(self.tcx.parent(def_id))
        } else {
            self.tcx.bound_type_of(def_id)
        };
        let substs = self.fresh_substs_for_item(span, def_id);
        let ty = item_ty.subst(self.tcx, substs);

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
            self.add_required_obligations_with_code(span, def_id, substs, move |_, _| code.clone());
        } else {
            self.add_required_obligations_for_hir(span, def_id, substs, hir_id);
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
    ) -> (Res, Option<Ty<'tcx>>, &'tcx [hir::PathSegment<'tcx>]) {
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
                // If we're trying to call a non-existent method on a trait
                // (e.g. `MyTrait::missing_method`), then resolution will
                // give us a `QPath::TypeRelative` with a trait object as
                // `qself`. In that case, we want to avoid registering a WF obligation
                // for `dyn MyTrait`, since we don't actually need the trait
                // to be object-safe.
                // We manually call `register_wf_obligation` in the success path
                // below.
                (<dyn AstConv<'_>>::ast_ty_to_ty_in_path(self, qself), qself, segment)
            }
            QPath::LangItem(..) => {
                bug!("`resolve_ty_and_res_fully_qualified_call` called on `LangItem`")
            }
        };
        if let Some(&cached_result) = self.typeck_results.borrow().type_dependent_defs().get(hir_id)
        {
            self.register_wf_obligation(ty.into(), qself.span, traits::WellFormed(None));
            // Return directly on cache hit. This is useful to avoid doubly reporting
            // errors with default match binding modes. See #44614.
            let def = cached_result.map_or(Res::Err, |(kind, def_id)| Res::Def(kind, def_id));
            return (def, Some(ty), slice::from_ref(&**item_segment));
        }
        let item_name = item_segment.ident;
        let result = self
            .resolve_fully_qualified_call(span, item_name, ty, qself.span, hir_id)
            .or_else(|error| {
                let result = match error {
                    method::MethodError::PrivateMatch(kind, def_id, _) => Ok((kind, def_id)),
                    _ => Err(ErrorGuaranteed::unchecked_claim_error_was_emitted()),
                };

                // If we have a path like `MyTrait::missing_method`, then don't register
                // a WF obligation for `dyn MyTrait` when method lookup fails. Otherwise,
                // register a WF obligation so that we can detect any additional
                // errors in the self type.
                if !(matches!(error, method::MethodError::NoMatch(_)) && ty.is_trait()) {
                    self.register_wf_obligation(ty.into(), qself.span, traits::WellFormed(None));
                }
                if item_name.name != kw::Empty {
                    if let Some(mut e) = self.report_method_error(
                        span,
                        ty,
                        item_name,
                        SelfSource::QPath(qself),
                        error,
                        None,
                    ) {
                        e.emit();
                    }
                }
                result
            });

        if result.is_ok() {
            self.register_wf_obligation(ty.into(), qself.span, traits::WellFormed(None));
        }

        // Write back the new resolution.
        self.write_resolution(hir_id, result);
        (
            result.map_or(Res::Err, |(kind, def_id)| Res::Def(kind, def_id)),
            Some(ty),
            slice::from_ref(&**item_segment),
        )
    }

    /// Given a function `Node`, return its `FnDecl` if it exists, or `None` otherwise.
    pub(in super::super) fn get_node_fn_decl(
        &self,
        node: Node<'tcx>,
    ) -> Option<(&'tcx hir::FnDecl<'tcx>, Ident, bool)> {
        match node {
            Node::Item(&hir::Item { ident, kind: hir::ItemKind::Fn(ref sig, ..), .. }) => {
                // This is less than ideal, it will not suggest a return type span on any
                // method called `main`, regardless of whether it is actually the entry point,
                // but it will still present it as the reason for the expected type.
                Some((&sig.decl, ident, ident.name != sym::main))
            }
            Node::TraitItem(&hir::TraitItem {
                ident,
                kind: hir::TraitItemKind::Fn(ref sig, ..),
                ..
            }) => Some((&sig.decl, ident, true)),
            Node::ImplItem(&hir::ImplItem {
                ident,
                kind: hir::ImplItemKind::Fn(ref sig, ..),
                ..
            }) => Some((&sig.decl, ident, false)),
            _ => None,
        }
    }

    /// Given a `HirId`, return the `FnDecl` of the method it is enclosed by and whether a
    /// suggestion can be made, `None` otherwise.
    pub fn get_fn_decl(&self, blk_id: hir::HirId) -> Option<(&'tcx hir::FnDecl<'tcx>, bool)> {
        // Get enclosing Fn, if it is a function or a trait method, unless there's a `loop` or
        // `while` before reaching it, as block tail returns are not available in them.
        self.tcx.hir().get_return_block(blk_id).and_then(|blk_id| {
            let parent = self.tcx.hir().get(blk_id);
            self.get_node_fn_decl(parent).map(|(fn_decl, _, is_main)| (fn_decl, is_main))
        })
    }

    pub(in super::super) fn note_internal_mutation_in_method(
        &self,
        err: &mut Diagnostic,
        expr: &hir::Expr<'_>,
        expected: Ty<'tcx>,
        found: Ty<'tcx>,
    ) {
        if found != self.tcx.types.unit {
            return;
        }
        if let ExprKind::MethodCall(path_segment, rcvr, ..) = expr.kind {
            if self
                .typeck_results
                .borrow()
                .expr_ty_adjusted_opt(rcvr)
                .map_or(true, |ty| expected.peel_refs() != ty.peel_refs())
            {
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
            sp.push_span_label(
                rcvr.span,
                "you probably want to use this value after calling the method...",
            );
            err.span_note(
                sp,
                &format!("method `{}` modifies its receiver in-place", path_segment.ident),
            );
            err.note(&format!("...instead of the `()` output of method `{}`", path_segment.ident));
        }
    }

    pub(in super::super) fn note_need_for_fn_pointer(
        &self,
        err: &mut Diagnostic,
        expected: Ty<'tcx>,
        found: Ty<'tcx>,
    ) {
        let (sig, did, substs) = match (&expected.kind(), &found.kind()) {
            (ty::FnDef(did1, substs1), ty::FnDef(did2, substs2)) => {
                let sig1 = self.tcx.bound_fn_sig(*did1).subst(self.tcx, substs1);
                let sig2 = self.tcx.bound_fn_sig(*did2).subst(self.tcx, substs2);
                if sig1 != sig2 {
                    return;
                }
                err.note(
                    "different `fn` items always have unique types, even if their signatures are \
                     the same",
                );
                (sig1, *did1, substs1)
            }
            (ty::FnDef(did, substs), ty::FnPtr(sig2)) => {
                let sig1 = self.tcx.bound_fn_sig(*did).subst(self.tcx, substs);
                if sig1 != *sig2 {
                    return;
                }
                (sig1, *did, substs)
            }
            _ => return,
        };
        err.help(&format!("change the expected type to be function pointer `{}`", sig));
        err.help(&format!(
            "if the expected type is due to type inference, cast the expected `fn` to a function \
             pointer: `{} as {}`",
            self.tcx.def_path_str_with_substs(did, substs),
            sig
        ));
    }

    // Instantiates the given path, which must refer to an item with the given
    // number of type parameters and type.
    #[instrument(skip(self, span), level = "debug")]
    pub fn instantiate_value_path(
        &self,
        segments: &[hir::PathSegment<'_>],
        self_ty: Option<Ty<'tcx>>,
        res: Res,
        span: Span,
        hir_id: hir::HirId,
    ) -> (Ty<'tcx>, Res) {
        let tcx = self.tcx;

        let path_segs = match res {
            Res::Local(_) | Res::SelfCtor(_) => vec![],
            Res::Def(kind, def_id) => <dyn AstConv<'_>>::def_ids_for_value_path_segments(
                self, segments, self_ty, kind, def_id,
            ),
            _ => bug!("instantiate_value_path on {:?}", res),
        };

        let mut user_self_ty = None;
        let mut is_alias_variant_ctor = false;
        match res {
            Res::Def(DefKind::Ctor(CtorOf::Variant, _), _)
                if let Some(self_ty) = self_ty =>
            {
                let adt_def = self_ty.ty_adt_def().unwrap();
                user_self_ty = Some(UserSelfTy { impl_def_id: adt_def.did(), self_ty });
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
                            let self_ty = self_ty.expect("UFCS sugared assoc missing Self");
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
        let generics_has_err = <dyn AstConv<'_>>::prohibit_generics(
            self,
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
            let ty = self.local_ty(span, hid).decl_ty;
            let ty = self.normalize_associated_types_in(span, ty);
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
            let arg_count = <dyn AstConv<'_>>::check_generic_arg_count_for_call(
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

            if let Err(GenericArgCountMismatch { reported: Some(_), .. }) = arg_count.correct {
                infer_args_for_err.insert(index);
                self.set_tainted_by_errors(); // See issue #53251.
            }
        }

        let has_self = path_segs
            .last()
            .map(|PathSeg(def_id, _)| tcx.generics_of(*def_id).has_self)
            .unwrap_or(false);

        let (res, self_ctor_substs) = if let Res::SelfCtor(impl_def_id) = res {
            let ty = self.normalize_ty(span, tcx.at(span).type_of(impl_def_id));
            match *ty.kind() {
                ty::Adt(adt_def, substs) if adt_def.has_ctor() => {
                    let variant = adt_def.non_enum_variant();
                    let ctor_def_id = variant.ctor_def_id.unwrap();
                    (
                        Res::Def(DefKind::Ctor(CtorOf::Struct, variant.ctor_kind), ctor_def_id),
                        Some(substs),
                    )
                }
                _ => {
                    let mut err = tcx.sess.struct_span_err(
                        span,
                        "the `Self` constructor can only be used with tuple or unit structs",
                    );
                    if let Some(adt_def) = ty.ty_adt_def() {
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
                    err.emit();

                    return (tcx.ty_error(), res);
                }
            }
        } else {
            (res, None)
        };
        let def_id = res.def_id();

        // The things we are substituting into the type should not contain
        // escaping late-bound regions, and nor should the base type scheme.
        let ty = tcx.type_of(def_id);

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
            ) -> subst::GenericArg<'tcx> {
                match (&param.kind, arg) {
                    (GenericParamDefKind::Lifetime, GenericArg::Lifetime(lt)) => {
                        <dyn AstConv<'_>>::ast_region_to_region(self.fcx, lt, Some(param)).into()
                    }
                    (GenericParamDefKind::Type { .. }, GenericArg::Type(ty)) => {
                        self.fcx.to_ty(ty).into()
                    }
                    (GenericParamDefKind::Const { .. }, GenericArg::Const(ct)) => {
                        self.fcx.const_arg_to_const(&ct.value, param.def_id).into()
                    }
                    (GenericParamDefKind::Type { .. }, GenericArg::Infer(inf)) => {
                        self.fcx.ty_infer(Some(param), inf.span).into()
                    }
                    (GenericParamDefKind::Const { .. }, GenericArg::Infer(inf)) => {
                        let tcx = self.fcx.tcx();
                        self.fcx.ct_infer(tcx.type_of(param.def_id), Some(param), inf.span).into()
                    }
                    _ => unreachable!(),
                }
            }

            fn inferred_kind(
                &mut self,
                substs: Option<&[subst::GenericArg<'tcx>]>,
                param: &ty::GenericParamDef,
                infer_args: bool,
            ) -> subst::GenericArg<'tcx> {
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
                            let default = tcx.bound_type_of(param.def_id);
                            self.fcx
                                .normalize_ty(self.span, default.subst(tcx, substs.unwrap()))
                                .into()
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
                            tcx.bound_const_param_default(param.def_id)
                                .subst(tcx, substs.unwrap())
                                .into()
                        } else {
                            self.fcx.var_for_def(self.span, param)
                        }
                    }
                }
            }
        }

        let substs = self_ctor_substs.unwrap_or_else(|| {
            <dyn AstConv<'_>>::create_substs_for_generic_args(
                tcx,
                def_id,
                &[],
                has_self,
                self_ty,
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
        assert!(!substs.has_escaping_bound_vars());
        assert!(!ty.has_escaping_bound_vars());

        // First, store the "user substs" for later.
        self.write_user_type_annotation_from_substs(hir_id, def_id, substs, user_self_ty);

        self.add_required_obligations_for_hir(span, def_id, &substs, hir_id);

        // Substitute the values for the type parameters into the type of
        // the referenced item.
        let ty_substituted = self.instantiate_type_scheme(span, &substs, ty);

        if let Some(UserSelfTy { impl_def_id, self_ty }) = user_self_ty {
            // In the case of `Foo<T>::method` and `<Foo<T>>::method`, if `method`
            // is inherent, there is no `Self` parameter; instead, the impl needs
            // type parameters, which we can infer by unifying the provided `Self`
            // with the substituted impl type.
            // This also occurs for an enum variant on a type alias.
            let ty = tcx.type_of(impl_def_id);

            let impl_ty = self.instantiate_type_scheme(span, &substs, ty);
            match self.at(&self.misc(span), self.param_env).eq(impl_ty, self_ty) {
                Ok(ok) => self.register_infer_ok_obligations(ok),
                Err(_) => {
                    self.tcx.sess.delay_span_bug(
                        span,
                        &format!(
                        "instantiate_value_path: (UFCS) {:?} was a subtype of {:?} but now is not?",
                        self_ty,
                        impl_ty,
                    ),
                    );
                }
            }
        }

        debug!("instantiate_value_path: type of {:?} is {:?}", hir_id, ty_substituted);
        self.write_substs(hir_id, substs);

        (ty_substituted, res)
    }

    /// Add all the obligations that are required, substituting and normalized appropriately.
    pub(crate) fn add_required_obligations_for_hir(
        &self,
        span: Span,
        def_id: DefId,
        substs: SubstsRef<'tcx>,
        hir_id: hir::HirId,
    ) {
        self.add_required_obligations_with_code(span, def_id, substs, |idx, span| {
            if span.is_dummy() {
                ObligationCauseCode::ExprItemObligation(def_id, hir_id, idx)
            } else {
                ObligationCauseCode::ExprBindingObligation(def_id, span, hir_id, idx)
            }
        })
    }

    #[instrument(level = "debug", skip(self, code, span, def_id, substs))]
    fn add_required_obligations_with_code(
        &self,
        span: Span,
        def_id: DefId,
        substs: SubstsRef<'tcx>,
        code: impl Fn(usize, Span) -> ObligationCauseCode<'tcx>,
    ) {
        let (bounds, _) = self.instantiate_bounds(span, def_id, &substs);

        for obligation in traits::predicates_for_generics(
            |idx, predicate_span| {
                traits::ObligationCause::new(span, self.body_id, code(idx, predicate_span))
            },
            self.param_env,
            bounds,
        ) {
            self.register_predicate(obligation);
        }
    }

    /// Resolves `typ` by a single level if `typ` is a type variable.
    /// If no resolution is possible, then an error is reported.
    /// Numeric inference variables may be left unresolved.
    pub fn structurally_resolved_type(&self, sp: Span, ty: Ty<'tcx>) -> Ty<'tcx> {
        let ty = self.resolve_vars_with_obligations(ty);
        if !ty.is_ty_var() {
            ty
        } else {
            if !self.is_tainted_by_errors() {
                self.emit_inference_failure_err((**self).body_id, sp, ty.into(), E0282, true)
                    .emit();
            }
            let err = self.tcx.ty_error();
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

        while let hir::Node::Expr(parent_expr) =
            self.tcx.hir().get(self.tcx.hir().get_parent_node(expr_id))
        {
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
