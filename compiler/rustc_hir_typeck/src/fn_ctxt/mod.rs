mod _impl;
mod adjust_fulfillment_errors;
mod arg_matrix;
mod checks;
mod inspect_obligations;
mod suggestions;

use std::cell::{Cell, RefCell};
use std::ops::Deref;

use hir::def_id::CRATE_DEF_ID;
use rustc_errors::DiagCtxtHandle;
use rustc_hir::def_id::{DefId, LocalDefId};
use rustc_hir::{self as hir, HirId, ItemLocalMap};
use rustc_hir_analysis::hir_ty_lowering::{
    HirTyLowerer, InherentAssocCandidate, RegionInferReason,
};
use rustc_infer::infer;
use rustc_infer::traits::{DynCompatibilityViolation, Obligation};
use rustc_middle::ty::{self, Const, Ty, TyCtxt, TypeVisitableExt};
use rustc_session::Session;
use rustc_span::{self, DUMMY_SP, ErrorGuaranteed, Ident, Span, sym};
use rustc_trait_selection::error_reporting::TypeErrCtxt;
use rustc_trait_selection::error_reporting::infer::sub_relations::SubRelations;
use rustc_trait_selection::traits::{
    self, FulfillmentError, ObligationCause, ObligationCauseCode, ObligationCtxt,
};

use crate::coercion::DynamicCoerceMany;
use crate::fallback::DivergingFallbackBehavior;
use crate::fn_ctxt::checks::DivergingBlockBehavior;
use crate::{CoroutineTypes, Diverges, EnclosingBreakables, TypeckRootCtxt};

/// The `FnCtxt` stores type-checking context needed to type-check bodies of
/// functions, closures, and `const`s, including performing type inference
/// with [`InferCtxt`].
///
/// This is in contrast to `rustc_hir_analysis::collect::ItemCtxt`, which is
/// used to type-check item *signatures* and thus does not perform type
/// inference.
///
/// See `ItemCtxt`'s docs for more.
///
/// [`InferCtxt`]: infer::InferCtxt
pub(crate) struct FnCtxt<'a, 'tcx> {
    pub(super) body_id: LocalDefId,

    /// The parameter environment used for proving trait obligations
    /// in this function. This can change when we descend into
    /// closures (as they bring new things into scope), hence it is
    /// not part of `Inherited` (as of the time of this writing,
    /// closures do not yet change the environment, but they will
    /// eventually).
    pub(super) param_env: ty::ParamEnv<'tcx>,

    /// If `Some`, this stores coercion information for returned
    /// expressions. If `None`, this is in a context where return is
    /// inappropriate, such as a const expression.
    ///
    /// This is a `RefCell<DynamicCoerceMany>`, which means that we
    /// can track all the return expressions and then use them to
    /// compute a useful coercion from the set, similar to a match
    /// expression or other branching context. You can use methods
    /// like `expected_ty` to access the declared return type (if
    /// any).
    pub(super) ret_coercion: Option<RefCell<DynamicCoerceMany<'tcx>>>,

    /// First span of a return site that we find. Used in error messages.
    pub(super) ret_coercion_span: Cell<Option<Span>>,

    pub(super) coroutine_types: Option<CoroutineTypes<'tcx>>,

    /// Whether the last checked node generates a divergence (e.g.,
    /// `return` will set this to `Always`). In general, when entering
    /// an expression or other node in the tree, the initial value
    /// indicates whether prior parts of the containing expression may
    /// have diverged. It is then typically set to `Maybe` (and the
    /// old value remembered) for processing the subparts of the
    /// current expression. As each subpart is processed, they may set
    /// the flag to `Always`, etc. Finally, at the end, we take the
    /// result and "union" it with the original value, so that when we
    /// return the flag indicates if any subpart of the parent
    /// expression (up to and including this part) has diverged. So,
    /// if you read it after evaluating a subexpression `X`, the value
    /// you get indicates whether any subexpression that was
    /// evaluating up to and including `X` diverged.
    ///
    /// We currently use this flag for the following purposes:
    ///
    /// - To warn about unreachable code: if, after processing a
    ///   sub-expression but before we have applied the effects of the
    ///   current node, we see that the flag is set to `Always`, we
    ///   can issue a warning. This corresponds to something like
    ///   `foo(return)`; we warn on the `foo()` expression. (We then
    ///   update the flag to `WarnedAlways` to suppress duplicate
    ///   reports.) Similarly, if we traverse to a fresh statement (or
    ///   tail expression) from an `Always` setting, we will issue a
    ///   warning. This corresponds to something like `{return;
    ///   foo();}` or `{return; 22}`, where we would warn on the
    ///   `foo()` or `22`.
    /// - To assign the `!` type to block expressions with diverging
    ///   statements.
    ///
    /// An expression represents dead code if, after checking it,
    /// the diverges flag is set to something other than `Maybe`.
    pub(super) diverges: Cell<Diverges>,

    /// If one of the function arguments is a never pattern, this counts as diverging code. This
    /// affect typechecking of the function body.
    pub(super) function_diverges_because_of_empty_arguments: Cell<Diverges>,

    /// Whether the currently checked node is the whole body of the function.
    pub(super) is_whole_body: Cell<bool>,

    pub(super) enclosing_breakables: RefCell<EnclosingBreakables<'tcx>>,

    pub(super) root_ctxt: &'a TypeckRootCtxt<'tcx>,

    pub(super) fallback_has_occurred: Cell<bool>,

    pub(super) diverging_fallback_behavior: DivergingFallbackBehavior,
    pub(super) diverging_block_behavior: DivergingBlockBehavior,

    /// Clauses that we lowered as part of the `impl_trait_in_bindings` feature.
    ///
    /// These are stored here so we may collect them when canonicalizing user
    /// type ascriptions later.
    pub(super) trait_ascriptions: RefCell<ItemLocalMap<Vec<ty::Clause<'tcx>>>>,
}

impl<'a, 'tcx> FnCtxt<'a, 'tcx> {
    pub(crate) fn new(
        root_ctxt: &'a TypeckRootCtxt<'tcx>,
        param_env: ty::ParamEnv<'tcx>,
        body_id: LocalDefId,
    ) -> FnCtxt<'a, 'tcx> {
        let (diverging_fallback_behavior, diverging_block_behavior) =
            never_type_behavior(root_ctxt.tcx);
        FnCtxt {
            body_id,
            param_env,
            ret_coercion: None,
            ret_coercion_span: Cell::new(None),
            coroutine_types: None,
            diverges: Cell::new(Diverges::Maybe),
            function_diverges_because_of_empty_arguments: Cell::new(Diverges::Maybe),
            is_whole_body: Cell::new(false),
            enclosing_breakables: RefCell::new(EnclosingBreakables {
                stack: Vec::new(),
                by_id: Default::default(),
            }),
            root_ctxt,
            fallback_has_occurred: Cell::new(false),
            diverging_fallback_behavior,
            diverging_block_behavior,
            trait_ascriptions: Default::default(),
        }
    }

    pub(crate) fn dcx(&self) -> DiagCtxtHandle<'a> {
        self.root_ctxt.infcx.dcx()
    }

    pub(crate) fn cause(
        &self,
        span: Span,
        code: ObligationCauseCode<'tcx>,
    ) -> ObligationCause<'tcx> {
        ObligationCause::new(span, self.body_id, code)
    }

    pub(crate) fn misc(&self, span: Span) -> ObligationCause<'tcx> {
        self.cause(span, ObligationCauseCode::Misc)
    }

    pub(crate) fn sess(&self) -> &Session {
        self.tcx.sess
    }

    /// Creates an `TypeErrCtxt` with a reference to the in-progress
    /// `TypeckResults` which is used for diagnostics.
    /// Use [`InferCtxtErrorExt::err_ctxt`] to start one without a `TypeckResults`.
    ///
    /// [`InferCtxtErrorExt::err_ctxt`]: rustc_trait_selection::error_reporting::InferCtxtErrorExt::err_ctxt
    pub(crate) fn err_ctxt(&'a self) -> TypeErrCtxt<'a, 'tcx> {
        let mut sub_relations = SubRelations::default();
        sub_relations.add_constraints(
            self,
            self.fulfillment_cx.borrow_mut().pending_obligations().iter().map(|o| o.predicate),
        );
        TypeErrCtxt {
            infcx: &self.infcx,
            sub_relations: RefCell::new(sub_relations),
            typeck_results: Some(self.typeck_results.borrow()),
            fallback_has_occurred: self.fallback_has_occurred.get(),
            normalize_fn_sig: Box::new(|fn_sig| {
                if fn_sig.has_escaping_bound_vars() {
                    return fn_sig;
                }
                self.probe(|_| {
                    let ocx = ObligationCtxt::new(self);
                    let normalized_fn_sig =
                        ocx.normalize(&ObligationCause::dummy(), self.param_env, fn_sig);
                    if ocx.select_all_or_error().is_empty() {
                        let normalized_fn_sig = self.resolve_vars_if_possible(normalized_fn_sig);
                        if !normalized_fn_sig.has_infer() {
                            return normalized_fn_sig;
                        }
                    }
                    fn_sig
                })
            }),
            autoderef_steps: Box::new(|ty| {
                let mut autoderef = self.autoderef(DUMMY_SP, ty).silence_errors();
                let mut steps = vec![];
                while let Some((ty, _)) = autoderef.next() {
                    steps.push((ty, autoderef.current_obligations()));
                }
                steps
            }),
        }
    }
}

impl<'a, 'tcx> Deref for FnCtxt<'a, 'tcx> {
    type Target = TypeckRootCtxt<'tcx>;
    fn deref(&self) -> &Self::Target {
        self.root_ctxt
    }
}

impl<'tcx> HirTyLowerer<'tcx> for FnCtxt<'_, 'tcx> {
    fn tcx(&self) -> TyCtxt<'tcx> {
        self.tcx
    }

    fn dcx(&self) -> DiagCtxtHandle<'_> {
        self.root_ctxt.dcx()
    }

    fn item_def_id(&self) -> LocalDefId {
        self.body_id
    }

    fn re_infer(&self, span: Span, reason: RegionInferReason<'_>) -> ty::Region<'tcx> {
        let v = match reason {
            RegionInferReason::Param(def) => infer::RegionParameterDefinition(span, def.name),
            _ => infer::MiscVariable(span),
        };
        self.next_region_var(v)
    }

    fn ty_infer(&self, param: Option<&ty::GenericParamDef>, span: Span) -> Ty<'tcx> {
        match param {
            Some(param) => self.var_for_def(span, param).as_type().unwrap(),
            None => self.next_ty_var(span),
        }
    }

    fn ct_infer(&self, param: Option<&ty::GenericParamDef>, span: Span) -> Const<'tcx> {
        // FIXME ideally this shouldn't use unwrap
        match param {
            Some(param) => self.var_for_def(span, param).as_const().unwrap(),
            None => self.next_const_var(span),
        }
    }

    fn register_trait_ascription_bounds(
        &self,
        bounds: Vec<(ty::Clause<'tcx>, Span)>,
        hir_id: HirId,
        _span: Span,
    ) {
        for (clause, span) in bounds {
            if clause.has_escaping_bound_vars() {
                self.dcx().span_delayed_bug(span, "clause should have no escaping bound vars");
                continue;
            }

            self.trait_ascriptions.borrow_mut().entry(hir_id.local_id).or_default().push(clause);

            let clause = self.normalize(span, clause);
            self.register_predicate(Obligation::new(
                self.tcx,
                self.misc(span),
                self.param_env,
                clause,
            ));
        }
    }

    fn probe_ty_param_bounds(
        &self,
        _: Span,
        def_id: LocalDefId,
        _: Ident,
    ) -> ty::EarlyBinder<'tcx, &'tcx [(ty::Clause<'tcx>, Span)]> {
        let tcx = self.tcx;
        let item_def_id = tcx.hir_ty_param_owner(def_id);
        let generics = tcx.generics_of(item_def_id);
        let index = generics.param_def_id_to_index[&def_id.to_def_id()];
        // HACK(eddyb) should get the original `Span`.
        let span = tcx.def_span(def_id);

        ty::EarlyBinder::bind(tcx.arena.alloc_from_iter(
            self.param_env.caller_bounds().iter().filter_map(|predicate| {
                match predicate.kind().skip_binder() {
                    ty::ClauseKind::Trait(data) if data.self_ty().is_param(index) => {
                        Some((predicate, span))
                    }
                    _ => None,
                }
            }),
        ))
    }

    fn select_inherent_assoc_candidates(
        &self,
        span: Span,
        self_ty: Ty<'tcx>,
        candidates: Vec<InherentAssocCandidate>,
    ) -> (Vec<InherentAssocCandidate>, Vec<FulfillmentError<'tcx>>) {
        let tcx = self.tcx();
        let infcx = &self.infcx;
        let mut fulfillment_errors = vec![];

        let mut filter_iat_candidate = |self_ty, impl_| {
            let ocx = ObligationCtxt::new_with_diagnostics(self);
            let self_ty = ocx.normalize(&ObligationCause::dummy(), self.param_env, self_ty);

            let impl_args = infcx.fresh_args_for_item(span, impl_);
            let impl_ty = tcx.type_of(impl_).instantiate(tcx, impl_args);
            let impl_ty = ocx.normalize(&ObligationCause::dummy(), self.param_env, impl_ty);

            // Check that the self types can be related.
            if ocx.eq(&ObligationCause::dummy(), self.param_env, impl_ty, self_ty).is_err() {
                return false;
            }

            // Check whether the impl imposes obligations we have to worry about.
            let impl_bounds = tcx.predicates_of(impl_).instantiate(tcx, impl_args);
            let impl_bounds = ocx.normalize(&ObligationCause::dummy(), self.param_env, impl_bounds);
            let impl_obligations = traits::predicates_for_generics(
                |_, _| ObligationCause::dummy(),
                self.param_env,
                impl_bounds,
            );
            ocx.register_obligations(impl_obligations);

            let mut errors = ocx.select_where_possible();
            if !errors.is_empty() {
                fulfillment_errors.append(&mut errors);
                return false;
            }

            true
        };

        let mut universes = if self_ty.has_escaping_bound_vars() {
            vec![None; self_ty.outer_exclusive_binder().as_usize()]
        } else {
            vec![]
        };

        let candidates =
            traits::with_replaced_escaping_bound_vars(infcx, &mut universes, self_ty, |self_ty| {
                candidates
                    .into_iter()
                    .filter(|&InherentAssocCandidate { impl_, .. }| {
                        infcx.probe(|_| filter_iat_candidate(self_ty, impl_))
                    })
                    .collect()
            });

        (candidates, fulfillment_errors)
    }

    fn lower_assoc_item_path(
        &self,
        span: Span,
        item_def_id: DefId,
        item_segment: &rustc_hir::PathSegment<'tcx>,
        poly_trait_ref: ty::PolyTraitRef<'tcx>,
    ) -> Result<(DefId, ty::GenericArgsRef<'tcx>), ErrorGuaranteed> {
        let trait_ref = self.instantiate_binder_with_fresh_vars(
            span,
            // FIXME(mgca): `item_def_id` can be an AssocConst; rename this variant.
            infer::BoundRegionConversionTime::AssocTypeProjection(item_def_id),
            poly_trait_ref,
        );

        let item_args = self.lowerer().lower_generic_args_of_assoc_item(
            span,
            item_def_id,
            item_segment,
            trait_ref.args,
        );

        Ok((item_def_id, item_args))
    }

    fn probe_adt(&self, span: Span, ty: Ty<'tcx>) -> Option<ty::AdtDef<'tcx>> {
        match ty.kind() {
            ty::Adt(adt_def, _) => Some(*adt_def),
            // FIXME(#104767): Should we handle bound regions here?
            ty::Alias(ty::Projection | ty::Inherent | ty::Free, _)
                if !ty.has_escaping_bound_vars() =>
            {
                if self.next_trait_solver() {
                    self.try_structurally_resolve_type(span, ty).ty_adt_def()
                } else {
                    self.normalize(span, ty).ty_adt_def()
                }
            }
            _ => None,
        }
    }

    fn record_ty(&self, hir_id: hir::HirId, ty: Ty<'tcx>, span: Span) {
        // FIXME: normalization and escaping regions
        let ty = if !ty.has_escaping_bound_vars() {
            // NOTE: These obligations are 100% redundant and are implied by
            // WF obligations that are registered elsewhere, but they have a
            // better cause code assigned to them in `add_required_obligations_for_hir`.
            // This means that they should shadow obligations with worse spans.
            if let ty::Alias(ty::Projection | ty::Free, ty::AliasTy { args, def_id, .. }) =
                ty.kind()
            {
                self.add_required_obligations_for_hir(span, *def_id, args, hir_id);
            }

            self.normalize(span, ty)
        } else {
            ty
        };
        self.write_ty(hir_id, ty)
    }

    fn infcx(&self) -> Option<&infer::InferCtxt<'tcx>> {
        Some(&self.infcx)
    }

    fn lower_fn_sig(
        &self,
        decl: &rustc_hir::FnDecl<'tcx>,
        _generics: Option<&rustc_hir::Generics<'_>>,
        _hir_id: rustc_hir::HirId,
        _hir_ty: Option<&hir::Ty<'_>>,
    ) -> (Vec<Ty<'tcx>>, Ty<'tcx>) {
        let input_tys = decl.inputs.iter().map(|a| self.lowerer().lower_ty(a)).collect();

        let output_ty = match decl.output {
            hir::FnRetTy::Return(output) => self.lowerer().lower_ty(output),
            hir::FnRetTy::DefaultReturn(..) => self.tcx().types.unit,
        };
        (input_tys, output_ty)
    }

    fn dyn_compatibility_violations(&self, trait_def_id: DefId) -> Vec<DynCompatibilityViolation> {
        self.tcx.dyn_compatibility_violations(trait_def_id).to_vec()
    }
}

/// The `ty` representation of a user-provided type. Depending on the use-site
/// we want to either use the unnormalized or the normalized form of this type.
///
/// This is a bridge between the interface of HIR ty lowering, which outputs a raw
/// `Ty`, and the API in this module, which expect `Ty` to be fully normalized.
#[derive(Clone, Copy, Debug)]
pub(crate) struct LoweredTy<'tcx> {
    /// The unnormalized type provided by the user.
    pub raw: Ty<'tcx>,

    /// The normalized form of `raw`, stored here for efficiency.
    pub normalized: Ty<'tcx>,
}

impl<'tcx> LoweredTy<'tcx> {
    fn from_raw(fcx: &FnCtxt<'_, 'tcx>, span: Span, raw: Ty<'tcx>) -> LoweredTy<'tcx> {
        // FIXME(-Znext-solver=no): This is easier than requiring all uses of `LoweredTy`
        // to call `try_structurally_resolve_type` instead. This seems like a lot of
        // effort, especially as we're still supporting the old solver. We may revisit
        // this in the future.
        let normalized = if fcx.next_trait_solver() {
            fcx.try_structurally_resolve_type(span, raw)
        } else {
            fcx.normalize(span, raw)
        };
        LoweredTy { raw, normalized }
    }
}

fn never_type_behavior(tcx: TyCtxt<'_>) -> (DivergingFallbackBehavior, DivergingBlockBehavior) {
    let (fallback, block) = parse_never_type_options_attr(tcx);
    let fallback = fallback.unwrap_or_else(|| default_fallback(tcx));
    let block = block.unwrap_or_default();

    (fallback, block)
}

/// Returns the default fallback which is used when there is no explicit override via `#![never_type_options(...)]`.
fn default_fallback(tcx: TyCtxt<'_>) -> DivergingFallbackBehavior {
    // Edition 2024: fallback to `!`
    if tcx.sess.edition().at_least_rust_2024() {
        return DivergingFallbackBehavior::ToNever;
    }

    // `feature(never_type_fallback)`: fallback to `!` or `()` trying to not break stuff
    if tcx.features().never_type_fallback() {
        return DivergingFallbackBehavior::ContextDependent;
    }

    // Otherwise: fallback to `()`
    DivergingFallbackBehavior::ToUnit
}

fn parse_never_type_options_attr(
    tcx: TyCtxt<'_>,
) -> (Option<DivergingFallbackBehavior>, Option<DivergingBlockBehavior>) {
    // Error handling is dubious here (unwraps), but that's probably fine for an internal attribute.
    // Just don't write incorrect attributes <3

    let mut fallback = None;
    let mut block = None;

    let items = tcx
        .get_attr(CRATE_DEF_ID, sym::rustc_never_type_options)
        .map(|attr| attr.meta_item_list().unwrap())
        .unwrap_or_default();

    for item in items {
        if item.has_name(sym::fallback) && fallback.is_none() {
            let mode = item.value_str().unwrap();
            match mode {
                sym::unit => fallback = Some(DivergingFallbackBehavior::ToUnit),
                sym::niko => fallback = Some(DivergingFallbackBehavior::ContextDependent),
                sym::never => fallback = Some(DivergingFallbackBehavior::ToNever),
                sym::no => fallback = Some(DivergingFallbackBehavior::NoFallback),
                _ => {
                    tcx.dcx().span_err(item.span(), format!("unknown never type fallback mode: `{mode}` (supported: `unit`, `niko`, `never` and `no`)"));
                }
            };
            continue;
        }

        if item.has_name(sym::diverging_block_default) && block.is_none() {
            let default = item.value_str().unwrap();
            match default {
                sym::unit => block = Some(DivergingBlockBehavior::Unit),
                sym::never => block = Some(DivergingBlockBehavior::Never),
                _ => {
                    tcx.dcx().span_err(item.span(), format!("unknown diverging block default: `{default}` (supported: `unit` and `never`)"));
                }
            };
            continue;
        }

        tcx.dcx().span_err(
            item.span(),
            format!(
                "unknown or duplicate never type option: `{}` (supported: `fallback`, `diverging_block_default`)",
                item.name().unwrap()
            ),
        );
    }

    (fallback, block)
}
