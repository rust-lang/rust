use super::potentially_plural_count;
use crate::errors::LifetimesOrBoundsMismatchOnTrait;
use hir::def_id::DefId;
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_errors::{pluralize, struct_span_err, Applicability, DiagnosticId, ErrorGuaranteed};
use rustc_hir as hir;
use rustc_hir::def::{DefKind, Res};
use rustc_hir::intravisit;
use rustc_hir::{GenericParamKind, ImplItemKind, TraitItemKind};
use rustc_infer::infer::outlives::env::OutlivesEnvironment;
use rustc_infer::infer::type_variable::{TypeVariableOrigin, TypeVariableOriginKind};
use rustc_infer::infer::{self, TyCtxtInferExt};
use rustc_infer::traits::util;
use rustc_middle::ty::error::{ExpectedFound, TypeError};
use rustc_middle::ty::subst::{InternalSubsts, Subst};
use rustc_middle::ty::util::ExplicitSelf;
use rustc_middle::ty::{
    self, AssocItem, DefIdTree, Ty, TypeFoldable, TypeFolder, TypeSuperFoldable, TypeVisitable,
};
use rustc_middle::ty::{GenericParamDefKind, ToPredicate, TyCtxt};
use rustc_span::Span;
use rustc_trait_selection::traits::error_reporting::InferCtxtExt;
use rustc_trait_selection::traits::outlives_bounds::InferCtxtExt as _;
use rustc_trait_selection::traits::{
    self, ObligationCause, ObligationCauseCode, ObligationCtxt, Reveal,
};
use std::iter;

/// Checks that a method from an impl conforms to the signature of
/// the same method as declared in the trait.
///
/// # Parameters
///
/// - `impl_m`: type of the method we are checking
/// - `impl_m_span`: span to use for reporting errors
/// - `trait_m`: the method in the trait
/// - `impl_trait_ref`: the TraitRef corresponding to the trait implementation
pub(crate) fn compare_impl_method<'tcx>(
    tcx: TyCtxt<'tcx>,
    impl_m: &ty::AssocItem,
    trait_m: &ty::AssocItem,
    impl_trait_ref: ty::TraitRef<'tcx>,
    trait_item_span: Option<Span>,
) {
    debug!("compare_impl_method(impl_trait_ref={:?})", impl_trait_ref);

    let impl_m_span = tcx.def_span(impl_m.def_id);

    if let Err(_) = compare_self_type(tcx, impl_m, impl_m_span, trait_m, impl_trait_ref) {
        return;
    }

    if let Err(_) = compare_number_of_generics(tcx, impl_m, impl_m_span, trait_m, trait_item_span) {
        return;
    }

    if let Err(_) = compare_generic_param_kinds(tcx, impl_m, trait_m) {
        return;
    }

    if let Err(_) =
        compare_number_of_method_arguments(tcx, impl_m, impl_m_span, trait_m, trait_item_span)
    {
        return;
    }

    if let Err(_) = compare_synthetic_generics(tcx, impl_m, trait_m) {
        return;
    }

    if let Err(_) = compare_predicate_entailment(tcx, impl_m, impl_m_span, trait_m, impl_trait_ref)
    {
        return;
    }
}

/// This function is best explained by example. Consider a trait:
///
///     trait Trait<'t, T> {
///         // `trait_m`
///         fn method<'a, M>(t: &'t T, m: &'a M) -> Self;
///     }
///
/// And an impl:
///
///     impl<'i, 'j, U> Trait<'j, &'i U> for Foo {
///          // `impl_m`
///          fn method<'b, N>(t: &'j &'i U, m: &'b N) -> Foo;
///     }
///
/// We wish to decide if those two method types are compatible.
/// For this we have to show that, assuming the bounds of the impl hold, the
/// bounds of `trait_m` imply the bounds of `impl_m`.
///
/// We start out with `trait_to_impl_substs`, that maps the trait
/// type parameters to impl type parameters. This is taken from the
/// impl trait reference:
///
///     trait_to_impl_substs = {'t => 'j, T => &'i U, Self => Foo}
///
/// We create a mapping `dummy_substs` that maps from the impl type
/// parameters to fresh types and regions. For type parameters,
/// this is the identity transform, but we could as well use any
/// placeholder types. For regions, we convert from bound to free
/// regions (Note: but only early-bound regions, i.e., those
/// declared on the impl or used in type parameter bounds).
///
///     impl_to_placeholder_substs = {'i => 'i0, U => U0, N => N0 }
///
/// Now we can apply `placeholder_substs` to the type of the impl method
/// to yield a new function type in terms of our fresh, placeholder
/// types:
///
///     <'b> fn(t: &'i0 U0, m: &'b) -> Foo
///
/// We now want to extract and substitute the type of the *trait*
/// method and compare it. To do so, we must create a compound
/// substitution by combining `trait_to_impl_substs` and
/// `impl_to_placeholder_substs`, and also adding a mapping for the method
/// type parameters. We extend the mapping to also include
/// the method parameters.
///
///     trait_to_placeholder_substs = { T => &'i0 U0, Self => Foo, M => N0 }
///
/// Applying this to the trait method type yields:
///
///     <'a> fn(t: &'i0 U0, m: &'a) -> Foo
///
/// This type is also the same but the name of the bound region (`'a`
/// vs `'b`).  However, the normal subtyping rules on fn types handle
/// this kind of equivalency just fine.
///
/// We now use these substitutions to ensure that all declared bounds are
/// satisfied by the implementation's method.
///
/// We do this by creating a parameter environment which contains a
/// substitution corresponding to `impl_to_placeholder_substs`. We then build
/// `trait_to_placeholder_substs` and use it to convert the predicates contained
/// in the `trait_m` generics to the placeholder form.
///
/// Finally we register each of these predicates as an obligation and check that
/// they hold.
fn compare_predicate_entailment<'tcx>(
    tcx: TyCtxt<'tcx>,
    impl_m: &AssocItem,
    impl_m_span: Span,
    trait_m: &AssocItem,
    impl_trait_ref: ty::TraitRef<'tcx>,
) -> Result<(), ErrorGuaranteed> {
    let trait_to_impl_substs = impl_trait_ref.substs;

    // This node-id should be used for the `body_id` field on each
    // `ObligationCause` (and the `FnCtxt`).
    //
    // FIXME(@lcnr): remove that after removing `cause.body_id` from
    // obligations.
    let impl_m_hir_id = tcx.hir().local_def_id_to_hir_id(impl_m.def_id.expect_local());
    // We sometimes modify the span further down.
    let mut cause = ObligationCause::new(
        impl_m_span,
        impl_m_hir_id,
        ObligationCauseCode::CompareImplItemObligation {
            impl_item_def_id: impl_m.def_id.expect_local(),
            trait_item_def_id: trait_m.def_id,
            kind: impl_m.kind,
        },
    );

    // Create mapping from impl to placeholder.
    let impl_to_placeholder_substs = InternalSubsts::identity_for_item(tcx, impl_m.def_id);

    // Create mapping from trait to placeholder.
    let trait_to_placeholder_substs =
        impl_to_placeholder_substs.rebase_onto(tcx, impl_m.container_id(tcx), trait_to_impl_substs);
    debug!("compare_impl_method: trait_to_placeholder_substs={:?}", trait_to_placeholder_substs);

    let impl_m_generics = tcx.generics_of(impl_m.def_id);
    let trait_m_generics = tcx.generics_of(trait_m.def_id);
    let impl_m_predicates = tcx.predicates_of(impl_m.def_id);
    let trait_m_predicates = tcx.predicates_of(trait_m.def_id);

    // Check region bounds.
    check_region_bounds_on_impl_item(tcx, impl_m, trait_m, &trait_m_generics, &impl_m_generics)?;

    // Create obligations for each predicate declared by the impl
    // definition in the context of the trait's parameter
    // environment. We can't just use `impl_env.caller_bounds`,
    // however, because we want to replace all late-bound regions with
    // region variables.
    let impl_predicates = tcx.predicates_of(impl_m_predicates.parent.unwrap());
    let mut hybrid_preds = impl_predicates.instantiate_identity(tcx);

    debug!("compare_impl_method: impl_bounds={:?}", hybrid_preds);

    // This is the only tricky bit of the new way we check implementation methods
    // We need to build a set of predicates where only the method-level bounds
    // are from the trait and we assume all other bounds from the implementation
    // to be previously satisfied.
    //
    // We then register the obligations from the impl_m and check to see
    // if all constraints hold.
    hybrid_preds
        .predicates
        .extend(trait_m_predicates.instantiate_own(tcx, trait_to_placeholder_substs).predicates);

    // Construct trait parameter environment and then shift it into the placeholder viewpoint.
    // The key step here is to update the caller_bounds's predicates to be
    // the new hybrid bounds we computed.
    let normalize_cause = traits::ObligationCause::misc(impl_m_span, impl_m_hir_id);
    let param_env = ty::ParamEnv::new(
        tcx.intern_predicates(&hybrid_preds.predicates),
        Reveal::UserFacing,
        hir::Constness::NotConst,
    );
    let param_env = traits::normalize_param_env_or_error(tcx, param_env, normalize_cause);

    tcx.infer_ctxt().enter(|ref infcx| {
        let ocx = ObligationCtxt::new(infcx);

        debug!("compare_impl_method: caller_bounds={:?}", param_env.caller_bounds());

        let mut selcx = traits::SelectionContext::new(&infcx);
        let impl_m_own_bounds = impl_m_predicates.instantiate_own(tcx, impl_to_placeholder_substs);
        for (predicate, span) in iter::zip(impl_m_own_bounds.predicates, impl_m_own_bounds.spans) {
            let normalize_cause = traits::ObligationCause::misc(span, impl_m_hir_id);
            let traits::Normalized { value: predicate, obligations } =
                traits::normalize(&mut selcx, param_env, normalize_cause, predicate);

            ocx.register_obligations(obligations);
            let cause = ObligationCause::new(
                span,
                impl_m_hir_id,
                ObligationCauseCode::CompareImplItemObligation {
                    impl_item_def_id: impl_m.def_id.expect_local(),
                    trait_item_def_id: trait_m.def_id,
                    kind: impl_m.kind,
                },
            );
            ocx.register_obligation(traits::Obligation::new(cause, param_env, predicate));
        }

        // We now need to check that the signature of the impl method is
        // compatible with that of the trait method. We do this by
        // checking that `impl_fty <: trait_fty`.
        //
        // FIXME. Unfortunately, this doesn't quite work right now because
        // associated type normalization is not integrated into subtype
        // checks. For the comparison to be valid, we need to
        // normalize the associated types in the impl/trait methods
        // first. However, because function types bind regions, just
        // calling `normalize_associated_types_in` would have no effect on
        // any associated types appearing in the fn arguments or return
        // type.

        // Compute placeholder form of impl and trait method tys.
        let tcx = infcx.tcx;

        let mut wf_tys = FxHashSet::default();

        let impl_sig = infcx.replace_bound_vars_with_fresh_vars(
            impl_m_span,
            infer::HigherRankedType,
            tcx.fn_sig(impl_m.def_id),
        );

        let norm_cause = ObligationCause::misc(impl_m_span, impl_m_hir_id);
        let impl_sig = ocx.normalize(norm_cause.clone(), param_env, impl_sig);
        let impl_fty = tcx.mk_fn_ptr(ty::Binder::dummy(impl_sig));
        debug!("compare_impl_method: impl_fty={:?}", impl_fty);

        let trait_sig = tcx.bound_fn_sig(trait_m.def_id).subst(tcx, trait_to_placeholder_substs);
        let trait_sig = tcx.liberate_late_bound_regions(impl_m.def_id, trait_sig);

        // Next, add all inputs and output as well-formed tys. Importantly,
        // we have to do this before normalization, since the normalized ty may
        // not contain the input parameters. See issue #87748.
        wf_tys.extend(trait_sig.inputs_and_output.iter());
        let trait_sig = ocx.normalize(norm_cause, param_env, trait_sig);
        // We also have to add the normalized trait signature
        // as we don't normalize during implied bounds computation.
        wf_tys.extend(trait_sig.inputs_and_output.iter());
        let trait_fty = tcx.mk_fn_ptr(ty::Binder::dummy(trait_sig));

        debug!("compare_impl_method: trait_fty={:?}", trait_fty);

        // FIXME: We'd want to keep more accurate spans than "the method signature" when
        // processing the comparison between the trait and impl fn, but we sadly lose them
        // and point at the whole signature when a trait bound or specific input or output
        // type would be more appropriate. In other places we have a `Vec<Span>`
        // corresponding to their `Vec<Predicate>`, but we don't have that here.
        // Fixing this would improve the output of test `issue-83765.rs`.
        let mut result = infcx
            .at(&cause, param_env)
            .sup(trait_fty, impl_fty)
            .map(|infer_ok| ocx.register_infer_ok_obligations(infer_ok));

        // HACK(RPITIT): #101614. When we are trying to infer the hidden types for
        // RPITITs, we need to equate the output tys instead of just subtyping. If
        // we just use `sup` above, we'll end up `&'static str <: _#1t`, which causes
        // us to infer `_#1t = #'_#2r str`, where `'_#2r` is unconstrained, which gets
        // fixed up to `ReEmpty`, and which is certainly not what we want.
        if trait_fty.has_infer_types() {
            result = result.and_then(|()| {
                infcx
                    .at(&cause, param_env)
                    .eq(trait_sig.output(), impl_sig.output())
                    .map(|infer_ok| ocx.register_infer_ok_obligations(infer_ok))
            });
        }

        if let Err(terr) = result {
            debug!("sub_types failed: impl ty {:?}, trait ty {:?}", impl_fty, trait_fty);

            let (impl_err_span, trait_err_span) =
                extract_spans_for_error_reporting(&infcx, terr, &cause, impl_m, trait_m);

            cause.span = impl_err_span;

            let mut diag = struct_span_err!(
                tcx.sess,
                cause.span(),
                E0053,
                "method `{}` has an incompatible type for trait",
                trait_m.name
            );
            match &terr {
                TypeError::ArgumentMutability(0) | TypeError::ArgumentSorts(_, 0)
                    if trait_m.fn_has_self_parameter =>
                {
                    let ty = trait_sig.inputs()[0];
                    let sugg = match ExplicitSelf::determine(ty, |_| ty == impl_trait_ref.self_ty())
                    {
                        ExplicitSelf::ByValue => "self".to_owned(),
                        ExplicitSelf::ByReference(_, hir::Mutability::Not) => "&self".to_owned(),
                        ExplicitSelf::ByReference(_, hir::Mutability::Mut) => {
                            "&mut self".to_owned()
                        }
                        _ => format!("self: {ty}"),
                    };

                    // When the `impl` receiver is an arbitrary self type, like `self: Box<Self>`, the
                    // span points only at the type `Box<Self`>, but we want to cover the whole
                    // argument pattern and type.
                    let span = match tcx.hir().expect_impl_item(impl_m.def_id.expect_local()).kind {
                        ImplItemKind::Fn(ref sig, body) => tcx
                            .hir()
                            .body_param_names(body)
                            .zip(sig.decl.inputs.iter())
                            .map(|(param, ty)| param.span.to(ty.span))
                            .next()
                            .unwrap_or(impl_err_span),
                        _ => bug!("{:?} is not a method", impl_m),
                    };

                    diag.span_suggestion(
                        span,
                        "change the self-receiver type to match the trait",
                        sugg,
                        Applicability::MachineApplicable,
                    );
                }
                TypeError::ArgumentMutability(i) | TypeError::ArgumentSorts(_, i) => {
                    if trait_sig.inputs().len() == *i {
                        // Suggestion to change output type. We do not suggest in `async` functions
                        // to avoid complex logic or incorrect output.
                        match tcx.hir().expect_impl_item(impl_m.def_id.expect_local()).kind {
                            ImplItemKind::Fn(ref sig, _)
                                if sig.header.asyncness == hir::IsAsync::NotAsync =>
                            {
                                let msg = "change the output type to match the trait";
                                let ap = Applicability::MachineApplicable;
                                match sig.decl.output {
                                    hir::FnRetTy::DefaultReturn(sp) => {
                                        let sugg = format!("-> {} ", trait_sig.output());
                                        diag.span_suggestion_verbose(sp, msg, sugg, ap);
                                    }
                                    hir::FnRetTy::Return(hir_ty) => {
                                        let sugg = trait_sig.output();
                                        diag.span_suggestion(hir_ty.span, msg, sugg, ap);
                                    }
                                };
                            }
                            _ => {}
                        };
                    } else if let Some(trait_ty) = trait_sig.inputs().get(*i) {
                        diag.span_suggestion(
                            impl_err_span,
                            "change the parameter type to match the trait",
                            trait_ty,
                            Applicability::MachineApplicable,
                        );
                    }
                }
                _ => {}
            }

            infcx.note_type_err(
                &mut diag,
                &cause,
                trait_err_span.map(|sp| (sp, "type in trait".to_owned())),
                Some(infer::ValuePairs::Terms(ExpectedFound {
                    expected: trait_fty.into(),
                    found: impl_fty.into(),
                })),
                terr,
                false,
                false,
            );

            return Err(diag.emit());
        }

        // Check that all obligations are satisfied by the implementation's
        // version.
        let errors = ocx.select_all_or_error();
        if !errors.is_empty() {
            let reported = infcx.report_fulfillment_errors(&errors, None, false);
            return Err(reported);
        }

        // Finally, resolve all regions. This catches wily misuses of
        // lifetime parameters.
        let outlives_environment = OutlivesEnvironment::with_bounds(
            param_env,
            Some(infcx),
            infcx.implied_bounds_tys(param_env, impl_m_hir_id, wf_tys),
        );
        infcx.check_region_obligations_and_report_errors(
            impl_m.def_id.expect_local(),
            &outlives_environment,
        );

        Ok(())
    })
}

pub fn collect_trait_impl_trait_tys<'tcx>(
    tcx: TyCtxt<'tcx>,
    def_id: DefId,
) -> Result<&'tcx FxHashMap<DefId, Ty<'tcx>>, ErrorGuaranteed> {
    let impl_m = tcx.opt_associated_item(def_id).unwrap();
    let trait_m = tcx.opt_associated_item(impl_m.trait_item_def_id.unwrap()).unwrap();
    let impl_trait_ref = tcx.impl_trait_ref(impl_m.impl_container(tcx).unwrap()).unwrap();
    let param_env = tcx.param_env(def_id);

    let trait_to_impl_substs = impl_trait_ref.substs;

    let impl_m_hir_id = tcx.hir().local_def_id_to_hir_id(impl_m.def_id.expect_local());
    let return_span = tcx.hir().fn_decl_by_hir_id(impl_m_hir_id).unwrap().output.span();
    let cause = ObligationCause::new(
        return_span,
        impl_m_hir_id,
        ObligationCauseCode::CompareImplItemObligation {
            impl_item_def_id: impl_m.def_id.expect_local(),
            trait_item_def_id: trait_m.def_id,
            kind: impl_m.kind,
        },
    );

    // Create mapping from impl to placeholder.
    let impl_to_placeholder_substs = InternalSubsts::identity_for_item(tcx, impl_m.def_id);

    // Create mapping from trait to placeholder.
    let trait_to_placeholder_substs =
        impl_to_placeholder_substs.rebase_onto(tcx, impl_m.container_id(tcx), trait_to_impl_substs);

    tcx.infer_ctxt().enter(|ref infcx| {
        let ocx = ObligationCtxt::new(infcx);

        let norm_cause = ObligationCause::misc(return_span, impl_m_hir_id);
        let impl_return_ty = ocx.normalize(
            norm_cause.clone(),
            param_env,
            infcx
                .replace_bound_vars_with_fresh_vars(
                    return_span,
                    infer::HigherRankedType,
                    tcx.fn_sig(impl_m.def_id),
                )
                .output(),
        );

        let mut collector =
            ImplTraitInTraitCollector::new(&ocx, return_span, param_env, impl_m_hir_id);
        let unnormalized_trait_return_ty = tcx
            .liberate_late_bound_regions(
                impl_m.def_id,
                tcx.bound_fn_sig(trait_m.def_id).subst(tcx, trait_to_placeholder_substs),
            )
            .output()
            .fold_with(&mut collector);
        let trait_return_ty =
            ocx.normalize(norm_cause.clone(), param_env, unnormalized_trait_return_ty);

        let wf_tys = FxHashSet::from_iter([unnormalized_trait_return_ty, trait_return_ty]);

        match infcx.at(&cause, param_env).eq(trait_return_ty, impl_return_ty) {
            Ok(infer::InferOk { value: (), obligations }) => {
                ocx.register_obligations(obligations);
            }
            Err(terr) => {
                let mut diag = struct_span_err!(
                    tcx.sess,
                    cause.span(),
                    E0053,
                    "method `{}` has an incompatible return type for trait",
                    trait_m.name
                );
                let hir = tcx.hir();
                infcx.note_type_err(
                    &mut diag,
                    &cause,
                    hir.get_if_local(impl_m.def_id)
                        .and_then(|node| node.fn_decl())
                        .map(|decl| (decl.output.span(), "return type in trait".to_owned())),
                    Some(infer::ValuePairs::Terms(ExpectedFound {
                        expected: trait_return_ty.into(),
                        found: impl_return_ty.into(),
                    })),
                    terr,
                    false,
                    false,
                );
                return Err(diag.emit());
            }
        }

        // Check that all obligations are satisfied by the implementation's
        // RPITs.
        let errors = ocx.select_all_or_error();
        if !errors.is_empty() {
            let reported = infcx.report_fulfillment_errors(&errors, None, false);
            return Err(reported);
        }

        // Finally, resolve all regions. This catches wily misuses of
        // lifetime parameters.
        let outlives_environment = OutlivesEnvironment::with_bounds(
            param_env,
            Some(infcx),
            infcx.implied_bounds_tys(param_env, impl_m_hir_id, wf_tys),
        );
        infcx.check_region_obligations_and_report_errors(
            impl_m.def_id.expect_local(),
            &outlives_environment,
        );

        let mut collected_tys = FxHashMap::default();
        for (def_id, (ty, substs)) in collector.types {
            match infcx.fully_resolve(ty) {
                Ok(ty) => {
                    // `ty` contains free regions that we created earlier while liberating the
                    // trait fn signature.  However, projection normalization expects `ty` to
                    // contains `def_id`'s early-bound regions.
                    let id_substs = InternalSubsts::identity_for_item(tcx, def_id);
                    debug!(?id_substs, ?substs);
                    let map: FxHashMap<ty::GenericArg<'tcx>, ty::GenericArg<'tcx>> = substs
                        .iter()
                        .enumerate()
                        .map(|(index, arg)| (arg, id_substs[index]))
                        .collect();
                    debug!(?map);

                    let ty = tcx.fold_regions(ty, |region, _| {
                        if let ty::ReFree(_) = region.kind() {
                            map[&region.into()].expect_region()
                        } else {
                            region
                        }
                    });
                    debug!(%ty);
                    collected_tys.insert(def_id, ty);
                }
                Err(err) => {
                    tcx.sess.delay_span_bug(
                        return_span,
                        format!("could not fully resolve: {ty} => {err:?}"),
                    );
                    collected_tys.insert(def_id, tcx.ty_error());
                }
            }
        }

        Ok(&*tcx.arena.alloc(collected_tys))
    })
}

struct ImplTraitInTraitCollector<'a, 'tcx> {
    ocx: &'a ObligationCtxt<'a, 'tcx>,
    types: FxHashMap<DefId, (Ty<'tcx>, ty::SubstsRef<'tcx>)>,
    span: Span,
    param_env: ty::ParamEnv<'tcx>,
    body_id: hir::HirId,
}

impl<'a, 'tcx> ImplTraitInTraitCollector<'a, 'tcx> {
    fn new(
        ocx: &'a ObligationCtxt<'a, 'tcx>,
        span: Span,
        param_env: ty::ParamEnv<'tcx>,
        body_id: hir::HirId,
    ) -> Self {
        ImplTraitInTraitCollector { ocx, types: FxHashMap::default(), span, param_env, body_id }
    }
}

impl<'tcx> TypeFolder<'tcx> for ImplTraitInTraitCollector<'_, 'tcx> {
    fn tcx<'a>(&'a self) -> TyCtxt<'tcx> {
        self.ocx.infcx.tcx
    }

    fn fold_ty(&mut self, ty: Ty<'tcx>) -> Ty<'tcx> {
        if let ty::Projection(proj) = ty.kind()
            && self.tcx().def_kind(proj.item_def_id) == DefKind::ImplTraitPlaceholder
        {
            if let Some((ty, _)) = self.types.get(&proj.item_def_id) {
                return *ty;
            }
            //FIXME(RPITIT): Deny nested RPITIT in substs too
            if proj.substs.has_escaping_bound_vars() {
                bug!("FIXME(RPITIT): error here");
            }
            // Replace with infer var
            let infer_ty = self.ocx.infcx.next_ty_var(TypeVariableOrigin {
                span: self.span,
                kind: TypeVariableOriginKind::MiscVariable,
            });
            self.types.insert(proj.item_def_id, (infer_ty, proj.substs));
            // Recurse into bounds
            for pred in self.tcx().bound_explicit_item_bounds(proj.item_def_id).transpose_iter() {
                let pred_span = pred.0.1;

                let pred = pred.map_bound(|(pred, _)| *pred).subst(self.tcx(), proj.substs);
                let pred = pred.fold_with(self);
                let pred = self.ocx.normalize(
                    ObligationCause::misc(self.span, self.body_id),
                    self.param_env,
                    pred,
                );

                self.ocx.register_obligation(traits::Obligation::new(
                    ObligationCause::new(
                        self.span,
                        self.body_id,
                        ObligationCauseCode::BindingObligation(proj.item_def_id, pred_span),
                    ),
                    self.param_env,
                    pred,
                ));
            }
            infer_ty
        } else {
            ty.super_fold_with(self)
        }
    }
}

fn check_region_bounds_on_impl_item<'tcx>(
    tcx: TyCtxt<'tcx>,
    impl_m: &ty::AssocItem,
    trait_m: &ty::AssocItem,
    trait_generics: &ty::Generics,
    impl_generics: &ty::Generics,
) -> Result<(), ErrorGuaranteed> {
    let trait_params = trait_generics.own_counts().lifetimes;
    let impl_params = impl_generics.own_counts().lifetimes;

    debug!(
        "check_region_bounds_on_impl_item: \
            trait_generics={:?} \
            impl_generics={:?}",
        trait_generics, impl_generics
    );

    // Must have same number of early-bound lifetime parameters.
    // Unfortunately, if the user screws up the bounds, then this
    // will change classification between early and late.  E.g.,
    // if in trait we have `<'a,'b:'a>`, and in impl we just have
    // `<'a,'b>`, then we have 2 early-bound lifetime parameters
    // in trait but 0 in the impl. But if we report "expected 2
    // but found 0" it's confusing, because it looks like there
    // are zero. Since I don't quite know how to phrase things at
    // the moment, give a kind of vague error message.
    if trait_params != impl_params {
        let span = tcx
            .hir()
            .get_generics(impl_m.def_id.expect_local())
            .expect("expected impl item to have generics or else we can't compare them")
            .span;
        let generics_span = if let Some(local_def_id) = trait_m.def_id.as_local() {
            Some(
                tcx.hir()
                    .get_generics(local_def_id)
                    .expect("expected trait item to have generics or else we can't compare them")
                    .span,
            )
        } else {
            None
        };

        let reported = tcx.sess.emit_err(LifetimesOrBoundsMismatchOnTrait {
            span,
            item_kind: assoc_item_kind_str(impl_m),
            ident: impl_m.ident(tcx),
            generics_span,
        });
        return Err(reported);
    }

    Ok(())
}

#[instrument(level = "debug", skip(infcx))]
fn extract_spans_for_error_reporting<'a, 'tcx>(
    infcx: &infer::InferCtxt<'a, 'tcx>,
    terr: TypeError<'_>,
    cause: &ObligationCause<'tcx>,
    impl_m: &ty::AssocItem,
    trait_m: &ty::AssocItem,
) -> (Span, Option<Span>) {
    let tcx = infcx.tcx;
    let mut impl_args = match tcx.hir().expect_impl_item(impl_m.def_id.expect_local()).kind {
        ImplItemKind::Fn(ref sig, _) => {
            sig.decl.inputs.iter().map(|t| t.span).chain(iter::once(sig.decl.output.span()))
        }
        _ => bug!("{:?} is not a method", impl_m),
    };
    let trait_args =
        trait_m.def_id.as_local().map(|def_id| match tcx.hir().expect_trait_item(def_id).kind {
            TraitItemKind::Fn(ref sig, _) => {
                sig.decl.inputs.iter().map(|t| t.span).chain(iter::once(sig.decl.output.span()))
            }
            _ => bug!("{:?} is not a TraitItemKind::Fn", trait_m),
        });

    match terr {
        TypeError::ArgumentMutability(i) => {
            (impl_args.nth(i).unwrap(), trait_args.and_then(|mut args| args.nth(i)))
        }
        TypeError::ArgumentSorts(ExpectedFound { .. }, i) => {
            (impl_args.nth(i).unwrap(), trait_args.and_then(|mut args| args.nth(i)))
        }
        _ => (cause.span(), tcx.hir().span_if_local(trait_m.def_id)),
    }
}

fn compare_self_type<'tcx>(
    tcx: TyCtxt<'tcx>,
    impl_m: &ty::AssocItem,
    impl_m_span: Span,
    trait_m: &ty::AssocItem,
    impl_trait_ref: ty::TraitRef<'tcx>,
) -> Result<(), ErrorGuaranteed> {
    // Try to give more informative error messages about self typing
    // mismatches.  Note that any mismatch will also be detected
    // below, where we construct a canonical function type that
    // includes the self parameter as a normal parameter.  It's just
    // that the error messages you get out of this code are a bit more
    // inscrutable, particularly for cases where one method has no
    // self.

    let self_string = |method: &ty::AssocItem| {
        let untransformed_self_ty = match method.container {
            ty::ImplContainer => impl_trait_ref.self_ty(),
            ty::TraitContainer => tcx.types.self_param,
        };
        let self_arg_ty = tcx.fn_sig(method.def_id).input(0);
        let param_env = ty::ParamEnv::reveal_all();

        tcx.infer_ctxt().enter(|infcx| {
            let self_arg_ty = tcx.liberate_late_bound_regions(method.def_id, self_arg_ty);
            let can_eq_self = |ty| infcx.can_eq(param_env, untransformed_self_ty, ty).is_ok();
            match ExplicitSelf::determine(self_arg_ty, can_eq_self) {
                ExplicitSelf::ByValue => "self".to_owned(),
                ExplicitSelf::ByReference(_, hir::Mutability::Not) => "&self".to_owned(),
                ExplicitSelf::ByReference(_, hir::Mutability::Mut) => "&mut self".to_owned(),
                _ => format!("self: {self_arg_ty}"),
            }
        })
    };

    match (trait_m.fn_has_self_parameter, impl_m.fn_has_self_parameter) {
        (false, false) | (true, true) => {}

        (false, true) => {
            let self_descr = self_string(impl_m);
            let mut err = struct_span_err!(
                tcx.sess,
                impl_m_span,
                E0185,
                "method `{}` has a `{}` declaration in the impl, but not in the trait",
                trait_m.name,
                self_descr
            );
            err.span_label(impl_m_span, format!("`{self_descr}` used in impl"));
            if let Some(span) = tcx.hir().span_if_local(trait_m.def_id) {
                err.span_label(span, format!("trait method declared without `{self_descr}`"));
            } else {
                err.note_trait_signature(trait_m.name, trait_m.signature(tcx));
            }
            let reported = err.emit();
            return Err(reported);
        }

        (true, false) => {
            let self_descr = self_string(trait_m);
            let mut err = struct_span_err!(
                tcx.sess,
                impl_m_span,
                E0186,
                "method `{}` has a `{}` declaration in the trait, but not in the impl",
                trait_m.name,
                self_descr
            );
            err.span_label(impl_m_span, format!("expected `{self_descr}` in impl"));
            if let Some(span) = tcx.hir().span_if_local(trait_m.def_id) {
                err.span_label(span, format!("`{self_descr}` used in trait"));
            } else {
                err.note_trait_signature(trait_m.name, trait_m.signature(tcx));
            }
            let reported = err.emit();
            return Err(reported);
        }
    }

    Ok(())
}

/// Checks that the number of generics on a given assoc item in a trait impl is the same
/// as the number of generics on the respective assoc item in the trait definition.
///
/// For example this code emits the errors in the following code:
/// ```
/// trait Trait {
///     fn foo();
///     type Assoc<T>;
/// }
///
/// impl Trait for () {
///     fn foo<T>() {}
///     //~^ error
///     type Assoc = u32;
///     //~^ error
/// }
/// ```
///
/// Notably this does not error on `foo<T>` implemented as `foo<const N: u8>` or
/// `foo<const N: u8>` implemented as `foo<const N: u32>`. This is handled in
/// [`compare_generic_param_kinds`]. This function also does not handle lifetime parameters
fn compare_number_of_generics<'tcx>(
    tcx: TyCtxt<'tcx>,
    impl_: &ty::AssocItem,
    _impl_span: Span,
    trait_: &ty::AssocItem,
    trait_span: Option<Span>,
) -> Result<(), ErrorGuaranteed> {
    let trait_own_counts = tcx.generics_of(trait_.def_id).own_counts();
    let impl_own_counts = tcx.generics_of(impl_.def_id).own_counts();

    // This avoids us erroring on `foo<T>` implemented as `foo<const N: u8>` as this is implemented
    // in `compare_generic_param_kinds` which will give a nicer error message than something like:
    // "expected 1 type parameter, found 0 type parameters"
    if (trait_own_counts.types + trait_own_counts.consts)
        == (impl_own_counts.types + impl_own_counts.consts)
    {
        return Ok(());
    }

    let matchings = [
        ("type", trait_own_counts.types, impl_own_counts.types),
        ("const", trait_own_counts.consts, impl_own_counts.consts),
    ];

    let item_kind = assoc_item_kind_str(impl_);

    let mut err_occurred = None;
    for (kind, trait_count, impl_count) in matchings {
        if impl_count != trait_count {
            let arg_spans = |kind: ty::AssocKind, generics: &hir::Generics<'_>| {
                let mut spans = generics
                    .params
                    .iter()
                    .filter(|p| match p.kind {
                        hir::GenericParamKind::Lifetime {
                            kind: hir::LifetimeParamKind::Elided,
                        } => {
                            // A fn can have an arbitrary number of extra elided lifetimes for the
                            // same signature.
                            !matches!(kind, ty::AssocKind::Fn)
                        }
                        _ => true,
                    })
                    .map(|p| p.span)
                    .collect::<Vec<Span>>();
                if spans.is_empty() {
                    spans = vec![generics.span]
                }
                spans
            };
            let (trait_spans, impl_trait_spans) = if let Some(def_id) = trait_.def_id.as_local() {
                let trait_item = tcx.hir().expect_trait_item(def_id);
                let arg_spans: Vec<Span> = arg_spans(trait_.kind, trait_item.generics);
                let impl_trait_spans: Vec<Span> = trait_item
                    .generics
                    .params
                    .iter()
                    .filter_map(|p| match p.kind {
                        GenericParamKind::Type { synthetic: true, .. } => Some(p.span),
                        _ => None,
                    })
                    .collect();
                (Some(arg_spans), impl_trait_spans)
            } else {
                (trait_span.map(|s| vec![s]), vec![])
            };

            let impl_item = tcx.hir().expect_impl_item(impl_.def_id.expect_local());
            let impl_item_impl_trait_spans: Vec<Span> = impl_item
                .generics
                .params
                .iter()
                .filter_map(|p| match p.kind {
                    GenericParamKind::Type { synthetic: true, .. } => Some(p.span),
                    _ => None,
                })
                .collect();
            let spans = arg_spans(impl_.kind, impl_item.generics);
            let span = spans.first().copied();

            let mut err = tcx.sess.struct_span_err_with_code(
                spans,
                &format!(
                    "{} `{}` has {} {kind} parameter{} but its trait \
                     declaration has {} {kind} parameter{}",
                    item_kind,
                    trait_.name,
                    impl_count,
                    pluralize!(impl_count),
                    trait_count,
                    pluralize!(trait_count),
                    kind = kind,
                ),
                DiagnosticId::Error("E0049".into()),
            );

            let mut suffix = None;

            if let Some(spans) = trait_spans {
                let mut spans = spans.iter();
                if let Some(span) = spans.next() {
                    err.span_label(
                        *span,
                        format!(
                            "expected {} {} parameter{}",
                            trait_count,
                            kind,
                            pluralize!(trait_count),
                        ),
                    );
                }
                for span in spans {
                    err.span_label(*span, "");
                }
            } else {
                suffix = Some(format!(", expected {trait_count}"));
            }

            if let Some(span) = span {
                err.span_label(
                    span,
                    format!(
                        "found {} {} parameter{}{}",
                        impl_count,
                        kind,
                        pluralize!(impl_count),
                        suffix.unwrap_or_else(String::new),
                    ),
                );
            }

            for span in impl_trait_spans.iter().chain(impl_item_impl_trait_spans.iter()) {
                err.span_label(*span, "`impl Trait` introduces an implicit type parameter");
            }

            let reported = err.emit();
            err_occurred = Some(reported);
        }
    }

    if let Some(reported) = err_occurred { Err(reported) } else { Ok(()) }
}

fn compare_number_of_method_arguments<'tcx>(
    tcx: TyCtxt<'tcx>,
    impl_m: &ty::AssocItem,
    impl_m_span: Span,
    trait_m: &ty::AssocItem,
    trait_item_span: Option<Span>,
) -> Result<(), ErrorGuaranteed> {
    let impl_m_fty = tcx.fn_sig(impl_m.def_id);
    let trait_m_fty = tcx.fn_sig(trait_m.def_id);
    let trait_number_args = trait_m_fty.inputs().skip_binder().len();
    let impl_number_args = impl_m_fty.inputs().skip_binder().len();
    if trait_number_args != impl_number_args {
        let trait_span = if let Some(def_id) = trait_m.def_id.as_local() {
            match tcx.hir().expect_trait_item(def_id).kind {
                TraitItemKind::Fn(ref trait_m_sig, _) => {
                    let pos = if trait_number_args > 0 { trait_number_args - 1 } else { 0 };
                    if let Some(arg) = trait_m_sig.decl.inputs.get(pos) {
                        Some(if pos == 0 {
                            arg.span
                        } else {
                            arg.span.with_lo(trait_m_sig.decl.inputs[0].span.lo())
                        })
                    } else {
                        trait_item_span
                    }
                }
                _ => bug!("{:?} is not a method", impl_m),
            }
        } else {
            trait_item_span
        };
        let impl_span = match tcx.hir().expect_impl_item(impl_m.def_id.expect_local()).kind {
            ImplItemKind::Fn(ref impl_m_sig, _) => {
                let pos = if impl_number_args > 0 { impl_number_args - 1 } else { 0 };
                if let Some(arg) = impl_m_sig.decl.inputs.get(pos) {
                    if pos == 0 {
                        arg.span
                    } else {
                        arg.span.with_lo(impl_m_sig.decl.inputs[0].span.lo())
                    }
                } else {
                    impl_m_span
                }
            }
            _ => bug!("{:?} is not a method", impl_m),
        };
        let mut err = struct_span_err!(
            tcx.sess,
            impl_span,
            E0050,
            "method `{}` has {} but the declaration in trait `{}` has {}",
            trait_m.name,
            potentially_plural_count(impl_number_args, "parameter"),
            tcx.def_path_str(trait_m.def_id),
            trait_number_args
        );
        if let Some(trait_span) = trait_span {
            err.span_label(
                trait_span,
                format!(
                    "trait requires {}",
                    potentially_plural_count(trait_number_args, "parameter")
                ),
            );
        } else {
            err.note_trait_signature(trait_m.name, trait_m.signature(tcx));
        }
        err.span_label(
            impl_span,
            format!(
                "expected {}, found {}",
                potentially_plural_count(trait_number_args, "parameter"),
                impl_number_args
            ),
        );
        let reported = err.emit();
        return Err(reported);
    }

    Ok(())
}

fn compare_synthetic_generics<'tcx>(
    tcx: TyCtxt<'tcx>,
    impl_m: &ty::AssocItem,
    trait_m: &ty::AssocItem,
) -> Result<(), ErrorGuaranteed> {
    // FIXME(chrisvittal) Clean up this function, list of FIXME items:
    //     1. Better messages for the span labels
    //     2. Explanation as to what is going on
    // If we get here, we already have the same number of generics, so the zip will
    // be okay.
    let mut error_found = None;
    let impl_m_generics = tcx.generics_of(impl_m.def_id);
    let trait_m_generics = tcx.generics_of(trait_m.def_id);
    let impl_m_type_params = impl_m_generics.params.iter().filter_map(|param| match param.kind {
        GenericParamDefKind::Type { synthetic, .. } => Some((param.def_id, synthetic)),
        GenericParamDefKind::Lifetime | GenericParamDefKind::Const { .. } => None,
    });
    let trait_m_type_params = trait_m_generics.params.iter().filter_map(|param| match param.kind {
        GenericParamDefKind::Type { synthetic, .. } => Some((param.def_id, synthetic)),
        GenericParamDefKind::Lifetime | GenericParamDefKind::Const { .. } => None,
    });
    for ((impl_def_id, impl_synthetic), (trait_def_id, trait_synthetic)) in
        iter::zip(impl_m_type_params, trait_m_type_params)
    {
        if impl_synthetic != trait_synthetic {
            let impl_def_id = impl_def_id.expect_local();
            let impl_span = tcx.def_span(impl_def_id);
            let trait_span = tcx.def_span(trait_def_id);
            let mut err = struct_span_err!(
                tcx.sess,
                impl_span,
                E0643,
                "method `{}` has incompatible signature for trait",
                trait_m.name
            );
            err.span_label(trait_span, "declaration in trait here");
            match (impl_synthetic, trait_synthetic) {
                // The case where the impl method uses `impl Trait` but the trait method uses
                // explicit generics
                (true, false) => {
                    err.span_label(impl_span, "expected generic parameter, found `impl Trait`");
                    (|| {
                        // try taking the name from the trait impl
                        // FIXME: this is obviously suboptimal since the name can already be used
                        // as another generic argument
                        let new_name = tcx.opt_item_name(trait_def_id)?;
                        let trait_m = trait_m.def_id.as_local()?;
                        let trait_m = tcx.hir().expect_trait_item(trait_m);

                        let impl_m = impl_m.def_id.as_local()?;
                        let impl_m = tcx.hir().expect_impl_item(impl_m);

                        // in case there are no generics, take the spot between the function name
                        // and the opening paren of the argument list
                        let new_generics_span = tcx.def_ident_span(impl_def_id)?.shrink_to_hi();
                        // in case there are generics, just replace them
                        let generics_span =
                            impl_m.generics.span.substitute_dummy(new_generics_span);
                        // replace with the generics from the trait
                        let new_generics =
                            tcx.sess.source_map().span_to_snippet(trait_m.generics.span).ok()?;

                        err.multipart_suggestion(
                            "try changing the `impl Trait` argument to a generic parameter",
                            vec![
                                // replace `impl Trait` with `T`
                                (impl_span, new_name.to_string()),
                                // replace impl method generics with trait method generics
                                // This isn't quite right, as users might have changed the names
                                // of the generics, but it works for the common case
                                (generics_span, new_generics),
                            ],
                            Applicability::MaybeIncorrect,
                        );
                        Some(())
                    })();
                }
                // The case where the trait method uses `impl Trait`, but the impl method uses
                // explicit generics.
                (false, true) => {
                    err.span_label(impl_span, "expected `impl Trait`, found generic parameter");
                    (|| {
                        let impl_m = impl_m.def_id.as_local()?;
                        let impl_m = tcx.hir().expect_impl_item(impl_m);
                        let input_tys = match impl_m.kind {
                            hir::ImplItemKind::Fn(ref sig, _) => sig.decl.inputs,
                            _ => unreachable!(),
                        };
                        struct Visitor(Option<Span>, hir::def_id::LocalDefId);
                        impl<'v> intravisit::Visitor<'v> for Visitor {
                            fn visit_ty(&mut self, ty: &'v hir::Ty<'v>) {
                                intravisit::walk_ty(self, ty);
                                if let hir::TyKind::Path(hir::QPath::Resolved(None, ref path)) =
                                    ty.kind
                                    && let Res::Def(DefKind::TyParam, def_id) = path.res
                                    && def_id == self.1.to_def_id()
                                {
                                    self.0 = Some(ty.span);
                                }
                            }
                        }
                        let mut visitor = Visitor(None, impl_def_id);
                        for ty in input_tys {
                            intravisit::Visitor::visit_ty(&mut visitor, ty);
                        }
                        let span = visitor.0?;

                        let bounds = impl_m.generics.bounds_for_param(impl_def_id).next()?.bounds;
                        let bounds = bounds.first()?.span().to(bounds.last()?.span());
                        let bounds = tcx.sess.source_map().span_to_snippet(bounds).ok()?;

                        err.multipart_suggestion(
                            "try removing the generic parameter and using `impl Trait` instead",
                            vec![
                                // delete generic parameters
                                (impl_m.generics.span, String::new()),
                                // replace param usage with `impl Trait`
                                (span, format!("impl {bounds}")),
                            ],
                            Applicability::MaybeIncorrect,
                        );
                        Some(())
                    })();
                }
                _ => unreachable!(),
            }
            let reported = err.emit();
            error_found = Some(reported);
        }
    }
    if let Some(reported) = error_found { Err(reported) } else { Ok(()) }
}

/// Checks that all parameters in the generics of a given assoc item in a trait impl have
/// the same kind as the respective generic parameter in the trait def.
///
/// For example all 4 errors in the following code are emitted here:
/// ```
/// trait Foo {
///     fn foo<const N: u8>();
///     type bar<const N: u8>;
///     fn baz<const N: u32>();
///     type blah<T>;
/// }
///
/// impl Foo for () {
///     fn foo<const N: u64>() {}
///     //~^ error
///     type bar<const N: u64> {}
///     //~^ error
///     fn baz<T>() {}
///     //~^ error
///     type blah<const N: i64> = u32;
///     //~^ error
/// }
/// ```
///
/// This function does not handle lifetime parameters
fn compare_generic_param_kinds<'tcx>(
    tcx: TyCtxt<'tcx>,
    impl_item: &ty::AssocItem,
    trait_item: &ty::AssocItem,
) -> Result<(), ErrorGuaranteed> {
    assert_eq!(impl_item.kind, trait_item.kind);

    let ty_const_params_of = |def_id| {
        tcx.generics_of(def_id).params.iter().filter(|param| {
            matches!(
                param.kind,
                GenericParamDefKind::Const { .. } | GenericParamDefKind::Type { .. }
            )
        })
    };

    for (param_impl, param_trait) in
        iter::zip(ty_const_params_of(impl_item.def_id), ty_const_params_of(trait_item.def_id))
    {
        use GenericParamDefKind::*;
        if match (&param_impl.kind, &param_trait.kind) {
            (Const { .. }, Const { .. })
                if tcx.type_of(param_impl.def_id) != tcx.type_of(param_trait.def_id) =>
            {
                true
            }
            (Const { .. }, Type { .. }) | (Type { .. }, Const { .. }) => true,
            // this is exhaustive so that anyone adding new generic param kinds knows
            // to make sure this error is reported for them.
            (Const { .. }, Const { .. }) | (Type { .. }, Type { .. }) => false,
            (Lifetime { .. }, _) | (_, Lifetime { .. }) => unreachable!(),
        } {
            let param_impl_span = tcx.def_span(param_impl.def_id);
            let param_trait_span = tcx.def_span(param_trait.def_id);

            let mut err = struct_span_err!(
                tcx.sess,
                param_impl_span,
                E0053,
                "{} `{}` has an incompatible generic parameter for trait `{}`",
                assoc_item_kind_str(&impl_item),
                trait_item.name,
                &tcx.def_path_str(tcx.parent(trait_item.def_id))
            );

            let make_param_message = |prefix: &str, param: &ty::GenericParamDef| match param.kind {
                Const { .. } => {
                    format!("{} const parameter of type `{}`", prefix, tcx.type_of(param.def_id))
                }
                Type { .. } => format!("{} type parameter", prefix),
                Lifetime { .. } => unreachable!(),
            };

            let trait_header_span = tcx.def_ident_span(tcx.parent(trait_item.def_id)).unwrap();
            err.span_label(trait_header_span, "");
            err.span_label(param_trait_span, make_param_message("expected", param_trait));

            let impl_header_span = tcx.def_span(tcx.parent(impl_item.def_id));
            err.span_label(impl_header_span, "");
            err.span_label(param_impl_span, make_param_message("found", param_impl));

            let reported = err.emit();
            return Err(reported);
        }
    }

    Ok(())
}

pub(crate) fn compare_const_impl<'tcx>(
    tcx: TyCtxt<'tcx>,
    impl_c: &ty::AssocItem,
    impl_c_span: Span,
    trait_c: &ty::AssocItem,
    impl_trait_ref: ty::TraitRef<'tcx>,
) {
    debug!("compare_const_impl(impl_trait_ref={:?})", impl_trait_ref);

    tcx.infer_ctxt().enter(|infcx| {
        let param_env = tcx.param_env(impl_c.def_id);
        let ocx = ObligationCtxt::new(&infcx);

        // The below is for the most part highly similar to the procedure
        // for methods above. It is simpler in many respects, especially
        // because we shouldn't really have to deal with lifetimes or
        // predicates. In fact some of this should probably be put into
        // shared functions because of DRY violations...
        let trait_to_impl_substs = impl_trait_ref.substs;

        // Create a parameter environment that represents the implementation's
        // method.
        let impl_c_hir_id = tcx.hir().local_def_id_to_hir_id(impl_c.def_id.expect_local());

        // Compute placeholder form of impl and trait const tys.
        let impl_ty = tcx.type_of(impl_c.def_id);
        let trait_ty = tcx.bound_type_of(trait_c.def_id).subst(tcx, trait_to_impl_substs);
        let mut cause = ObligationCause::new(
            impl_c_span,
            impl_c_hir_id,
            ObligationCauseCode::CompareImplItemObligation {
                impl_item_def_id: impl_c.def_id.expect_local(),
                trait_item_def_id: trait_c.def_id,
                kind: impl_c.kind,
            },
        );

        // There is no "body" here, so just pass dummy id.
        let impl_ty = ocx.normalize(cause.clone(), param_env, impl_ty);

        debug!("compare_const_impl: impl_ty={:?}", impl_ty);

        let trait_ty = ocx.normalize(cause.clone(), param_env, trait_ty);

        debug!("compare_const_impl: trait_ty={:?}", trait_ty);

        let err = infcx
            .at(&cause, param_env)
            .sup(trait_ty, impl_ty)
            .map(|ok| ocx.register_infer_ok_obligations(ok));

        if let Err(terr) = err {
            debug!(
                "checking associated const for compatibility: impl ty {:?}, trait ty {:?}",
                impl_ty, trait_ty
            );

            // Locate the Span containing just the type of the offending impl
            match tcx.hir().expect_impl_item(impl_c.def_id.expect_local()).kind {
                ImplItemKind::Const(ref ty, _) => cause.span = ty.span,
                _ => bug!("{:?} is not a impl const", impl_c),
            }

            let mut diag = struct_span_err!(
                tcx.sess,
                cause.span,
                E0326,
                "implemented const `{}` has an incompatible type for trait",
                trait_c.name
            );

            let trait_c_span = trait_c.def_id.as_local().map(|trait_c_def_id| {
                // Add a label to the Span containing just the type of the const
                match tcx.hir().expect_trait_item(trait_c_def_id).kind {
                    TraitItemKind::Const(ref ty, _) => ty.span,
                    _ => bug!("{:?} is not a trait const", trait_c),
                }
            });

            infcx.note_type_err(
                &mut diag,
                &cause,
                trait_c_span.map(|span| (span, "type in trait".to_owned())),
                Some(infer::ValuePairs::Terms(ExpectedFound {
                    expected: trait_ty.into(),
                    found: impl_ty.into(),
                })),
                terr,
                false,
                false,
            );
            diag.emit();
        }

        // Check that all obligations are satisfied by the implementation's
        // version.
        let errors = ocx.select_all_or_error();
        if !errors.is_empty() {
            infcx.report_fulfillment_errors(&errors, None, false);
            return;
        }

        let outlives_environment = OutlivesEnvironment::new(param_env);
        infcx.check_region_obligations_and_report_errors(
            impl_c.def_id.expect_local(),
            &outlives_environment,
        );
    });
}

pub(crate) fn compare_ty_impl<'tcx>(
    tcx: TyCtxt<'tcx>,
    impl_ty: &ty::AssocItem,
    impl_ty_span: Span,
    trait_ty: &ty::AssocItem,
    impl_trait_ref: ty::TraitRef<'tcx>,
    trait_item_span: Option<Span>,
) {
    debug!("compare_impl_type(impl_trait_ref={:?})", impl_trait_ref);

    let _: Result<(), ErrorGuaranteed> = (|| {
        compare_number_of_generics(tcx, impl_ty, impl_ty_span, trait_ty, trait_item_span)?;

        compare_generic_param_kinds(tcx, impl_ty, trait_ty)?;

        let sp = tcx.def_span(impl_ty.def_id);
        compare_type_predicate_entailment(tcx, impl_ty, sp, trait_ty, impl_trait_ref)?;

        check_type_bounds(tcx, trait_ty, impl_ty, impl_ty_span, impl_trait_ref)
    })();
}

/// The equivalent of [compare_predicate_entailment], but for associated types
/// instead of associated functions.
fn compare_type_predicate_entailment<'tcx>(
    tcx: TyCtxt<'tcx>,
    impl_ty: &ty::AssocItem,
    impl_ty_span: Span,
    trait_ty: &ty::AssocItem,
    impl_trait_ref: ty::TraitRef<'tcx>,
) -> Result<(), ErrorGuaranteed> {
    let impl_substs = InternalSubsts::identity_for_item(tcx, impl_ty.def_id);
    let trait_to_impl_substs =
        impl_substs.rebase_onto(tcx, impl_ty.container_id(tcx), impl_trait_ref.substs);

    let impl_ty_generics = tcx.generics_of(impl_ty.def_id);
    let trait_ty_generics = tcx.generics_of(trait_ty.def_id);
    let impl_ty_predicates = tcx.predicates_of(impl_ty.def_id);
    let trait_ty_predicates = tcx.predicates_of(trait_ty.def_id);

    check_region_bounds_on_impl_item(
        tcx,
        impl_ty,
        trait_ty,
        &trait_ty_generics,
        &impl_ty_generics,
    )?;

    let impl_ty_own_bounds = impl_ty_predicates.instantiate_own(tcx, impl_substs);

    if impl_ty_own_bounds.is_empty() {
        // Nothing to check.
        return Ok(());
    }

    // This `HirId` should be used for the `body_id` field on each
    // `ObligationCause` (and the `FnCtxt`). This is what
    // `regionck_item` expects.
    let impl_ty_hir_id = tcx.hir().local_def_id_to_hir_id(impl_ty.def_id.expect_local());
    debug!("compare_type_predicate_entailment: trait_to_impl_substs={:?}", trait_to_impl_substs);

    // The predicates declared by the impl definition, the trait and the
    // associated type in the trait are assumed.
    let impl_predicates = tcx.predicates_of(impl_ty_predicates.parent.unwrap());
    let mut hybrid_preds = impl_predicates.instantiate_identity(tcx);
    hybrid_preds
        .predicates
        .extend(trait_ty_predicates.instantiate_own(tcx, trait_to_impl_substs).predicates);

    debug!("compare_type_predicate_entailment: bounds={:?}", hybrid_preds);

    let normalize_cause = traits::ObligationCause::misc(impl_ty_span, impl_ty_hir_id);
    let param_env = ty::ParamEnv::new(
        tcx.intern_predicates(&hybrid_preds.predicates),
        Reveal::UserFacing,
        hir::Constness::NotConst,
    );
    let param_env = traits::normalize_param_env_or_error(tcx, param_env, normalize_cause);
    tcx.infer_ctxt().enter(|infcx| {
        let ocx = ObligationCtxt::new(&infcx);

        debug!("compare_type_predicate_entailment: caller_bounds={:?}", param_env.caller_bounds());

        let mut selcx = traits::SelectionContext::new(&infcx);

        assert_eq!(impl_ty_own_bounds.predicates.len(), impl_ty_own_bounds.spans.len());
        for (span, predicate) in
            std::iter::zip(impl_ty_own_bounds.spans, impl_ty_own_bounds.predicates)
        {
            let cause = ObligationCause::misc(span, impl_ty_hir_id);
            let traits::Normalized { value: predicate, obligations } =
                traits::normalize(&mut selcx, param_env, cause, predicate);

            let cause = ObligationCause::new(
                span,
                impl_ty_hir_id,
                ObligationCauseCode::CompareImplItemObligation {
                    impl_item_def_id: impl_ty.def_id.expect_local(),
                    trait_item_def_id: trait_ty.def_id,
                    kind: impl_ty.kind,
                },
            );
            ocx.register_obligations(obligations);
            ocx.register_obligation(traits::Obligation::new(cause, param_env, predicate));
        }

        // Check that all obligations are satisfied by the implementation's
        // version.
        let errors = ocx.select_all_or_error();
        if !errors.is_empty() {
            let reported = infcx.report_fulfillment_errors(&errors, None, false);
            return Err(reported);
        }

        // Finally, resolve all regions. This catches wily misuses of
        // lifetime parameters.
        let outlives_environment = OutlivesEnvironment::new(param_env);
        infcx.check_region_obligations_and_report_errors(
            impl_ty.def_id.expect_local(),
            &outlives_environment,
        );

        Ok(())
    })
}

/// Validate that `ProjectionCandidate`s created for this associated type will
/// be valid.
///
/// Usually given
///
/// trait X { type Y: Copy } impl X for T { type Y = S; }
///
/// We are able to normalize `<T as X>::U` to `S`, and so when we check the
/// impl is well-formed we have to prove `S: Copy`.
///
/// For default associated types the normalization is not possible (the value
/// from the impl could be overridden). We also can't normalize generic
/// associated types (yet) because they contain bound parameters.
#[instrument(level = "debug", skip(tcx))]
pub fn check_type_bounds<'tcx>(
    tcx: TyCtxt<'tcx>,
    trait_ty: &ty::AssocItem,
    impl_ty: &ty::AssocItem,
    impl_ty_span: Span,
    impl_trait_ref: ty::TraitRef<'tcx>,
) -> Result<(), ErrorGuaranteed> {
    // Given
    //
    // impl<A, B> Foo<u32> for (A, B) {
    //     type Bar<C> =...
    // }
    //
    // - `impl_trait_ref` would be `<(A, B) as Foo<u32>>
    // - `impl_ty_substs` would be `[A, B, ^0.0]` (`^0.0` here is the bound var with db 0 and index 0)
    // - `rebased_substs` would be `[(A, B), u32, ^0.0]`, combining the substs from
    //    the *trait* with the generic associated type parameters (as bound vars).
    //
    // A note regarding the use of bound vars here:
    // Imagine as an example
    // ```
    // trait Family {
    //     type Member<C: Eq>;
    // }
    //
    // impl Family for VecFamily {
    //     type Member<C: Eq> = i32;
    // }
    // ```
    // Here, we would generate
    // ```notrust
    // forall<C> { Normalize(<VecFamily as Family>::Member<C> => i32) }
    // ```
    // when we really would like to generate
    // ```notrust
    // forall<C> { Normalize(<VecFamily as Family>::Member<C> => i32) :- Implemented(C: Eq) }
    // ```
    // But, this is probably fine, because although the first clause can be used with types C that
    // do not implement Eq, for it to cause some kind of problem, there would have to be a
    // VecFamily::Member<X> for some type X where !(X: Eq), that appears in the value of type
    // Member<C: Eq> = .... That type would fail a well-formedness check that we ought to be doing
    // elsewhere, which would check that any <T as Family>::Member<X> meets the bounds declared in
    // the trait (notably, that X: Eq and T: Family).
    let defs: &ty::Generics = tcx.generics_of(impl_ty.def_id);
    let mut substs = smallvec::SmallVec::with_capacity(defs.count());
    if let Some(def_id) = defs.parent {
        let parent_defs = tcx.generics_of(def_id);
        InternalSubsts::fill_item(&mut substs, tcx, parent_defs, &mut |param, _| {
            tcx.mk_param_from_def(param)
        });
    }
    let mut bound_vars: smallvec::SmallVec<[ty::BoundVariableKind; 8]> =
        smallvec::SmallVec::with_capacity(defs.count());
    InternalSubsts::fill_single(&mut substs, defs, &mut |param, _| match param.kind {
        GenericParamDefKind::Type { .. } => {
            let kind = ty::BoundTyKind::Param(param.name);
            let bound_var = ty::BoundVariableKind::Ty(kind);
            bound_vars.push(bound_var);
            tcx.mk_ty(ty::Bound(
                ty::INNERMOST,
                ty::BoundTy { var: ty::BoundVar::from_usize(bound_vars.len() - 1), kind },
            ))
            .into()
        }
        GenericParamDefKind::Lifetime => {
            let kind = ty::BoundRegionKind::BrNamed(param.def_id, param.name);
            let bound_var = ty::BoundVariableKind::Region(kind);
            bound_vars.push(bound_var);
            tcx.mk_region(ty::ReLateBound(
                ty::INNERMOST,
                ty::BoundRegion { var: ty::BoundVar::from_usize(bound_vars.len() - 1), kind },
            ))
            .into()
        }
        GenericParamDefKind::Const { .. } => {
            let bound_var = ty::BoundVariableKind::Const;
            bound_vars.push(bound_var);
            tcx.mk_const(ty::ConstS {
                ty: tcx.type_of(param.def_id),
                kind: ty::ConstKind::Bound(
                    ty::INNERMOST,
                    ty::BoundVar::from_usize(bound_vars.len() - 1),
                ),
            })
            .into()
        }
    });
    let bound_vars = tcx.mk_bound_variable_kinds(bound_vars.into_iter());
    let impl_ty_substs = tcx.intern_substs(&substs);
    let container_id = impl_ty.container_id(tcx);

    let rebased_substs = impl_ty_substs.rebase_onto(tcx, container_id, impl_trait_ref.substs);
    let impl_ty_value = tcx.type_of(impl_ty.def_id);

    let param_env = tcx.param_env(impl_ty.def_id);

    // When checking something like
    //
    // trait X { type Y: PartialEq<<Self as X>::Y> }
    // impl X for T { default type Y = S; }
    //
    // We will have to prove the bound S: PartialEq<<T as X>::Y>. In this case
    // we want <T as X>::Y to normalize to S. This is valid because we are
    // checking the default value specifically here. Add this equality to the
    // ParamEnv for normalization specifically.
    let normalize_param_env = {
        let mut predicates = param_env.caller_bounds().iter().collect::<Vec<_>>();
        match impl_ty_value.kind() {
            ty::Projection(proj)
                if proj.item_def_id == trait_ty.def_id && proj.substs == rebased_substs =>
            {
                // Don't include this predicate if the projected type is
                // exactly the same as the projection. This can occur in
                // (somewhat dubious) code like this:
                //
                // impl<T> X for T where T: X { type Y = <T as X>::Y; }
            }
            _ => predicates.push(
                ty::Binder::bind_with_vars(
                    ty::ProjectionPredicate {
                        projection_ty: ty::ProjectionTy {
                            item_def_id: trait_ty.def_id,
                            substs: rebased_substs,
                        },
                        term: impl_ty_value.into(),
                    },
                    bound_vars,
                )
                .to_predicate(tcx),
            ),
        };
        ty::ParamEnv::new(
            tcx.intern_predicates(&predicates),
            Reveal::UserFacing,
            param_env.constness(),
        )
    };
    debug!(?normalize_param_env);

    let impl_ty_hir_id = tcx.hir().local_def_id_to_hir_id(impl_ty.def_id.expect_local());
    let impl_ty_substs = InternalSubsts::identity_for_item(tcx, impl_ty.def_id);
    let rebased_substs = impl_ty_substs.rebase_onto(tcx, container_id, impl_trait_ref.substs);

    tcx.infer_ctxt().enter(move |infcx| {
        let ocx = ObligationCtxt::new(&infcx);

        let assumed_wf_types =
            ocx.assumed_wf_types(param_env, impl_ty_span, impl_ty.def_id.expect_local());

        let mut selcx = traits::SelectionContext::new(&infcx);
        let normalize_cause = ObligationCause::new(
            impl_ty_span,
            impl_ty_hir_id,
            ObligationCauseCode::CheckAssociatedTypeBounds {
                impl_item_def_id: impl_ty.def_id.expect_local(),
                trait_item_def_id: trait_ty.def_id,
            },
        );
        let mk_cause = |span: Span| {
            let code = if span.is_dummy() {
                traits::ItemObligation(trait_ty.def_id)
            } else {
                traits::BindingObligation(trait_ty.def_id, span)
            };
            ObligationCause::new(impl_ty_span, impl_ty_hir_id, code)
        };

        let obligations = tcx
            .bound_explicit_item_bounds(trait_ty.def_id)
            .transpose_iter()
            .map(|e| e.map_bound(|e| *e).transpose_tuple2())
            .map(|(bound, span)| {
                debug!(?bound);
                // this is where opaque type is found
                let concrete_ty_bound = bound.subst(tcx, rebased_substs);
                debug!("check_type_bounds: concrete_ty_bound = {:?}", concrete_ty_bound);

                traits::Obligation::new(mk_cause(span.0), param_env, concrete_ty_bound)
            })
            .collect();
        debug!("check_type_bounds: item_bounds={:?}", obligations);

        for mut obligation in util::elaborate_obligations(tcx, obligations) {
            let traits::Normalized { value: normalized_predicate, obligations } = traits::normalize(
                &mut selcx,
                normalize_param_env,
                normalize_cause.clone(),
                obligation.predicate,
            );
            debug!("compare_projection_bounds: normalized predicate = {:?}", normalized_predicate);
            obligation.predicate = normalized_predicate;

            ocx.register_obligations(obligations);
            ocx.register_obligation(obligation);
        }
        // Check that all obligations are satisfied by the implementation's
        // version.
        let errors = ocx.select_all_or_error();
        if !errors.is_empty() {
            let reported = infcx.report_fulfillment_errors(&errors, None, false);
            return Err(reported);
        }

        // Finally, resolve all regions. This catches wily misuses of
        // lifetime parameters.
        let implied_bounds = infcx.implied_bounds_tys(param_env, impl_ty_hir_id, assumed_wf_types);
        let outlives_environment =
            OutlivesEnvironment::with_bounds(param_env, Some(&infcx), implied_bounds);

        infcx.check_region_obligations_and_report_errors(
            impl_ty.def_id.expect_local(),
            &outlives_environment,
        );

        let constraints = infcx.inner.borrow_mut().opaque_type_storage.take_opaque_types();
        for (key, value) in constraints {
            infcx
                .report_mismatched_types(
                    &ObligationCause::misc(
                        value.hidden_type.span,
                        tcx.hir().local_def_id_to_hir_id(impl_ty.def_id.expect_local()),
                    ),
                    tcx.mk_opaque(key.def_id.to_def_id(), key.substs),
                    value.hidden_type.ty,
                    TypeError::Mismatch,
                )
                .emit();
        }

        Ok(())
    })
}

fn assoc_item_kind_str(impl_item: &ty::AssocItem) -> &'static str {
    match impl_item.kind {
        ty::AssocKind::Const => "const",
        ty::AssocKind::Fn => "method",
        ty::AssocKind::Type => "type",
    }
}
