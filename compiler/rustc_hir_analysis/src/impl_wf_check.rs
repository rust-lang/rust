//! This pass enforces various "well-formedness constraints" on impls.
//! Logically, it is part of wfcheck -- but we do it early so that we
//! can stop compilation afterwards, since part of the trait matching
//! infrastructure gets very grumpy if these conditions don't hold. In
//! particular, if there are type parameters that are not part of the
//! impl, then coherence will report strange inference ambiguity
//! errors; if impls have duplicate items, we get misleading
//! specialization errors. These things can (and probably should) be
//! fixed, but for the moment it's easier to do these checks early.

use std::assert_matches::debug_assert_matches;

use itertools::Itertools;
use rustc_data_structures::fx::FxHashSet;
use rustc_errors::codes::*;
use rustc_hir::def::DefKind;
use rustc_hir::def_id::{DefId, LOCAL_CRATE, LocalDefId};
use rustc_infer::infer::TyCtxtInferExt;
use rustc_lint_defs::builtin::DYN_OVERLAP;
use rustc_middle::ty::{
    self, ExistentialPredicateStableCmpExt, Ty, TyCtxt, TypeVisitableExt, TypingMode, Upcast,
    elaborate,
};
use rustc_span::{DUMMY_SP, ErrorGuaranteed, sym};
use rustc_trait_selection::traits::{Obligation, ObligationCause, ObligationCtxt};

use crate::constrained_generic_params as cgp;
use crate::errors::UnconstrainedGenericParameter;
use crate::impl_wf_check::min_specialization::check_min_specialization;

mod min_specialization;

/// Checks that all the type/lifetime parameters on an impl also
/// appear in the trait ref or self type (or are constrained by a
/// where-clause). These rules are needed to ensure that, given a
/// trait ref like `<T as Trait<U>>`, we can derive the values of all
/// parameters on the impl (which is needed to make specialization
/// possible).
///
/// However, in the case of lifetimes, we only enforce these rules if
/// the lifetime parameter is used in an associated type. This is a
/// concession to backwards compatibility; see comment at the end of
/// the fn for details.
///
/// Example:
///
/// ```rust,ignore (pseudo-Rust)
/// impl<T> Trait<Foo> for Bar { ... }
/// //   ^ T does not appear in `Foo` or `Bar`, error!
///
/// impl<T> Trait<Foo<T>> for Bar { ... }
/// //   ^ T appears in `Foo<T>`, ok.
///
/// impl<T> Trait<Foo> for Bar where Bar: Iterator<Item = T> { ... }
/// //   ^ T is bound to `<Bar as Iterator>::Item`, ok.
///
/// impl<'a> Trait<Foo> for Bar { }
/// //   ^ 'a is unused, but for back-compat we allow it
///
/// impl<'a> Trait<Foo> for Bar { type X = &'a i32; }
/// //   ^ 'a is unused and appears in assoc type, error
/// ```
pub(crate) fn check_impl_wf(
    tcx: TyCtxt<'_>,
    impl_def_id: LocalDefId,
) -> Result<(), ErrorGuaranteed> {
    debug_assert_matches!(tcx.def_kind(impl_def_id), DefKind::Impl { .. });

    // Check that the args are constrained. We queryfied the check for ty/const params
    // since unconstrained type/const params cause ICEs in projection, so we want to
    // detect those specifically and project those to `TyKind::Error`.
    let mut res = tcx.ensure_ok().enforce_impl_non_lifetime_params_are_constrained(impl_def_id);
    res = res.and(enforce_impl_lifetime_params_are_constrained(tcx, impl_def_id));

    if tcx.features().min_specialization() {
        res = res.and(check_min_specialization(tcx, impl_def_id));
    }

    if let Some(trait_def_id) = tcx.trait_id_of_impl(impl_def_id.to_def_id()) {
        for &subtrait_def_id in tcx
            .crates(())
            .into_iter()
            .copied()
            .chain([LOCAL_CRATE])
            .flat_map(|cnum| tcx.traits(cnum))
        {
            if ty::elaborate::supertrait_def_ids(tcx, subtrait_def_id).contains(&trait_def_id) {
                tcx.ensure_ok()
                    .lint_object_blanket_impl((impl_def_id.to_def_id(), subtrait_def_id));
            }
        }
    }

    res
}

pub(crate) fn enforce_impl_lifetime_params_are_constrained(
    tcx: TyCtxt<'_>,
    impl_def_id: LocalDefId,
) -> Result<(), ErrorGuaranteed> {
    let impl_self_ty = tcx.type_of(impl_def_id).instantiate_identity();
    if impl_self_ty.references_error() {
        // Don't complain about unconstrained type params when self ty isn't known due to errors.
        // (#36836)
        tcx.dcx().span_delayed_bug(
            tcx.def_span(impl_def_id),
            format!(
                "potentially unconstrained type parameters weren't evaluated: {impl_self_ty:?}",
            ),
        );
        // This is super fishy, but our current `rustc_hir_analysis::check_crate` pipeline depends on
        // `type_of` having been called much earlier, and thus this value being read from cache.
        // Compilation must continue in order for other important diagnostics to keep showing up.
        return Ok(());
    }

    let impl_generics = tcx.generics_of(impl_def_id);
    let impl_predicates = tcx.predicates_of(impl_def_id);
    let impl_trait_ref = tcx.impl_trait_ref(impl_def_id).map(ty::EarlyBinder::instantiate_identity);

    impl_trait_ref.error_reported()?;

    let mut input_parameters = cgp::parameters_for_impl(tcx, impl_self_ty, impl_trait_ref);
    cgp::identify_constrained_generic_params(
        tcx,
        impl_predicates,
        impl_trait_ref,
        &mut input_parameters,
    );

    // Disallow unconstrained lifetimes, but only if they appear in assoc types.
    let lifetimes_in_associated_types: FxHashSet<_> = tcx
        .associated_item_def_ids(impl_def_id)
        .iter()
        .flat_map(|def_id| {
            let item = tcx.associated_item(def_id);
            match item.kind {
                ty::AssocKind::Type { .. } => {
                    if item.defaultness(tcx).has_value() {
                        cgp::parameters_for(tcx, tcx.type_of(def_id).instantiate_identity(), true)
                    } else {
                        vec![]
                    }
                }
                ty::AssocKind::Fn { .. } | ty::AssocKind::Const { .. } => vec![],
            }
        })
        .collect();

    let mut res = Ok(());
    for param in &impl_generics.own_params {
        match param.kind {
            ty::GenericParamDefKind::Lifetime => {
                // This is a horrible concession to reality. I think it'd be
                // better to just ban unconstrained lifetimes outright, but in
                // practice people do non-hygienic macros like:
                //
                // ```
                // macro_rules! __impl_slice_eq1 {
                //   ($Lhs: ty, $Rhs: ty, $Bound: ident) => {
                //     impl<'a, 'b, A: $Bound, B> PartialEq<$Rhs> for $Lhs where A: PartialEq<B> {
                //        ....
                //     }
                //   }
                // }
                // ```
                //
                // In a concession to backwards compatibility, we continue to
                // permit those, so long as the lifetimes aren't used in
                // associated types. I believe this is sound, because lifetimes
                // used elsewhere are not projected back out.
                let param_lt = cgp::Parameter::from(param.to_early_bound_region_data());
                if lifetimes_in_associated_types.contains(&param_lt)
                    && !input_parameters.contains(&param_lt)
                {
                    let mut diag = tcx.dcx().create_err(UnconstrainedGenericParameter {
                        span: tcx.def_span(param.def_id),
                        param_name: tcx.item_ident(param.def_id),
                        param_def_kind: tcx.def_descr(param.def_id),
                        const_param_note: false,
                        const_param_note2: false,
                    });
                    diag.code(E0207);
                    res = Err(diag.emit());
                }
            }
            ty::GenericParamDefKind::Type { .. } | ty::GenericParamDefKind::Const { .. } => {
                // Enforced in `enforce_impl_non_lifetime_params_are_constrained`.
            }
        }
    }
    res
}

pub(crate) fn enforce_impl_non_lifetime_params_are_constrained(
    tcx: TyCtxt<'_>,
    impl_def_id: LocalDefId,
) -> Result<(), ErrorGuaranteed> {
    let impl_self_ty = tcx.type_of(impl_def_id).instantiate_identity();
    if impl_self_ty.references_error() {
        // Don't complain about unconstrained type params when self ty isn't known due to errors.
        // (#36836)
        tcx.dcx().span_delayed_bug(
            tcx.def_span(impl_def_id),
            format!(
                "potentially unconstrained type parameters weren't evaluated: {impl_self_ty:?}",
            ),
        );
        // This is super fishy, but our current `rustc_hir_analysis::check_crate` pipeline depends on
        // `type_of` having been called much earlier, and thus this value being read from cache.
        // Compilation must continue in order for other important diagnostics to keep showing up.
        return Ok(());
    }
    let impl_generics = tcx.generics_of(impl_def_id);
    let impl_predicates = tcx.predicates_of(impl_def_id);
    let impl_trait_ref = tcx.impl_trait_ref(impl_def_id).map(ty::EarlyBinder::instantiate_identity);

    impl_trait_ref.error_reported()?;

    let mut input_parameters = cgp::parameters_for_impl(tcx, impl_self_ty, impl_trait_ref);
    cgp::identify_constrained_generic_params(
        tcx,
        impl_predicates,
        impl_trait_ref,
        &mut input_parameters,
    );

    let mut res = Ok(());
    for param in &impl_generics.own_params {
        let err = match param.kind {
            // Disallow ANY unconstrained type parameters.
            ty::GenericParamDefKind::Type { .. } => {
                let param_ty = ty::ParamTy::for_def(param);
                !input_parameters.contains(&cgp::Parameter::from(param_ty))
            }
            ty::GenericParamDefKind::Const { .. } => {
                let param_ct = ty::ParamConst::for_def(param);
                !input_parameters.contains(&cgp::Parameter::from(param_ct))
            }
            ty::GenericParamDefKind::Lifetime => {
                // Enforced in `enforce_impl_type_params_are_constrained`.
                false
            }
        };
        if err {
            let const_param_note = matches!(param.kind, ty::GenericParamDefKind::Const { .. });
            let mut diag = tcx.dcx().create_err(UnconstrainedGenericParameter {
                span: tcx.def_span(param.def_id),
                param_name: tcx.item_ident(param.def_id),
                param_def_kind: tcx.def_descr(param.def_id),
                const_param_note,
                const_param_note2: const_param_note,
            });
            diag.code(E0207);
            res = Err(diag.emit());
        }
    }
    res
}

pub(crate) fn lint_object_blanket_impl<'tcx>(
    tcx: TyCtxt<'tcx>,
    (impl_def_id, trait_def_id): (DefId, DefId),
) {
    if tcx.is_diagnostic_item(sym::Any, trait_def_id) {
        return;
    }

    if !tcx.is_dyn_compatible(trait_def_id) {
        return;
    }

    let infcx = tcx.infer_ctxt().with_next_trait_solver(true).build(TypingMode::CheckObjectOverlap);

    let principal_trait_args = infcx.fresh_args_for_item(DUMMY_SP, trait_def_id);
    let principal_trait = ty::TraitRef::new_from_args(tcx, trait_def_id, principal_trait_args);

    let mut needed_associated_types = vec![];
    let clause: ty::Clause<'tcx> = ty::TraitRef::identity(tcx, trait_def_id).upcast(tcx);
    for clause in elaborate::elaborate(tcx, [clause]).filter_only_self() {
        let clause = clause.instantiate_supertrait(tcx, ty::Binder::dummy(principal_trait));

        let bound_predicate = clause.kind();
        match bound_predicate.skip_binder() {
            ty::ClauseKind::Trait(pred) => {
                // FIXME(negative_bounds): Handle this correctly...
                let trait_ref = tcx.anonymize_bound_vars(bound_predicate.rebind(pred.trait_ref));
                needed_associated_types.extend(
                    tcx.associated_items(pred.trait_ref.def_id)
                        .in_definition_order()
                        // We only care about associated types.
                        .filter(|item| item.is_type())
                        // No RPITITs -- they're not dyn-compatible for now.
                        .filter(|item| !item.is_impl_trait_in_trait())
                        // If the associated type has a `where Self: Sized` bound,
                        // we do not need to constrain the associated type.
                        .filter(|item| !tcx.generics_require_sized_self(item.def_id))
                        .map(|item| (item.def_id, trait_ref)),
                );
            }
            _ => (),
        }
    }

    let mut data: Vec<_> = [ty::Binder::dummy(ty::ExistentialPredicate::Trait(
        ty::ExistentialTraitRef::erase_self_ty(tcx, principal_trait),
    ))]
    .into_iter()
    .chain(needed_associated_types.into_iter().map(|(def_id, trait_ref)| {
        trait_ref.map_bound(|trait_ref| {
            ty::ExistentialPredicate::Projection(ty::ExistentialProjection::erase_self_ty(
                tcx,
                ty::ProjectionPredicate {
                    projection_term: ty::AliasTerm::new_from_args(tcx, def_id, trait_ref.args),
                    term: infcx.next_ty_var(DUMMY_SP).into(),
                },
            ))
        })
    }))
    .collect();
    data.sort_by(|a, b| a.skip_binder().stable_cmp(tcx, &b.skip_binder()));

    let self_ty = Ty::new_dynamic(
        tcx,
        tcx.mk_poly_existential_predicates(&data),
        tcx.lifetimes.re_erased,
        ty::Dyn,
    );

    let impl_args = infcx.fresh_args_for_item(DUMMY_SP, impl_def_id);
    let impl_trait_ref = tcx.impl_trait_ref(impl_def_id).unwrap().instantiate(tcx, impl_args);

    let ocx = ObligationCtxt::new(&infcx);
    let Ok(()) =
        ocx.eq(&ObligationCause::dummy(), ty::ParamEnv::empty(), principal_trait, impl_trait_ref)
    else {
        return;
    };
    let Ok(()) =
        ocx.eq(&ObligationCause::dummy(), ty::ParamEnv::empty(), self_ty, impl_trait_ref.self_ty())
    else {
        return;
    };

    ocx.register_obligations(
        tcx.predicates_of(impl_def_id).instantiate(tcx, impl_args).into_iter().map(
            |(clause, _)| {
                Obligation::new(tcx, ObligationCause::dummy(), ty::ParamEnv::empty(), clause)
            },
        ),
    );

    if !ocx.select_where_possible().is_empty() {
        return;
    }

    let local_def_id = if let Some(impl_def_id) = impl_def_id.as_local() {
        impl_def_id
    } else if let Some(trait_def_id) = trait_def_id.as_local() {
        trait_def_id
    } else {
        panic!()
    };
    let hir_id = tcx.local_def_id_to_hir_id(local_def_id);

    let self_ty = infcx.resolve_vars_if_possible(self_ty);

    tcx.node_span_lint(DYN_OVERLAP, hir_id, tcx.def_span(local_def_id), |diag| {
        diag.primary_message("hi");
        diag.span_label(
            tcx.def_span(trait_def_id),
            format!("built-in `{self_ty}` implementation for this trait"),
        );
        diag.span_label(tcx.def_span(impl_def_id), "overlaps with this blanket impl");
    });
}
