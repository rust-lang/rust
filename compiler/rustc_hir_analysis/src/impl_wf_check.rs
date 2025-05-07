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

use min_specialization::check_min_specialization;
use rustc_data_structures::fx::FxHashSet;
use rustc_errors::codes::*;
use rustc_hir::def::DefKind;
use rustc_hir::def_id::LocalDefId;
use rustc_middle::ty::{self, TyCtxt, TypeVisitableExt};
use rustc_span::ErrorGuaranteed;

use crate::constrained_generic_params as cgp;
use crate::errors::UnconstrainedGenericParameter;

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
