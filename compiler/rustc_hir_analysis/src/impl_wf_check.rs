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
    let min_specialization = tcx.features().min_specialization();
    let mut res = Ok(());
    debug_assert_matches!(tcx.def_kind(impl_def_id), DefKind::Impl { .. });
    res = res.and(enforce_impl_params_are_constrained(tcx, impl_def_id));
    if min_specialization {
        res = res.and(check_min_specialization(tcx, impl_def_id));
    }

    res
}

fn enforce_impl_params_are_constrained(
    tcx: TyCtxt<'_>,
    impl_def_id: LocalDefId,
) -> Result<(), ErrorGuaranteed> {
    // Every lifetime used in an associated type must be constrained.
    let impl_self_ty = tcx.type_of(impl_def_id).instantiate_identity();
    impl_self_ty.error_reported()?;
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
            ty::GenericParamDefKind::Lifetime => false,
            ty::GenericParamDefKind::Const { .. } => {
                let param_ct = ty::ParamConst::for_def(param);
                !input_parameters.contains(&cgp::Parameter::from(param_ct))
            }
        };
        if err {
            let const_param_note = matches!(param.kind, ty::GenericParamDefKind::Const { .. });
            let mut diag = tcx.dcx().create_err(UnconstrainedGenericParameter {
                span: tcx.def_span(param.def_id),
                param_name: param.name,
                param_def_kind: tcx.def_descr(param.def_id),
                const_param_note,
                const_param_note2: const_param_note,
                lifetime_help: None,
            });
            diag.code(E0207);
            res = Err(diag.emit());
        }
    }
    res
}
