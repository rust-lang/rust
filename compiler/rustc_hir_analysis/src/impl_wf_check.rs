//! This pass enforces various "well-formedness constraints" on impls.
//! Logically, it is part of wfcheck -- but we do it early so that we
//! can stop compilation afterwards, since part of the trait matching
//! infrastructure gets very grumpy if these conditions don't hold. In
//! particular, if there are type parameters that are not part of the
//! impl, then coherence will report strange inference ambiguity
//! errors; if impls have duplicate items, we get misleading
//! specialization errors. These things can (and probably should) be
//! fixed, but for the moment it's easier to do these checks early.

use crate::constrained_generic_params as cgp;
use min_specialization::check_min_specialization;

use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_errors::{codes::*, struct_span_code_err};
use rustc_hir::def::DefKind;
use rustc_hir::def_id::LocalDefId;
use rustc_middle::ty::{self, TyCtxt, TypeVisitableExt};
use rustc_span::{ErrorGuaranteed, Span, Symbol};

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
pub fn check_impl_wf(tcx: TyCtxt<'_>, impl_def_id: LocalDefId) -> Result<(), ErrorGuaranteed> {
    let min_specialization = tcx.features().min_specialization;
    let mut res = Ok(());
    debug_assert!(matches!(tcx.def_kind(impl_def_id), DefKind::Impl { .. }));
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

    let mut constrained_parameters = cgp::parameters_for_impl(tcx, impl_self_ty, impl_trait_ref);
    cgp::identify_constrained_generic_params(
        tcx,
        impl_predicates,
        impl_trait_ref,
        &mut constrained_parameters,
    );

    // Reasons each generic is unconstrained, or Other if not present
    let mut unconstrained_reasons = FxHashMap::default();
    for (clause, _) in impl_predicates.predicates {
        if let Some(projection) = clause.as_projection_clause() {
            let mentioned_params = cgp::parameters_for(tcx, projection.term().skip_binder(), true);
            let is_clause_circular =
                Some(projection.required_poly_trait_ref(tcx).skip_binder()) == impl_trait_ref;

            for param in mentioned_params {
                if is_clause_circular {
                    // Potentially override BoundToUnconstrainedProjection
                    unconstrained_reasons.insert(param, UnconstrainedReason::BoundCircularly);
                } else {
                    // Don't override BoundCircularly
                    unconstrained_reasons
                        .entry(param)
                        .or_insert(UnconstrainedReason::BoundToUnconstrainedProjection);
                }
            }
        }
    }

    // Disallow unconstrained lifetimes, but only if they appear in assoc types.
    let lifetimes_in_associated_types: FxHashSet<_> = tcx
        .associated_item_def_ids(impl_def_id)
        .iter()
        .flat_map(|def_id| {
            let item = tcx.associated_item(def_id);
            match item.kind {
                ty::AssocKind::Type => {
                    if item.defaultness(tcx).has_value() {
                        cgp::parameters_for(tcx, tcx.type_of(def_id).instantiate_identity(), true)
                    } else {
                        vec![]
                    }
                }
                ty::AssocKind::Fn | ty::AssocKind::Const => vec![],
            }
        })
        .collect();

    let impl_kind = match impl_trait_ref {
        Some(_) => ImplKind::ImplTrait,
        None => ImplKind::InherentImpl,
    };

    let mut res = Ok(());
    for param in &impl_generics.own_params {
        let cgp_param = cgp::Parameter(param.index);
        let unconstrained_reason =
            unconstrained_reasons.get(&cgp_param).copied().unwrap_or_default();

        match param.kind {
            // Disallow ANY unconstrained type parameters.
            ty::GenericParamDefKind::Type { .. } => {
                let param_ty = ty::ParamTy::for_def(param);
                if !constrained_parameters.contains(&cgp_param) {
                    res = Err(report_unused_parameter(
                        tcx,
                        tcx.def_span(param.def_id),
                        "type",
                        param_ty.name,
                        impl_kind,
                        unconstrained_reason,
                    ));
                }
            }
            ty::GenericParamDefKind::Lifetime => {
                if lifetimes_in_associated_types.contains(&cgp_param) && // (*)
                    !constrained_parameters.contains(&cgp_param)
                {
                    res = Err(report_unused_parameter(
                        tcx,
                        tcx.def_span(param.def_id),
                        "lifetime",
                        param.name,
                        impl_kind,
                        unconstrained_reason,
                    ));
                }
            }
            ty::GenericParamDefKind::Const { .. } => {
                let param_ct = ty::ParamConst::for_def(param);
                if !constrained_parameters.contains(&cgp_param) {
                    res = Err(report_unused_parameter(
                        tcx,
                        tcx.def_span(param.def_id),
                        "const",
                        param_ct.name,
                        impl_kind,
                        unconstrained_reason,
                    ));
                }
            }
        }
    }
    res

    // (*) This is a horrible concession to reality. I think it'd be
    // better to just ban unconstrained lifetimes outright, but in
    // practice people do non-hygienic macros like:
    //
    // ```
    // macro_rules! __impl_slice_eq1 {
    //     ($Lhs: ty, $Rhs: ty, $Bound: ident) => {
    //         impl<'a, 'b, A: $Bound, B> PartialEq<$Rhs> for $Lhs where A: PartialEq<B> {
    //            ....
    //         }
    //     }
    // }
    // ```
    //
    // In a concession to backwards compatibility, we continue to
    // permit those, so long as the lifetimes aren't used in
    // associated types. I believe this is sound, because lifetimes
    // used elsewhere are not projected back out.
}

#[derive(Copy, Clone)]
enum ImplKind {
    ImplTrait,
    InherentImpl,
}

#[derive(Copy, Clone, Default)]
enum UnconstrainedReason {
    /// Not used in the impl trait, self type, nor bound to any projection
    #[default]
    Other,
    /// Bound to a projection, but the LHS was not constrained
    BoundToUnconstrainedProjection,
    /// Bound to a projection, but the LHS is the trait being implemented
    BoundCircularly,
}

fn report_unused_parameter(
    tcx: TyCtxt<'_>,
    span: Span,
    kind: &str,
    name: Symbol,
    impl_kind: ImplKind,
    unconstrained_reason: UnconstrainedReason,
) -> ErrorGuaranteed {
    let mut err = struct_span_code_err!(
        tcx.dcx(),
        span,
        E0207,
        "the {} parameter `{}` is not constrained by the \
        impl trait, self type, or predicates",
        kind,
        name
    );
    err.span_label(span, format!("unconstrained {kind} parameter"));

    match impl_kind {
        ImplKind::ImplTrait => {
            err.note(format!(
                "to constrain `{name}`, use it in the implemented trait, in the self type, \
                or in an equality with an associated type"
            ));
        }
        ImplKind::InherentImpl => {
            err.note(format!(
                "to constrain `{name}`, use it in the self type, \
                or in an equality with an associated type"
            ));
        }
    }

    match unconstrained_reason {
        UnconstrainedReason::BoundToUnconstrainedProjection => {
            err.note(format!(
                "`{name}` is bound to an associated type, \
                but the reference to the associated type itself uses unconstrained generic parameters"
            ));
        }
        UnconstrainedReason::BoundCircularly => {
            err.note(format!(
                "`{name}` is bound to an associated type, \
                but the associated type is circularly defined in this impl"
            ));
        }
        UnconstrainedReason::Other => {}
    }

    if kind == "const" {
        err.note(
            "expressions using a const parameter must map each value to a distinct output value",
        );
        err.note(
            "proving the result of expressions other than the parameter are unique is not supported",
        );
    }

    err.emit()
}
