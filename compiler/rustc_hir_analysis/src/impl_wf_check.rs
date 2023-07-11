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

use rustc_data_structures::fx::FxHashSet;
use rustc_errors::struct_span_err;
use rustc_hir::def::DefKind;
use rustc_hir::def_id::LocalDefId;
use rustc_middle::query::Providers;
use rustc_middle::ty::{self, TyCtxt, TypeVisitableExt};
use rustc_span::{Span, Symbol};

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
fn check_mod_impl_wf(tcx: TyCtxt<'_>, module_def_id: LocalDefId) {
    let min_specialization = tcx.features().min_specialization;
    let module = tcx.hir_module_items(module_def_id);
    for id in module.items() {
        if matches!(tcx.def_kind(id.owner_id), DefKind::Impl { .. }) {
            enforce_impl_params_are_constrained(tcx, id.owner_id.def_id);
            if min_specialization {
                check_min_specialization(tcx, id.owner_id.def_id);
            }
        }
    }
}

pub fn provide(providers: &mut Providers) {
    *providers = Providers { check_mod_impl_wf, ..*providers };
}

fn enforce_impl_params_are_constrained(tcx: TyCtxt<'_>, impl_def_id: LocalDefId) {
    // Every lifetime used in an associated type must be constrained.
    let impl_self_ty = tcx.type_of(impl_def_id).instantiate_identity();
    if impl_self_ty.references_error() {
        // Don't complain about unconstrained type params when self ty isn't known due to errors.
        // (#36836)
        tcx.sess.delay_span_bug(
            tcx.def_span(impl_def_id),
            format!(
                "potentially unconstrained type parameters weren't evaluated: {:?}",
                impl_self_ty,
            ),
        );
        return;
    }
    let impl_generics = tcx.generics_of(impl_def_id);
    let impl_predicates = tcx.predicates_of(impl_def_id);
    let impl_trait_ref = tcx.impl_trait_ref(impl_def_id).map(ty::EarlyBinder::instantiate_identity);

    let mut input_parameters = cgp::parameters_for_impl(impl_self_ty, impl_trait_ref);
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
                ty::AssocKind::Type => {
                    if item.defaultness(tcx).has_value() {
                        cgp::parameters_for(&tcx.type_of(def_id).instantiate_identity(), true)
                    } else {
                        vec![]
                    }
                }
                ty::AssocKind::Fn | ty::AssocKind::Const => vec![],
            }
        })
        .collect();

    for param in &impl_generics.params {
        match param.kind {
            // Disallow ANY unconstrained type parameters.
            ty::GenericParamDefKind::Type { .. } => {
                let param_ty = ty::ParamTy::for_def(param);
                if !input_parameters.contains(&cgp::Parameter::from(param_ty)) {
                    report_unused_parameter(tcx, tcx.def_span(param.def_id), "type", param_ty.name);
                }
            }
            ty::GenericParamDefKind::Lifetime => {
                let param_lt = cgp::Parameter::from(param.to_early_bound_region_data());
                if lifetimes_in_associated_types.contains(&param_lt) && // (*)
                    !input_parameters.contains(&param_lt)
                {
                    report_unused_parameter(
                        tcx,
                        tcx.def_span(param.def_id),
                        "lifetime",
                        param.name,
                    );
                }
            }
            ty::GenericParamDefKind::Const { .. } => {
                let param_ct = ty::ParamConst::for_def(param);
                if !input_parameters.contains(&cgp::Parameter::from(param_ct)) {
                    report_unused_parameter(
                        tcx,
                        tcx.def_span(param.def_id),
                        "const",
                        param_ct.name,
                    );
                }
            }
        }
    }

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

fn report_unused_parameter(tcx: TyCtxt<'_>, span: Span, kind: &str, name: Symbol) {
    let mut err = struct_span_err!(
        tcx.sess,
        span,
        E0207,
        "the {} parameter `{}` is not constrained by the \
        impl trait, self type, or predicates",
        kind,
        name
    );
    err.span_label(span, format!("unconstrained {} parameter", kind));
    if kind == "const" {
        err.note(
            "expressions using a const parameter must map each value to a distinct output value",
        );
        err.note(
            "proving the result of expressions other than the parameter are unique is not supported",
        );
    }
    err.emit();
}
