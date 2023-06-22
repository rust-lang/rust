// Coherence phase
//
// The job of the coherence phase of typechecking is to ensure that
// each trait has at most one implementation for each type. This is
// done by the orphan and overlap modules. Then we build up various
// mappings. That mapping code resides here.

use crate::errors;
use rustc_errors::{error_code, struct_span_err};
use rustc_hir::def_id::{DefId, LocalDefId};
use rustc_middle::query::Providers;
use rustc_middle::ty::{self, TyCtxt, TypeVisitableExt};
use rustc_trait_selection::traits;

mod builtin;
mod inherent_impls;
mod inherent_impls_overlap;
mod orphan;
mod unsafety;

fn check_impl(tcx: TyCtxt<'_>, impl_def_id: LocalDefId, trait_ref: ty::TraitRef<'_>) {
    debug!(
        "(checking implementation) adding impl for trait '{:?}', item '{}'",
        trait_ref,
        tcx.def_path_str(impl_def_id)
    );

    // Skip impls where one of the self type is an error type.
    // This occurs with e.g., resolve failures (#30589).
    if trait_ref.references_error() {
        return;
    }

    enforce_trait_manually_implementable(tcx, impl_def_id, trait_ref.def_id);
    enforce_empty_impls_for_marker_traits(tcx, impl_def_id, trait_ref.def_id);
}

fn enforce_trait_manually_implementable(
    tcx: TyCtxt<'_>,
    impl_def_id: LocalDefId,
    trait_def_id: DefId,
) {
    let impl_header_span = tcx.def_span(impl_def_id);

    // Disallow *all* explicit impls of traits marked `#[rustc_deny_explicit_impl]`
    if tcx.trait_def(trait_def_id).deny_explicit_impl {
        let trait_name = tcx.item_name(trait_def_id);
        let mut err = struct_span_err!(
            tcx.sess,
            impl_header_span,
            E0322,
            "explicit impls for the `{trait_name}` trait are not permitted"
        );
        err.span_label(impl_header_span, format!("impl of `{trait_name}` not allowed"));

        // Maintain explicit error code for `Unsize`, since it has a useful
        // explanation about using `CoerceUnsized` instead.
        if Some(trait_def_id) == tcx.lang_items().unsize_trait() {
            err.code(error_code!(E0328));
        }

        err.emit();
        return;
    }

    if let ty::trait_def::TraitSpecializationKind::AlwaysApplicable =
        tcx.trait_def(trait_def_id).specialization_kind
    {
        if !tcx.features().specialization && !tcx.features().min_specialization {
            tcx.sess.emit_err(errors::SpecializationTrait { span: impl_header_span });
            return;
        }
    }
}

/// We allow impls of marker traits to overlap, so they can't override impls
/// as that could make it ambiguous which associated item to use.
fn enforce_empty_impls_for_marker_traits(
    tcx: TyCtxt<'_>,
    impl_def_id: LocalDefId,
    trait_def_id: DefId,
) {
    if !tcx.trait_def(trait_def_id).is_marker {
        return;
    }

    if tcx.associated_item_def_ids(trait_def_id).is_empty() {
        return;
    }

    struct_span_err!(
        tcx.sess,
        tcx.def_span(impl_def_id),
        E0715,
        "impls for marker traits cannot contain items"
    )
    .emit();
}

pub fn provide(providers: &mut Providers) {
    use self::builtin::coerce_unsized_info;
    use self::inherent_impls::{crate_incoherent_impls, crate_inherent_impls, inherent_impls};
    use self::inherent_impls_overlap::crate_inherent_impls_overlap_check;
    use self::orphan::orphan_check_impl;

    *providers = Providers {
        coherent_trait,
        crate_inherent_impls,
        crate_incoherent_impls,
        inherent_impls,
        crate_inherent_impls_overlap_check,
        coerce_unsized_info,
        orphan_check_impl,
        ..*providers
    };
}

fn coherent_trait(tcx: TyCtxt<'_>, def_id: DefId) {
    // Trigger building the specialization graph for the trait. This will detect and report any
    // overlap errors.
    tcx.ensure().specialization_graph_of(def_id);

    let impls = tcx.hir().trait_impls(def_id);
    for &impl_def_id in impls {
        let trait_ref = tcx.impl_trait_ref(impl_def_id).unwrap().subst_identity();

        check_impl(tcx, impl_def_id, trait_ref);
        check_object_overlap(tcx, impl_def_id, trait_ref);

        unsafety::check_item(tcx, impl_def_id);
        tcx.ensure().orphan_check_impl(impl_def_id);
    }

    builtin::check_trait(tcx, def_id);
}

/// Checks whether an impl overlaps with the automatic `impl Trait for dyn Trait`.
fn check_object_overlap<'tcx>(
    tcx: TyCtxt<'tcx>,
    impl_def_id: LocalDefId,
    trait_ref: ty::TraitRef<'tcx>,
) {
    let trait_def_id = trait_ref.def_id;

    if trait_ref.references_error() {
        debug!("coherence: skipping impl {:?} with error {:?}", impl_def_id, trait_ref);
        return;
    }

    // check for overlap with the automatic `impl Trait for dyn Trait`
    if let ty::Dynamic(data, ..) = trait_ref.self_ty().kind() {
        // This is something like impl Trait1 for Trait2. Illegal
        // if Trait1 is a supertrait of Trait2 or Trait2 is not object safe.

        let component_def_ids = data.iter().flat_map(|predicate| {
            match predicate.skip_binder() {
                ty::ExistentialPredicate::Trait(tr) => Some(tr.def_id),
                ty::ExistentialPredicate::AutoTrait(def_id) => Some(def_id),
                // An associated type projection necessarily comes with
                // an additional `Trait` requirement.
                ty::ExistentialPredicate::Projection(..) => None,
            }
        });

        for component_def_id in component_def_ids {
            if !tcx.check_is_object_safe(component_def_id) {
                // Without the 'object_safe_for_dispatch' feature this is an error
                // which will be reported by wfcheck. Ignore it here.
                // This is tested by `coherence-impl-trait-for-trait-object-safe.rs`.
                // With the feature enabled, the trait is not implemented automatically,
                // so this is valid.
            } else {
                let mut supertrait_def_ids = traits::supertrait_def_ids(tcx, component_def_id);
                if supertrait_def_ids.any(|d| d == trait_def_id) {
                    let span = tcx.def_span(impl_def_id);
                    struct_span_err!(
                        tcx.sess,
                        span,
                        E0371,
                        "the object type `{}` automatically implements the trait `{}`",
                        trait_ref.self_ty(),
                        tcx.def_path_str(trait_def_id)
                    )
                    .span_label(
                        span,
                        format!(
                            "`{}` automatically implements trait `{}`",
                            trait_ref.self_ty(),
                            tcx.def_path_str(trait_def_id)
                        ),
                    )
                    .emit();
                }
            }
        }
    }
}
