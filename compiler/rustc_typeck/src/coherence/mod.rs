// Coherence phase
//
// The job of the coherence phase of typechecking is to ensure that
// each trait has at most one implementation for each type. This is
// done by the orphan and overlap modules. Then we build up various
// mappings. That mapping code resides here.

use rustc_errors::struct_span_err;
use rustc_hir::def_id::{DefId, LocalDefId};
use rustc_middle::ty::query::Providers;
use rustc_middle::ty::{self, TyCtxt, TypeFoldable};
use rustc_span::Span;
use rustc_trait_selection::traits;

mod builtin;
mod inherent_impls;
mod inherent_impls_overlap;
mod orphan;
mod unsafety;

/// Obtains the span of just the impl header of `impl_def_id`.
fn impl_header_span(tcx: TyCtxt<'_>, impl_def_id: LocalDefId) -> Span {
    tcx.sess.source_map().guess_head_span(tcx.span_of_impl(impl_def_id.to_def_id()).unwrap())
}

fn check_impl(tcx: TyCtxt<'_>, impl_def_id: LocalDefId, trait_ref: ty::TraitRef<'_>) {
    debug!(
        "(checking implementation) adding impl for trait '{:?}', item '{}'",
        trait_ref,
        tcx.def_path_str(impl_def_id.to_def_id())
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
    let did = Some(trait_def_id);
    let li = tcx.lang_items();

    // Disallow *all* explicit impls of `Pointee`, `DiscriminantKind`, `Sized` and `Unsize` for now.
    if did == li.pointee_trait() {
        let span = impl_header_span(tcx, impl_def_id);
        struct_span_err!(
            tcx.sess,
            span,
            E0322,
            "explicit impls for the `Pointee` trait are not permitted"
        )
        .span_label(span, "impl of 'Pointee' not allowed")
        .emit();
        return;
    }

    if did == li.discriminant_kind_trait() {
        let span = impl_header_span(tcx, impl_def_id);
        struct_span_err!(
            tcx.sess,
            span,
            E0322,
            "explicit impls for the `DiscriminantKind` trait are not permitted"
        )
        .span_label(span, "impl of 'DiscriminantKind' not allowed")
        .emit();
        return;
    }

    if did == li.sized_trait() {
        let span = impl_header_span(tcx, impl_def_id);
        struct_span_err!(
            tcx.sess,
            span,
            E0322,
            "explicit impls for the `Sized` trait are not permitted"
        )
        .span_label(span, "impl of 'Sized' not allowed")
        .emit();
        return;
    }

    if did == li.unsize_trait() {
        let span = impl_header_span(tcx, impl_def_id);
        struct_span_err!(
            tcx.sess,
            span,
            E0328,
            "explicit impls for the `Unsize` trait are not permitted"
        )
        .span_label(span, "impl of `Unsize` not allowed")
        .emit();
        return;
    }

    if tcx.features().unboxed_closures {
        // the feature gate allows all Fn traits
        return;
    }

    if let ty::trait_def::TraitSpecializationKind::AlwaysApplicable =
        tcx.trait_def(trait_def_id).specialization_kind
    {
        if !tcx.features().specialization && !tcx.features().min_specialization {
            let span = impl_header_span(tcx, impl_def_id);
            tcx.sess
                .struct_span_err(
                    span,
                    "implementing `rustc_specialization_trait` traits is unstable",
                )
                .help("add `#![feature(min_specialization)]` to the crate attributes to enable")
                .emit();
            return;
        }
    }

    let trait_name = if did == li.fn_trait() {
        "Fn"
    } else if did == li.fn_mut_trait() {
        "FnMut"
    } else if did == li.fn_once_trait() {
        "FnOnce"
    } else {
        return; // everything OK
    };

    let span = impl_header_span(tcx, impl_def_id);
    struct_span_err!(
        tcx.sess,
        span,
        E0183,
        "manual implementations of `{}` are experimental",
        trait_name
    )
    .span_label(span, format!("manual implementations of `{}` are experimental", trait_name))
    .help("add `#![feature(unboxed_closures)]` to the crate attributes to enable")
    .emit();
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

    let span = impl_header_span(tcx, impl_def_id);
    struct_span_err!(tcx.sess, span, E0715, "impls for marker traits cannot contain items").emit();
}

pub fn provide(providers: &mut Providers) {
    use self::builtin::coerce_unsized_info;
    use self::inherent_impls::{crate_inherent_impls, inherent_impls};
    use self::inherent_impls_overlap::crate_inherent_impls_overlap_check;

    *providers = Providers {
        coherent_trait,
        crate_inherent_impls,
        inherent_impls,
        crate_inherent_impls_overlap_check,
        coerce_unsized_info,
        ..*providers
    };
}

fn coherent_trait(tcx: TyCtxt<'_>, def_id: DefId) {
    // Trigger building the specialization graph for the trait. This will detect and report any
    // overlap errors.
    tcx.ensure().specialization_graph_of(def_id);

    let impls = tcx.hir().trait_impls(def_id);
    for &impl_def_id in impls {
        let trait_ref = tcx.impl_trait_ref(impl_def_id).unwrap();

        check_impl(tcx, impl_def_id, trait_ref);
        check_object_overlap(tcx, impl_def_id, trait_ref);
    }
    builtin::check_trait(tcx, def_id);
}

pub fn check_coherence(tcx: TyCtxt<'_>) {
    for &trait_def_id in tcx.all_local_trait_impls(()).keys() {
        tcx.ensure().coherent_trait(trait_def_id);
    }

    tcx.sess.time("unsafety_checking", || unsafety::check(tcx));
    tcx.sess.time("orphan_checking", || orphan::check(tcx));

    // these queries are executed for side-effects (error reporting):
    tcx.ensure().crate_inherent_impls(());
    tcx.ensure().crate_inherent_impls_overlap_check(());
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
            if !tcx.is_object_safe(component_def_id) {
                // Without the 'object_safe_for_dispatch' feature this is an error
                // which will be reported by wfcheck.  Ignore it here.
                // This is tested by `coherence-impl-trait-for-trait-object-safe.rs`.
                // With the feature enabled, the trait is not implemented automatically,
                // so this is valid.
            } else {
                let mut supertrait_def_ids = traits::supertrait_def_ids(tcx, component_def_id);
                if supertrait_def_ids.any(|d| d == trait_def_id) {
                    let span = impl_header_span(tcx, impl_def_id);
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
