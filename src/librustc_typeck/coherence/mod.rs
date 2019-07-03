// Coherence phase
//
// The job of the coherence phase of typechecking is to ensure that
// each trait has at most one implementation for each type. This is
// done by the orphan and overlap modules. Then we build up various
// mappings. That mapping code resides here.

use crate::hir::HirId;
use crate::hir::def_id::{DefId, LOCAL_CRATE};
use rustc::traits;
use rustc::ty::{self, TyCtxt, TypeFoldable};
use rustc::ty::query::Providers;
use rustc::util::common::time;

mod builtin;
mod inherent_impls;
mod inherent_impls_overlap;
mod orphan;
mod unsafety;

fn check_impl(tcx: TyCtxt<'_>, hir_id: HirId) {
    let impl_def_id = tcx.hir().local_def_id_from_hir_id(hir_id);

    // If there are no traits, then this implementation must have a
    // base type.

    if let Some(trait_ref) = tcx.impl_trait_ref(impl_def_id) {
        debug!("(checking implementation) adding impl for trait '{:?}', item '{}'",
               trait_ref,
               tcx.def_path_str(impl_def_id));

        // Skip impls where one of the self type is an error type.
        // This occurs with e.g., resolve failures (#30589).
        if trait_ref.references_error() {
            return;
        }

        enforce_trait_manually_implementable(tcx, impl_def_id, trait_ref.def_id);
        enforce_empty_impls_for_marker_traits(tcx, impl_def_id, trait_ref.def_id);
    }
}

fn enforce_trait_manually_implementable(tcx: TyCtxt<'_>, impl_def_id: DefId, trait_def_id: DefId) {
    let did = Some(trait_def_id);
    let li = tcx.lang_items();
    let span = tcx.sess.source_map().def_span(tcx.span_of_impl(impl_def_id).unwrap());

    // Disallow *all* explicit impls of `Sized` and `Unsize` for now.
    if did == li.sized_trait() {
        struct_span_err!(tcx.sess,
                         span,
                         E0322,
                         "explicit impls for the `Sized` trait are not permitted")
            .span_label(span, "impl of 'Sized' not allowed")
            .emit();
        return;
    }

    if did == li.unsize_trait() {
        struct_span_err!(tcx.sess,
                         span,
                         E0328,
                         "explicit impls for the `Unsize` trait are not permitted")
            .span_label(span, "impl of `Unsize` not allowed")
            .emit();
        return;
    }

    if tcx.features().unboxed_closures {
        // the feature gate allows all Fn traits
        return;
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
    struct_span_err!(tcx.sess,
                     span,
                     E0183,
                     "manual implementations of `{}` are experimental",
                     trait_name)
        .span_label(span, format!("manual implementations of `{}` are experimental", trait_name))
        .help("add `#![feature(unboxed_closures)]` to the crate attributes to enable")
        .emit();
}

/// We allow impls of marker traits to overlap, so they can't override impls
/// as that could make it ambiguous which associated item to use.
fn enforce_empty_impls_for_marker_traits(tcx: TyCtxt<'_>, impl_def_id: DefId, trait_def_id: DefId) {
    if !tcx.trait_def(trait_def_id).is_marker {
        return;
    }

    if tcx.associated_item_def_ids(trait_def_id).is_empty() {
        return;
    }

    let span = tcx.sess.source_map().def_span(tcx.span_of_impl(impl_def_id).unwrap());
    struct_span_err!(tcx.sess,
                     span,
                     E0715,
                     "impls for marker traits cannot contain items")
        .emit();
}

pub fn provide(providers: &mut Providers<'_>) {
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
    let impls = tcx.hir().trait_impls(def_id);
    for &impl_id in impls {
        check_impl(tcx, impl_id);
    }
    for &impl_id in impls {
        check_impl_overlap(tcx, impl_id);
    }
    builtin::check_trait(tcx, def_id);
}

pub fn check_coherence(tcx: TyCtxt<'_>) {
    for &trait_def_id in tcx.hir().krate().trait_impls.keys() {
        tcx.ensure().coherent_trait(trait_def_id);
    }

    time(tcx.sess, "unsafety checking", || unsafety::check(tcx));
    time(tcx.sess, "orphan checking", || orphan::check(tcx));

    // these queries are executed for side-effects (error reporting):
    tcx.ensure().crate_inherent_impls(LOCAL_CRATE);
    tcx.ensure().crate_inherent_impls_overlap_check(LOCAL_CRATE);
}

/// Overlap: no two impls for the same trait are implemented for the
/// same type. Likewise, no two inherent impls for a given type
/// constructor provide a method with the same name.
fn check_impl_overlap(tcx: TyCtxt<'_>, hir_id: HirId) {
    let impl_def_id = tcx.hir().local_def_id_from_hir_id(hir_id);
    let trait_ref = tcx.impl_trait_ref(impl_def_id).unwrap();
    let trait_def_id = trait_ref.def_id;

    if trait_ref.references_error() {
        debug!("coherence: skipping impl {:?} with error {:?}",
               impl_def_id, trait_ref);
        return
    }

    // Trigger building the specialization graph for the trait of this impl.
    // This will detect any overlap errors.
    tcx.specialization_graph_of(trait_def_id);

    // check for overlap with the automatic `impl Trait for Trait`
    if let ty::Dynamic(ref data, ..) = trait_ref.self_ty().sty {
        // This is something like impl Trait1 for Trait2. Illegal
        // if Trait1 is a supertrait of Trait2 or Trait2 is not object safe.

        let component_def_ids = data.iter().flat_map(|predicate| {
            match predicate.skip_binder() {
                ty::ExistentialPredicate::Trait(tr) => Some(tr.def_id),
                ty::ExistentialPredicate::AutoTrait(def_id) => Some(*def_id),
                // An associated type projection necessarily comes with
                // an additional `Trait` requirement.
                ty::ExistentialPredicate::Projection(..) => None,
            }
        });

        for component_def_id in component_def_ids {
            if !tcx.is_object_safe(component_def_id) {
                // This is an error, but it will be reported by wfcheck.  Ignore it here.
                // This is tested by `coherence-impl-trait-for-trait-object-safe.rs`.
            } else {
                let mut supertrait_def_ids =
                    traits::supertrait_def_ids(tcx, component_def_id);
                if supertrait_def_ids.any(|d| d == trait_def_id) {
                    let sp = tcx.sess.source_map().def_span(tcx.span_of_impl(impl_def_id).unwrap());
                    struct_span_err!(tcx.sess,
                                     sp,
                                     E0371,
                                     "the object type `{}` automatically implements the trait `{}`",
                                     trait_ref.self_ty(),
                                     tcx.def_path_str(trait_def_id))
                        .span_label(sp, format!("`{}` automatically implements trait `{}`",
                                                trait_ref.self_ty(),
                                                tcx.def_path_str(trait_def_id)))
                        .emit();
                }
            }
        }
    }
}
