// Coherence phase
//
// The job of the coherence phase of typechecking is to ensure that
// each trait has at most one implementation for each type. This is
// done by the orphan and overlap modules. Then we build up various
// mappings. That mapping code resides here.

use rustc_errors::codes::*;
use rustc_errors::struct_span_code_err;
use rustc_hir::LangItem;
use rustc_hir::def_id::{DefId, LocalDefId};
use rustc_middle::query::Providers;
use rustc_middle::ty::{self, TyCtxt, TypeVisitableExt, elaborate};
use rustc_session::parse::feature_err;
use rustc_span::{ErrorGuaranteed, sym};
use tracing::debug;

use crate::check::always_applicable;
use crate::errors;

mod builtin;
mod inherent_impls;
mod inherent_impls_overlap;
mod orphan;
mod unsafety;

fn check_impl<'tcx>(
    tcx: TyCtxt<'tcx>,
    impl_def_id: LocalDefId,
    trait_ref: ty::TraitRef<'tcx>,
    trait_def: &'tcx ty::TraitDef,
    polarity: ty::ImplPolarity,
) -> Result<(), ErrorGuaranteed> {
    debug!(
        "(checking implementation) adding impl for trait '{:?}', item '{}'",
        trait_ref,
        tcx.def_path_str(impl_def_id)
    );

    // Skip impls where one of the self type is an error type.
    // This occurs with e.g., resolve failures (#30589).
    if trait_ref.references_error() {
        return Ok(());
    }

    enforce_trait_manually_implementable(tcx, impl_def_id, trait_ref.def_id, trait_def)
        .and(enforce_empty_impls_for_marker_traits(tcx, impl_def_id, trait_ref.def_id, trait_def))
        .and(always_applicable::check_negative_auto_trait_impl(
            tcx,
            impl_def_id,
            trait_ref,
            polarity,
        ))
}

fn enforce_trait_manually_implementable(
    tcx: TyCtxt<'_>,
    impl_def_id: LocalDefId,
    trait_def_id: DefId,
    trait_def: &ty::TraitDef,
) -> Result<(), ErrorGuaranteed> {
    let impl_header_span = tcx.def_span(impl_def_id);

    if tcx.is_lang_item(trait_def_id, LangItem::Freeze) && !tcx.features().freeze_impls() {
        feature_err(
            &tcx.sess,
            sym::freeze_impls,
            impl_header_span,
            "explicit impls for the `Freeze` trait are not permitted",
        )
        .with_span_label(impl_header_span, format!("impl of `Freeze` not allowed"))
        .emit();
    }

    // Disallow *all* explicit impls of traits marked `#[rustc_deny_explicit_impl]`
    if trait_def.deny_explicit_impl {
        let trait_name = tcx.item_name(trait_def_id);
        let mut err = struct_span_code_err!(
            tcx.dcx(),
            impl_header_span,
            E0322,
            "explicit impls for the `{trait_name}` trait are not permitted"
        );
        err.span_label(impl_header_span, format!("impl of `{trait_name}` not allowed"));

        // Maintain explicit error code for `Unsize`, since it has a useful
        // explanation about using `CoerceUnsized` instead.
        if tcx.is_lang_item(trait_def_id, LangItem::Unsize) {
            err.code(E0328);
        }

        return Err(err.emit());
    }

    if let ty::trait_def::TraitSpecializationKind::AlwaysApplicable = trait_def.specialization_kind
    {
        if !tcx.features().specialization()
            && !tcx.features().min_specialization()
            && !impl_header_span.allows_unstable(sym::specialization)
            && !impl_header_span.allows_unstable(sym::min_specialization)
        {
            return Err(tcx.dcx().emit_err(errors::SpecializationTrait { span: impl_header_span }));
        }
    }
    Ok(())
}

/// We allow impls of marker traits to overlap, so they can't override impls
/// as that could make it ambiguous which associated item to use.
fn enforce_empty_impls_for_marker_traits(
    tcx: TyCtxt<'_>,
    impl_def_id: LocalDefId,
    trait_def_id: DefId,
    trait_def: &ty::TraitDef,
) -> Result<(), ErrorGuaranteed> {
    if !trait_def.is_marker {
        return Ok(());
    }

    if tcx.associated_item_def_ids(trait_def_id).is_empty() {
        return Ok(());
    }

    Err(struct_span_code_err!(
        tcx.dcx(),
        tcx.def_span(impl_def_id),
        E0715,
        "impls for marker traits cannot contain items"
    )
    .emit())
}

pub(crate) fn provide(providers: &mut Providers) {
    use self::builtin::coerce_unsized_info;
    use self::inherent_impls::{
        crate_incoherent_impls, crate_inherent_impls, crate_inherent_impls_validity_check,
        inherent_impls,
    };
    use self::inherent_impls_overlap::crate_inherent_impls_overlap_check;
    use self::orphan::orphan_check_impl;

    *providers = Providers {
        coherent_trait,
        crate_inherent_impls,
        crate_incoherent_impls,
        inherent_impls,
        crate_inherent_impls_validity_check,
        crate_inherent_impls_overlap_check,
        coerce_unsized_info,
        orphan_check_impl,
        ..*providers
    };
}

fn coherent_trait(tcx: TyCtxt<'_>, def_id: DefId) -> Result<(), ErrorGuaranteed> {
    let impls = tcx.local_trait_impls(def_id);
    // If there are no impls for the trait, then "all impls" are trivially coherent and we won't check anything
    // anyway. Thus we bail out even before the specialization graph, avoiding the dep_graph edge.
    if impls.is_empty() {
        return Ok(());
    }
    // Trigger building the specialization graph for the trait. This will detect and report any
    // overlap errors.
    let mut res = tcx.ensure_ok().specialization_graph_of(def_id);

    for &impl_def_id in impls {
        let impl_header = tcx.impl_trait_header(impl_def_id).unwrap();
        let trait_ref = impl_header.trait_ref.instantiate_identity();
        let trait_def = tcx.trait_def(trait_ref.def_id);

        res = res
            .and(check_impl(tcx, impl_def_id, trait_ref, trait_def, impl_header.polarity))
            .and(check_object_overlap(tcx, impl_def_id, trait_ref))
            .and(unsafety::check_item(tcx, impl_def_id, impl_header, trait_def))
            .and(tcx.ensure_ok().orphan_check_impl(impl_def_id))
            .and(builtin::check_trait(tcx, def_id, impl_def_id, impl_header));
    }

    res
}

/// Checks whether an impl overlaps with the automatic `impl Trait for dyn Trait`.
fn check_object_overlap<'tcx>(
    tcx: TyCtxt<'tcx>,
    impl_def_id: LocalDefId,
    trait_ref: ty::TraitRef<'tcx>,
) -> Result<(), ErrorGuaranteed> {
    let trait_def_id = trait_ref.def_id;

    if trait_ref.references_error() {
        debug!("coherence: skipping impl {:?} with error {:?}", impl_def_id, trait_ref);
        return Ok(());
    }

    // check for overlap with the automatic `impl Trait for dyn Trait`
    if let ty::Dynamic(data, ..) = trait_ref.self_ty().kind() {
        // This is something like `impl Trait1 for Trait2`. Illegal if
        // Trait1 is a supertrait of Trait2 or Trait2 is not dyn compatible.

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
            if !tcx.is_dyn_compatible(component_def_id) {
                // This is a WF error tested by `coherence-impl-trait-for-trait-dyn-compatible.rs`.
            } else {
                let mut supertrait_def_ids = elaborate::supertrait_def_ids(tcx, component_def_id);
                if supertrait_def_ids
                    .any(|d| d == trait_def_id && tcx.trait_def(d).implement_via_object)
                {
                    let span = tcx.def_span(impl_def_id);
                    return Err(struct_span_code_err!(
                        tcx.dcx(),
                        span,
                        E0371,
                        "the object type `{}` automatically implements the trait `{}`",
                        trait_ref.self_ty(),
                        tcx.def_path_str(trait_def_id)
                    )
                    .with_span_label(
                        span,
                        format!(
                            "`{}` automatically implements trait `{}`",
                            trait_ref.self_ty(),
                            tcx.def_path_str(trait_def_id)
                        ),
                    )
                    .emit());
                }
            }
        }
    }
    Ok(())
}
