// FIXME(@lcnr): Move this module out of `rustc_hir_analysis`.
//
// We don't do any drop checking during hir typeck.
use rustc_data_structures::fx::FxHashSet;
use rustc_errors::{struct_span_err, ErrorGuaranteed};
use rustc_infer::infer::outlives::env::OutlivesEnvironment;
use rustc_infer::infer::{RegionResolutionError, TyCtxtInferExt};
use rustc_middle::ty::subst::SubstsRef;
use rustc_middle::ty::util::CheckRegions;
use rustc_middle::ty::{self, TyCtxt};
use rustc_trait_selection::traits::{self, ObligationCtxt};

use crate::errors;
use crate::hir::def_id::{DefId, LocalDefId};

/// This function confirms that the `Drop` implementation identified by
/// `drop_impl_did` is not any more specialized than the type it is
/// attached to (Issue #8142).
///
/// This means:
///
/// 1. The self type must be nominal (this is already checked during
///    coherence),
///
/// 2. The generic region/type parameters of the impl's self type must
///    all be parameters of the Drop impl itself (i.e., no
///    specialization like `impl Drop for Foo<i32>`), and,
///
/// 3. Any bounds on the generic parameters must be reflected in the
///    struct/enum definition for the nominal type itself (i.e.
///    cannot do `struct S<T>; impl<T:Clone> Drop for S<T> { ... }`).
///
pub fn check_drop_impl(tcx: TyCtxt<'_>, drop_impl_did: DefId) -> Result<(), ErrorGuaranteed> {
    match tcx.impl_polarity(drop_impl_did) {
        ty::ImplPolarity::Positive => {}
        ty::ImplPolarity::Negative => {
            return Err(tcx.sess.emit_err(errors::DropImplPolarity::Negative {
                span: tcx.def_span(drop_impl_did),
            }));
        }
        ty::ImplPolarity::Reservation => {
            return Err(tcx.sess.emit_err(errors::DropImplPolarity::Reservation {
                span: tcx.def_span(drop_impl_did),
            }));
        }
    }
    let dtor_self_type = tcx.type_of(drop_impl_did).subst_identity();
    match dtor_self_type.kind() {
        ty::Adt(adt_def, adt_to_impl_substs) => {
            ensure_drop_params_and_item_params_correspond(
                tcx,
                drop_impl_did.expect_local(),
                adt_def.did(),
                adt_to_impl_substs,
            )?;

            ensure_drop_predicates_are_implied_by_item_defn(
                tcx,
                drop_impl_did.expect_local(),
                adt_def.did().expect_local(),
                adt_to_impl_substs,
            )
        }
        _ => {
            // Destructors only work on nominal types. This was
            // already checked by coherence, but compilation may
            // not have been terminated.
            let span = tcx.def_span(drop_impl_did);
            let reported = tcx.sess.delay_span_bug(
                span,
                format!("should have been rejected by coherence check: {dtor_self_type}"),
            );
            Err(reported)
        }
    }
}

fn ensure_drop_params_and_item_params_correspond<'tcx>(
    tcx: TyCtxt<'tcx>,
    drop_impl_did: LocalDefId,
    self_type_did: DefId,
    adt_to_impl_substs: SubstsRef<'tcx>,
) -> Result<(), ErrorGuaranteed> {
    let Err(arg) = tcx.uses_unique_generic_params(adt_to_impl_substs, CheckRegions::OnlyEarlyBound) else {
        return Ok(())
    };

    let drop_impl_span = tcx.def_span(drop_impl_did);
    let item_span = tcx.def_span(self_type_did);
    let self_descr = tcx.def_descr(self_type_did);
    let mut err =
        struct_span_err!(tcx.sess, drop_impl_span, E0366, "`Drop` impls cannot be specialized");
    match arg {
        ty::util::NotUniqueParam::DuplicateParam(arg) => {
            err.note(format!("`{arg}` is mentioned multiple times"))
        }
        ty::util::NotUniqueParam::NotParam(arg) => {
            err.note(format!("`{arg}` is not a generic parameter"))
        }
    };
    err.span_note(
        item_span,
        format!(
            "use the same sequence of generic lifetime, type and const parameters \
                     as the {self_descr} definition",
        ),
    );
    Err(err.emit())
}

/// Confirms that every predicate imposed by dtor_predicates is
/// implied by assuming the predicates attached to self_type_did.
fn ensure_drop_predicates_are_implied_by_item_defn<'tcx>(
    tcx: TyCtxt<'tcx>,
    drop_impl_def_id: LocalDefId,
    adt_def_id: LocalDefId,
    adt_to_impl_substs: SubstsRef<'tcx>,
) -> Result<(), ErrorGuaranteed> {
    let infcx = tcx.infer_ctxt().build();
    let ocx = ObligationCtxt::new(&infcx);

    // Take the param-env of the adt and substitute the substs that show up in
    // the implementation's self type. This gives us the assumptions that the
    // self ty of the implementation is allowed to know just from it being a
    // well-formed adt, since that's all we're allowed to assume while proving
    // the Drop implementation is not specialized.
    //
    // We don't need to normalize this param-env or anything, since we're only
    // substituting it with free params, so no additional param-env normalization
    // can occur on top of what has been done in the param_env query itself.
    let param_env = ty::EarlyBinder::bind(tcx.param_env(adt_def_id))
        .subst(tcx, adt_to_impl_substs)
        .with_constness(tcx.constness(drop_impl_def_id));

    for (pred, span) in tcx.predicates_of(drop_impl_def_id).instantiate_identity(tcx) {
        let normalize_cause = traits::ObligationCause::misc(span, adt_def_id);
        let pred = ocx.normalize(&normalize_cause, param_env, pred);
        let cause = traits::ObligationCause::new(span, adt_def_id, traits::DropImpl);
        ocx.register_obligation(traits::Obligation::new(tcx, cause, param_env, pred));
    }

    // All of the custom error reporting logic is to preserve parity with the old
    // error messages.
    //
    // They can probably get removed with better treatment of the new `DropImpl`
    // obligation cause code, and perhaps some custom logic in `report_region_errors`.

    let errors = ocx.select_all_or_error();
    if !errors.is_empty() {
        let mut guar = None;
        let mut root_predicates = FxHashSet::default();
        for error in errors {
            let root_predicate = error.root_obligation.predicate;
            if root_predicates.insert(root_predicate) {
                let item_span = tcx.def_span(adt_def_id);
                let self_descr = tcx.def_descr(adt_def_id.to_def_id());
                guar = Some(
                    struct_span_err!(
                        tcx.sess,
                        error.root_obligation.cause.span,
                        E0367,
                        "`Drop` impl requires `{root_predicate}` \
                        but the {self_descr} it is implemented for does not",
                    )
                    .span_note(item_span, "the implementor must specify the same requirement")
                    .emit(),
                );
            }
        }
        return Err(guar.unwrap());
    }

    let errors = ocx.infcx.resolve_regions(&OutlivesEnvironment::new(param_env));
    if !errors.is_empty() {
        let mut guar = None;
        for error in errors {
            let item_span = tcx.def_span(adt_def_id);
            let self_descr = tcx.def_descr(adt_def_id.to_def_id());
            let outlives = match error {
                RegionResolutionError::ConcreteFailure(_, a, b) => format!("{b}: {a}"),
                RegionResolutionError::GenericBoundFailure(_, generic, r) => {
                    format!("{generic}: {r}")
                }
                RegionResolutionError::SubSupConflict(_, _, _, a, _, b, _) => format!("{b}: {a}"),
                RegionResolutionError::UpperBoundUniverseConflict(a, _, _, _, b) => {
                    format!("{b}: {a}", a = ty::Region::new_var(tcx, a))
                }
            };
            guar = Some(
                struct_span_err!(
                    tcx.sess,
                    error.origin().span(),
                    E0367,
                    "`Drop` impl requires `{outlives}` \
                    but the {self_descr} it is implemented for does not",
                )
                .span_note(item_span, "the implementor must specify the same requirement")
                .emit(),
            );
        }
        return Err(guar.unwrap());
    }

    Ok(())
}
