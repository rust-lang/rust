use crate::check::regionck::RegionCtxt;

use crate::hir;
use crate::hir::def_id::DefId;
use rustc::infer::outlives::env::OutlivesEnvironment;
use rustc::infer::{InferOk, SuppressRegionErrors};
use rustc::middle::region;
use rustc::traits::{ObligationCause, TraitEngine, TraitEngineExt};
use rustc::ty::subst::{Subst, SubstsRef};
use rustc::ty::{self, Ty, TyCtxt};
use crate::util::common::ErrorReported;

use syntax_pos::Span;

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
pub fn check_drop_impl(tcx: TyCtxt<'_>, drop_impl_did: DefId) -> Result<(), ErrorReported> {
    let dtor_self_type = tcx.type_of(drop_impl_did);
    let dtor_predicates = tcx.predicates_of(drop_impl_did);
    match dtor_self_type.kind {
        ty::Adt(adt_def, self_to_impl_substs) => {
            ensure_drop_params_and_item_params_correspond(
                tcx,
                drop_impl_did,
                dtor_self_type,
                adt_def.did,
            )?;

            ensure_drop_predicates_are_implied_by_item_defn(
                tcx,
                drop_impl_did,
                &dtor_predicates,
                adt_def.did,
                self_to_impl_substs,
            )
        }
        _ => {
            // Destructors only work on nominal types.  This was
            // already checked by coherence, but compilation may
            // not have been terminated.
            let span = tcx.def_span(drop_impl_did);
            tcx.sess.delay_span_bug(span,
                &format!("should have been rejected by coherence check: {}", dtor_self_type));
            Err(ErrorReported)
        }
    }
}

fn ensure_drop_params_and_item_params_correspond<'tcx>(
    tcx: TyCtxt<'tcx>,
    drop_impl_did: DefId,
    drop_impl_ty: Ty<'tcx>,
    self_type_did: DefId,
) -> Result<(), ErrorReported> {
    let drop_impl_hir_id = tcx.hir().as_local_hir_id(drop_impl_did).unwrap();

    // check that the impl type can be made to match the trait type.

    tcx.infer_ctxt().enter(|ref infcx| {
        let impl_param_env = tcx.param_env(self_type_did);
        let tcx = infcx.tcx;
        let mut fulfillment_cx = TraitEngine::new(tcx);

        let named_type = tcx.type_of(self_type_did);

        let drop_impl_span = tcx.def_span(drop_impl_did);
        let fresh_impl_substs = infcx.fresh_substs_for_item(drop_impl_span, drop_impl_did);
        let fresh_impl_self_ty = drop_impl_ty.subst(tcx, fresh_impl_substs);

        let cause = &ObligationCause::misc(drop_impl_span, drop_impl_hir_id);
        match infcx
            .at(cause, impl_param_env)
            .eq(named_type, fresh_impl_self_ty)
        {
            Ok(InferOk { obligations, .. }) => {
                fulfillment_cx.register_predicate_obligations(infcx, obligations);
            }
            Err(_) => {
                let item_span = tcx.def_span(self_type_did);
                struct_span_err!(
                    tcx.sess,
                    drop_impl_span,
                    E0366,
                    "Implementations of Drop cannot be specialized"
                ).span_note(
                    item_span,
                    "Use same sequence of generic type and region \
                     parameters that is on the struct/enum definition",
                )
                    .emit();
                return Err(ErrorReported);
            }
        }

        if let Err(ref errors) = fulfillment_cx.select_all_or_error(&infcx) {
            // this could be reached when we get lazy normalization
            infcx.report_fulfillment_errors(errors, None, false);
            return Err(ErrorReported);
        }

        let region_scope_tree = region::ScopeTree::default();

        // NB. It seems a bit... suspicious to use an empty param-env
        // here. The correct thing, I imagine, would be
        // `OutlivesEnvironment::new(impl_param_env)`, which would
        // allow region solving to take any `a: 'b` relations on the
        // impl into account. But I could not create a test case where
        // it did the wrong thing, so I chose to preserve existing
        // behavior, since it ought to be simply more
        // conservative. -nmatsakis
        let outlives_env = OutlivesEnvironment::new(ty::ParamEnv::empty());

        infcx.resolve_regions_and_report_errors(
            drop_impl_did,
            &region_scope_tree,
            &outlives_env,
            SuppressRegionErrors::default(),
        );
        Ok(())
    })
}

/// Confirms that every predicate imposed by dtor_predicates is
/// implied by assuming the predicates attached to self_type_did.
fn ensure_drop_predicates_are_implied_by_item_defn<'tcx>(
    tcx: TyCtxt<'tcx>,
    drop_impl_did: DefId,
    dtor_predicates: &ty::GenericPredicates<'tcx>,
    self_type_did: DefId,
    self_to_impl_substs: SubstsRef<'tcx>,
) -> Result<(), ErrorReported> {
    let mut result = Ok(());

    // Here is an example, analogous to that from
    // `compare_impl_method`.
    //
    // Consider a struct type:
    //
    //     struct Type<'c, 'b:'c, 'a> {
    //         x: &'a Contents            // (contents are irrelevant;
    //         y: &'c Cell<&'b Contents>, //  only the bounds matter for our purposes.)
    //     }
    //
    // and a Drop impl:
    //
    //     impl<'z, 'y:'z, 'x:'y> Drop for P<'z, 'y, 'x> {
    //         fn drop(&mut self) { self.y.set(self.x); } // (only legal if 'x: 'y)
    //     }
    //
    // We start out with self_to_impl_substs, that maps the generic
    // parameters of Type to that of the Drop impl.
    //
    //     self_to_impl_substs = {'c => 'z, 'b => 'y, 'a => 'x}
    //
    // Applying this to the predicates (i.e., assumptions) provided by the item
    // definition yields the instantiated assumptions:
    //
    //     ['y : 'z]
    //
    // We then check all of the predicates of the Drop impl:
    //
    //     ['y:'z, 'x:'y]
    //
    // and ensure each is in the list of instantiated
    // assumptions. Here, `'y:'z` is present, but `'x:'y` is
    // absent. So we report an error that the Drop impl injected a
    // predicate that is not present on the struct definition.

    let self_type_hir_id = tcx.hir().as_local_hir_id(self_type_did).unwrap();

    let drop_impl_span = tcx.def_span(drop_impl_did);

    // We can assume the predicates attached to struct/enum definition
    // hold.
    let generic_assumptions = tcx.predicates_of(self_type_did);

    let assumptions_in_impl_context = generic_assumptions.instantiate(tcx, &self_to_impl_substs);
    let assumptions_in_impl_context = assumptions_in_impl_context.predicates;

    // An earlier version of this code attempted to do this checking
    // via the traits::fulfill machinery. However, it ran into trouble
    // since the fulfill machinery merely turns outlives-predicates
    // 'a:'b and T:'b into region inference constraints. It is simpler
    // just to look for all the predicates directly.

    assert_eq!(dtor_predicates.parent, None);
    for (predicate, _) in &dtor_predicates.predicates {
        // (We do not need to worry about deep analysis of type
        // expressions etc because the Drop impls are already forced
        // to take on a structure that is roughly an alpha-renaming of
        // the generic parameters of the item definition.)

        // This path now just checks *all* predicates via the direct
        // lookup, rather than using fulfill machinery.
        //
        // However, it may be more efficient in the future to batch
        // the analysis together via the fulfill , rather than the
        // repeated `contains` calls.

        if !assumptions_in_impl_context.contains(&predicate) {
            let item_span = tcx.hir().span(self_type_hir_id);
            struct_span_err!(
                tcx.sess,
                drop_impl_span,
                E0367,
                "The requirement `{}` is added only by the Drop impl.",
                predicate
            ).span_note(
                item_span,
                "The same requirement must be part of \
                 the struct/enum definition",
            )
                .emit();
            result = Err(ErrorReported);
        }
    }

    result
}

/// This function is not only checking that the dropck obligations are met for
/// the given type, but it's also currently preventing non-regular recursion in
/// types from causing stack overflows (dropck_no_diverge_on_nonregular_*.rs).
crate fn check_drop_obligations<'a, 'tcx>(
    rcx: &mut RegionCtxt<'a, 'tcx>,
    ty: Ty<'tcx>,
    span: Span,
    body_id: hir::HirId,
) -> Result<(), ErrorReported> {
    debug!("check_drop_obligations typ: {:?}", ty);

    let cause = &ObligationCause::misc(span, body_id);
    let infer_ok = rcx.infcx.at(cause, rcx.fcx.param_env).dropck_outlives(ty);
    debug!("dropck_outlives = {:#?}", infer_ok);
    rcx.fcx.register_infer_ok_obligations(infer_ok);

    Ok(())
}
