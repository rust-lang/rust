use std::iter;

use crate::check::regionck::RegionCtxt;
use crate::check::Inherited;
use crate::hir;
use rustc_errors::{struct_span_err, ErrorGuaranteed};
use rustc_hir::def_id::{DefId, LocalDefId};
use rustc_hir::CRATE_HIR_ID;
use rustc_infer::infer::outlives::env::OutlivesEnvironment;
use rustc_infer::infer::RegionckMode;
use rustc_infer::infer::TyCtxtInferExt;
use rustc_infer::traits::Obligation;
use rustc_middle::ty::subst::{Subst, SubstsRef};
use rustc_middle::ty::util::IgnoreRegions;
use rustc_middle::ty::{self, Ty, TyCtxt};
use rustc_span::Span;
use rustc_trait_selection::traits::query::dropck_outlives::AtExt;
use rustc_trait_selection::traits::ObligationCause;

/// This function confirms that the `Drop` implementation identified by
/// `drop_impl_def_id` is not any more specialized than the type it is
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
pub fn check_drop_impl(tcx: TyCtxt<'_>, drop_impl_def_id: DefId) -> Result<(), ErrorGuaranteed> {
    let drop_impl_def_id = drop_impl_def_id.expect_local();
    let dtor_self_type = tcx.type_of(drop_impl_def_id);
    match dtor_self_type.kind() {
        ty::Adt(adt_def, self_to_impl_substs) => {
            ensure_drop_params_and_item_params_correspond(
                tcx,
                drop_impl_def_id,
                adt_def.did(),
                self_to_impl_substs,
            )?;

            drop_bounds_implied_by_item_definition(
                tcx,
                drop_impl_def_id,
                adt_def.did().expect_local(),
                self_to_impl_substs,
            )
        }
        _ => {
            // Destructors only work on nominal types. This was
            // already checked by coherence, but compilation may
            // not have been terminated.
            let span = tcx.def_span(drop_impl_def_id);
            let reported = tcx.sess.delay_span_bug(
                span,
                &format!("should have been rejected by coherence check: {dtor_self_type}"),
            );
            Err(reported)
        }
    }
}

fn ensure_drop_params_and_item_params_correspond<'tcx>(
    tcx: TyCtxt<'tcx>,
    drop_impl_def_id: LocalDefId,
    self_type_did: DefId,
    drop_impl_substs: SubstsRef<'tcx>,
) -> Result<(), ErrorGuaranteed> {
    let Err(arg) = tcx.uses_unique_generic_params(drop_impl_substs, IgnoreRegions::No) else {
        return Ok(())
    };

    let drop_impl_span = tcx.def_span(drop_impl_def_id);
    let item_span = tcx.def_span(self_type_did);
    let self_descr = tcx.def_kind(self_type_did).descr(self_type_did);
    let mut err =
        struct_span_err!(tcx.sess, drop_impl_span, E0366, "`Drop` impls cannot be specialized");
    match arg {
        ty::util::NotUniqueParam::DuplicateParam(arg) => {
            err.note(&format!("`{arg}` is mentioned multiple times"))
        }
        ty::util::NotUniqueParam::NotParam(arg) => {
            err.note(&format!("`{arg}` is not a generic parameter"))
        }
    };
    err.span_note(
        item_span,
        &format!(
            "use the same sequence of generic lifetime, type and const parameters \
                     as the {self_descr} definition",
        ),
    );
    Err(err.emit())
}

/// Confirms that the bounds of the `Drop` impl are implied
/// by the bounds of the struct definition.
fn drop_bounds_implied_by_item_definition<'tcx>(
    tcx: TyCtxt<'tcx>,
    drop_impl_def_id: LocalDefId,
    self_ty_def_id: LocalDefId,
    impl_substs: SubstsRef<'tcx>,
) -> Result<(), ErrorGuaranteed> {
    let self_ty_hir_id = tcx.hir().local_def_id_to_hir_id(self_ty_def_id);
    tcx.infer_ctxt().enter(|infcx| {
        let inh = Inherited::new(infcx, self_ty_def_id);
        let infcx = &inh.infcx;

        // The bounds of the `Drop` impl have to hold given the bounds
        // of the type definition.
        let param_env = tcx.bound_param_env(self_ty_def_id.to_def_id()).subst(tcx, impl_substs);

        // We now simply emit obligations for each predicate of the impl.
        let dtor_predicates = tcx.predicates_of(drop_impl_def_id).instantiate_identity(tcx);
        for (pred, span) in iter::zip(dtor_predicates.predicates, dtor_predicates.spans) {
            let cause = ObligationCause::misc(span, self_ty_hir_id);
            let obligation = Obligation::new(cause, param_env, pred);
            inh.register_predicate(obligation);
        }

        let errors = inh.fulfillment_cx.borrow_mut().select_all_or_error(&infcx);
        let mut result = Ok(());
        for error in errors {
            let item_span = tcx.def_span(self_ty_def_id);
            let self_descr = tcx.def_kind(self_ty_def_id).descr(self_ty_def_id.to_def_id());
            let reported = struct_span_err!(
                tcx.sess,
                error.obligation.cause.span,
                E0367,
                "`Drop` impl requires `{}` but the {self_descr} it is implemented for does not",
                error.obligation.predicate,
            )
            .span_note(item_span, "the implementor must specify the same requirement")
            .emit();
            result = Err(reported);
        }
        result?;

        let mut outlives_env = OutlivesEnvironment::new(param_env);
        outlives_env.save_implied_bounds(CRATE_HIR_ID);

        infcx.process_registered_region_obligations(
            outlives_env.region_bound_pairs_map(),
            Some(tcx.lifetimes.re_root_empty),
            param_env,
        );

        // This `DefId` isn't actually used.
        let errors = infcx.resolve_regions(
            drop_impl_def_id.to_def_id(),
            &outlives_env,
            RegionckMode::default(),
        );
        let mut result = Ok(());
        for error in errors {
            let item_span = tcx.def_span(self_ty_def_id);
            let self_descr = tcx.def_kind(self_ty_def_id).descr(self_ty_def_id.to_def_id());
            let reported = struct_span_err!(
                tcx.sess,
                error.span(),
                E0367,
                "`Drop` impl requires `{}` but the {self_descr} it is implemented for does not",
                error.as_bound()
            )
            .span_note(item_span, "the implementor must specify the same requirement")
            .emit();
            result = Err(reported);
        }
        result
    })
}

/// This function is not only checking that the dropck obligations are met for
/// the given type, but it's also currently preventing non-regular recursion in
/// types from causing stack overflows (dropck_no_diverge_on_nonregular_*.rs).
pub(crate) fn check_drop_obligations<'a, 'tcx>(
    rcx: &mut RegionCtxt<'a, 'tcx>,
    ty: Ty<'tcx>,
    span: Span,
    body_id: hir::HirId,
) {
    debug!("check_drop_obligations typ: {:?}", ty);

    let cause = &ObligationCause::misc(span, body_id);
    let infer_ok = rcx.infcx.at(cause, rcx.fcx.param_env).dropck_outlives(ty);
    debug!("dropck_outlives = {:#?}", infer_ok);
    rcx.fcx.register_infer_ok_obligations(infer_ok);
}
