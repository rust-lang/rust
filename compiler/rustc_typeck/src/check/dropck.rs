use crate::check::regionck::RegionCtxt;
use crate::hir;
use crate::hir::def_id::{DefId, LocalDefId};
use rustc_errors::{struct_span_err, ErrorReported};
use rustc_infer::infer::outlives::env::OutlivesEnvironment;
use rustc_infer::infer::{InferOk, RegionckMode, TyCtxtInferExt};
use rustc_infer::traits::TraitEngineExt as _;
use rustc_middle::ty::error::TypeError;
use rustc_middle::ty::relate::{Relate, RelateResult, TypeRelation};
use rustc_middle::ty::subst::{Subst, SubstsRef};
use rustc_middle::ty::{self, Predicate, Ty, TyCtxt};
use rustc_span::Span;
use rustc_trait_selection::traits::error_reporting::InferCtxtExt;
use rustc_trait_selection::traits::query::dropck_outlives::AtExt;
use rustc_trait_selection::traits::{ObligationCause, TraitEngine, TraitEngineExt};

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
    match dtor_self_type.kind() {
        ty::Adt(adt_def, self_to_impl_substs) => {
            ensure_drop_params_and_item_params_correspond(
                tcx,
                drop_impl_did.expect_local(),
                dtor_self_type,
                adt_def.did,
            )?;

            ensure_drop_predicates_are_implied_by_item_defn(
                tcx,
                dtor_predicates,
                adt_def.did.expect_local(),
                self_to_impl_substs,
            )
        }
        _ => {
            // Destructors only work on nominal types.  This was
            // already checked by coherence, but compilation may
            // not have been terminated.
            let span = tcx.def_span(drop_impl_did);
            tcx.sess.delay_span_bug(
                span,
                &format!("should have been rejected by coherence check: {}", dtor_self_type),
            );
            Err(ErrorReported)
        }
    }
}

fn ensure_drop_params_and_item_params_correspond<'tcx>(
    tcx: TyCtxt<'tcx>,
    drop_impl_did: LocalDefId,
    drop_impl_ty: Ty<'tcx>,
    self_type_did: DefId,
) -> Result<(), ErrorReported> {
    let drop_impl_hir_id = tcx.hir().local_def_id_to_hir_id(drop_impl_did);

    // check that the impl type can be made to match the trait type.

    tcx.infer_ctxt().enter(|ref infcx| {
        let impl_param_env = tcx.param_env(self_type_did);
        let tcx = infcx.tcx;
        let mut fulfillment_cx = TraitEngine::new(tcx);

        let named_type = tcx.type_of(self_type_did);

        let drop_impl_span = tcx.def_span(drop_impl_did);
        let fresh_impl_substs =
            infcx.fresh_substs_for_item(drop_impl_span, drop_impl_did.to_def_id());
        let fresh_impl_self_ty = drop_impl_ty.subst(tcx, fresh_impl_substs);

        let cause = &ObligationCause::misc(drop_impl_span, drop_impl_hir_id);
        match infcx.at(cause, impl_param_env).eq(named_type, fresh_impl_self_ty) {
            Ok(InferOk { obligations, .. }) => {
                fulfillment_cx.register_predicate_obligations(infcx, obligations);
            }
            Err(_) => {
                let item_span = tcx.def_span(self_type_did);
                let self_descr = tcx.def_kind(self_type_did).descr(self_type_did);
                struct_span_err!(
                    tcx.sess,
                    drop_impl_span,
                    E0366,
                    "`Drop` impls cannot be specialized"
                )
                .span_note(
                    item_span,
                    &format!(
                        "use the same sequence of generic type, lifetime and const parameters \
                        as the {} definition",
                        self_descr,
                    ),
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
            drop_impl_did.to_def_id(),
            &outlives_env,
            RegionckMode::default(),
        );
        Ok(())
    })
}

/// Confirms that every predicate imposed by dtor_predicates is
/// implied by assuming the predicates attached to self_type_did.
fn ensure_drop_predicates_are_implied_by_item_defn<'tcx>(
    tcx: TyCtxt<'tcx>,
    dtor_predicates: ty::GenericPredicates<'tcx>,
    self_type_did: LocalDefId,
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

    let self_type_hir_id = tcx.hir().local_def_id_to_hir_id(self_type_did);

    // We can assume the predicates attached to struct/enum definition
    // hold.
    let generic_assumptions = tcx.predicates_of(self_type_did);

    let assumptions_in_impl_context = generic_assumptions.instantiate(tcx, &self_to_impl_substs);
    let assumptions_in_impl_context = assumptions_in_impl_context.predicates;

    let self_param_env = tcx.param_env(self_type_did);

    // An earlier version of this code attempted to do this checking
    // via the traits::fulfill machinery. However, it ran into trouble
    // since the fulfill machinery merely turns outlives-predicates
    // 'a:'b and T:'b into region inference constraints. It is simpler
    // just to look for all the predicates directly.

    assert_eq!(dtor_predicates.parent, None);
    for &(predicate, predicate_sp) in dtor_predicates.predicates {
        // (We do not need to worry about deep analysis of type
        // expressions etc because the Drop impls are already forced
        // to take on a structure that is roughly an alpha-renaming of
        // the generic parameters of the item definition.)

        // This path now just checks *all* predicates via an instantiation of
        // the `SimpleEqRelation`, which simply forwards to the `relate` machinery
        // after taking care of anonymizing late bound regions.
        //
        // However, it may be more efficient in the future to batch
        // the analysis together via the fulfill (see comment above regarding
        // the usage of the fulfill machinery), rather than the
        // repeated `.iter().any(..)` calls.

        // This closure is a more robust way to check `Predicate` equality
        // than simple `==` checks (which were the previous implementation).
        // It relies on `ty::relate` for `TraitPredicate` and `ProjectionPredicate`
        // (which implement the Relate trait), while delegating on simple equality
        // for the other `Predicate`.
        // This implementation solves (Issue #59497) and (Issue #58311).
        // It is unclear to me at the moment whether the approach based on `relate`
        // could be extended easily also to the other `Predicate`.
        let predicate_matches_closure = |p: Predicate<'tcx>| {
            let mut relator: SimpleEqRelation<'tcx> = SimpleEqRelation::new(tcx, self_param_env);
            let predicate = predicate.bound_atom();
            let p = p.bound_atom();
            match (predicate.skip_binder(), p.skip_binder()) {
                (ty::PredicateAtom::Trait(a, _), ty::PredicateAtom::Trait(b, _)) => {
                    relator.relate(predicate.rebind(a), p.rebind(b)).is_ok()
                }
                (ty::PredicateAtom::Projection(a), ty::PredicateAtom::Projection(b)) => {
                    relator.relate(predicate.rebind(a), p.rebind(b)).is_ok()
                }
                _ => predicate == p,
            }
        };

        if !assumptions_in_impl_context.iter().copied().any(predicate_matches_closure) {
            let item_span = tcx.hir().span(self_type_hir_id);
            let self_descr = tcx.def_kind(self_type_did).descr(self_type_did.to_def_id());
            struct_span_err!(
                tcx.sess,
                predicate_sp,
                E0367,
                "`Drop` impl requires `{}` but the {} it is implemented for does not",
                predicate,
                self_descr,
            )
            .span_note(item_span, "the implementor must specify the same requirement")
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

// This is an implementation of the TypeRelation trait with the
// aim of simply comparing for equality (without side-effects).
// It is not intended to be used anywhere else other than here.
crate struct SimpleEqRelation<'tcx> {
    tcx: TyCtxt<'tcx>,
    param_env: ty::ParamEnv<'tcx>,
}

impl<'tcx> SimpleEqRelation<'tcx> {
    fn new(tcx: TyCtxt<'tcx>, param_env: ty::ParamEnv<'tcx>) -> SimpleEqRelation<'tcx> {
        SimpleEqRelation { tcx, param_env }
    }
}

impl TypeRelation<'tcx> for SimpleEqRelation<'tcx> {
    fn tcx(&self) -> TyCtxt<'tcx> {
        self.tcx
    }

    fn param_env(&self) -> ty::ParamEnv<'tcx> {
        self.param_env
    }

    fn tag(&self) -> &'static str {
        "dropck::SimpleEqRelation"
    }

    fn a_is_expected(&self) -> bool {
        true
    }

    fn relate_with_variance<T: Relate<'tcx>>(
        &mut self,
        _: ty::Variance,
        a: T,
        b: T,
    ) -> RelateResult<'tcx, T> {
        // Here we ignore variance because we require drop impl's types
        // to be *exactly* the same as to the ones in the struct definition.
        self.relate(a, b)
    }

    fn tys(&mut self, a: Ty<'tcx>, b: Ty<'tcx>) -> RelateResult<'tcx, Ty<'tcx>> {
        debug!("SimpleEqRelation::tys(a={:?}, b={:?})", a, b);
        ty::relate::super_relate_tys(self, a, b)
    }

    fn regions(
        &mut self,
        a: ty::Region<'tcx>,
        b: ty::Region<'tcx>,
    ) -> RelateResult<'tcx, ty::Region<'tcx>> {
        debug!("SimpleEqRelation::regions(a={:?}, b={:?})", a, b);

        // We can just equate the regions because LBRs have been
        // already anonymized.
        if a == b {
            Ok(a)
        } else {
            // I'm not sure is this `TypeError` is the right one, but
            // it should not matter as it won't be checked (the dropck
            // will emit its own, more informative and higher-level errors
            // in case anything goes wrong).
            Err(TypeError::RegionsPlaceholderMismatch)
        }
    }

    fn consts(
        &mut self,
        a: &'tcx ty::Const<'tcx>,
        b: &'tcx ty::Const<'tcx>,
    ) -> RelateResult<'tcx, &'tcx ty::Const<'tcx>> {
        debug!("SimpleEqRelation::consts(a={:?}, b={:?})", a, b);
        ty::relate::super_relate_consts(self, a, b)
    }

    fn binders<T>(
        &mut self,
        a: ty::Binder<T>,
        b: ty::Binder<T>,
    ) -> RelateResult<'tcx, ty::Binder<T>>
    where
        T: Relate<'tcx>,
    {
        debug!("SimpleEqRelation::binders({:?}: {:?}", a, b);

        // Anonymizing the LBRs is necessary to solve (Issue #59497).
        // After we do so, it should be totally fine to skip the binders.
        let anon_a = self.tcx.anonymize_late_bound_regions(&a);
        let anon_b = self.tcx.anonymize_late_bound_regions(&b);
        self.relate(anon_a.skip_binder(), anon_b.skip_binder())?;

        Ok(a)
    }
}
