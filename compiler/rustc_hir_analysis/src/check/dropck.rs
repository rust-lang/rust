// FIXME(@lcnr): Move this module out of `rustc_hir_analysis`.
//
// We don't do any drop checking during hir typeck.
use crate::hir::def_id::{DefId, LocalDefId};
use rustc_errors::{struct_span_err, ErrorGuaranteed};
use rustc_middle::ty::error::TypeError;
use rustc_middle::ty::relate::{Relate, RelateResult, TypeRelation};
use rustc_middle::ty::subst::SubstsRef;
use rustc_middle::ty::util::IgnoreRegions;
use rustc_middle::ty::{self, Predicate, Ty, TyCtxt};

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
    let dtor_self_type = tcx.type_of(drop_impl_did).subst_identity();
    let dtor_predicates = tcx.predicates_of(drop_impl_did);
    match dtor_self_type.kind() {
        ty::Adt(adt_def, self_to_impl_substs) => {
            ensure_drop_params_and_item_params_correspond(
                tcx,
                drop_impl_did.expect_local(),
                adt_def.did(),
                self_to_impl_substs,
            )?;

            ensure_drop_predicates_are_implied_by_item_defn(
                tcx,
                dtor_predicates,
                adt_def.did().expect_local(),
                self_to_impl_substs,
            )
        }
        _ => {
            // Destructors only work on nominal types. This was
            // already checked by coherence, but compilation may
            // not have been terminated.
            let span = tcx.def_span(drop_impl_did);
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
    drop_impl_did: LocalDefId,
    self_type_did: DefId,
    drop_impl_substs: SubstsRef<'tcx>,
) -> Result<(), ErrorGuaranteed> {
    let Err(arg) = tcx.uses_unique_generic_params(drop_impl_substs, IgnoreRegions::No) else {
        return Ok(())
    };

    let drop_impl_span = tcx.def_span(drop_impl_did);
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

/// Confirms that every predicate imposed by dtor_predicates is
/// implied by assuming the predicates attached to self_type_did.
fn ensure_drop_predicates_are_implied_by_item_defn<'tcx>(
    tcx: TyCtxt<'tcx>,
    dtor_predicates: ty::GenericPredicates<'tcx>,
    self_type_did: LocalDefId,
    self_to_impl_substs: SubstsRef<'tcx>,
) -> Result<(), ErrorGuaranteed> {
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

    // We can assume the predicates attached to struct/enum definition
    // hold.
    let generic_assumptions = tcx.predicates_of(self_type_did);

    let assumptions_in_impl_context = generic_assumptions.instantiate(tcx, &self_to_impl_substs);
    let assumptions_in_impl_context = assumptions_in_impl_context.predicates;

    debug!(?assumptions_in_impl_context, ?dtor_predicates.predicates);

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
        // It relies on `ty::relate` for `TraitPredicate`, `ProjectionPredicate`,
        // `ConstEvaluatable` and `TypeOutlives` (which implement the Relate trait),
        // while delegating on simple equality for the other `Predicate`.
        // This implementation solves (Issue #59497) and (Issue #58311).
        // It is unclear to me at the moment whether the approach based on `relate`
        // could be extended easily also to the other `Predicate`.
        let predicate_matches_closure = |p: Predicate<'tcx>| {
            let mut relator: SimpleEqRelation<'tcx> = SimpleEqRelation::new(tcx, self_param_env);
            let predicate = predicate.kind();
            let p = p.kind();
            match (predicate.skip_binder(), p.skip_binder()) {
                (
                    ty::PredicateKind::Clause(ty::Clause::Trait(a)),
                    ty::PredicateKind::Clause(ty::Clause::Trait(b)),
                ) => relator.relate(predicate.rebind(a), p.rebind(b)).is_ok(),
                (
                    ty::PredicateKind::Clause(ty::Clause::Projection(a)),
                    ty::PredicateKind::Clause(ty::Clause::Projection(b)),
                ) => relator.relate(predicate.rebind(a), p.rebind(b)).is_ok(),
                (
                    ty::PredicateKind::ConstEvaluatable(a),
                    ty::PredicateKind::ConstEvaluatable(b),
                ) => relator.relate(predicate.rebind(a), predicate.rebind(b)).is_ok(),
                (
                    ty::PredicateKind::Clause(ty::Clause::TypeOutlives(ty::OutlivesPredicate(
                        ty_a,
                        lt_a,
                    ))),
                    ty::PredicateKind::Clause(ty::Clause::TypeOutlives(ty::OutlivesPredicate(
                        ty_b,
                        lt_b,
                    ))),
                ) => {
                    relator.relate(predicate.rebind(ty_a), p.rebind(ty_b)).is_ok()
                        && relator.relate(predicate.rebind(lt_a), p.rebind(lt_b)).is_ok()
                }
                (ty::PredicateKind::WellFormed(arg_a), ty::PredicateKind::WellFormed(arg_b)) => {
                    relator.relate(predicate.rebind(arg_a), p.rebind(arg_b)).is_ok()
                }
                _ => predicate == p,
            }
        };

        if !assumptions_in_impl_context.iter().copied().any(predicate_matches_closure) {
            let item_span = tcx.def_span(self_type_did);
            let self_descr = tcx.def_kind(self_type_did).descr(self_type_did.to_def_id());
            let reported = struct_span_err!(
                tcx.sess,
                predicate_sp,
                E0367,
                "`Drop` impl requires `{predicate}` but the {self_descr} it is implemented for does not",
            )
            .span_note(item_span, "the implementor must specify the same requirement")
            .emit();
            result = Err(reported);
        }
    }

    result
}

/// This is an implementation of the [`TypeRelation`] trait with the
/// aim of simply comparing for equality (without side-effects).
///
/// It is not intended to be used anywhere else other than here.
pub(crate) struct SimpleEqRelation<'tcx> {
    tcx: TyCtxt<'tcx>,
    param_env: ty::ParamEnv<'tcx>,
}

impl<'tcx> SimpleEqRelation<'tcx> {
    fn new(tcx: TyCtxt<'tcx>, param_env: ty::ParamEnv<'tcx>) -> SimpleEqRelation<'tcx> {
        SimpleEqRelation { tcx, param_env }
    }
}

impl<'tcx> TypeRelation<'tcx> for SimpleEqRelation<'tcx> {
    fn tcx(&self) -> TyCtxt<'tcx> {
        self.tcx
    }

    fn intercrate(&self) -> bool {
        false
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

    fn mark_ambiguous(&mut self) {
        bug!()
    }

    fn relate_with_variance<T: Relate<'tcx>>(
        &mut self,
        _: ty::Variance,
        _info: ty::VarianceDiagInfo<'tcx>,
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
        a: ty::Const<'tcx>,
        b: ty::Const<'tcx>,
    ) -> RelateResult<'tcx, ty::Const<'tcx>> {
        debug!("SimpleEqRelation::consts(a={:?}, b={:?})", a, b);
        ty::relate::super_relate_consts(self, a, b)
    }

    fn binders<T>(
        &mut self,
        a: ty::Binder<'tcx, T>,
        b: ty::Binder<'tcx, T>,
    ) -> RelateResult<'tcx, ty::Binder<'tcx, T>>
    where
        T: Relate<'tcx>,
    {
        debug!("SimpleEqRelation::binders({:?}: {:?}", a, b);

        // Anonymizing the LBRs is necessary to solve (Issue #59497).
        // After we do so, it should be totally fine to skip the binders.
        let anon_a = self.tcx.anonymize_bound_vars(a);
        let anon_b = self.tcx.anonymize_bound_vars(b);
        self.relate(anon_a.skip_binder(), anon_b.skip_binder())?;

        Ok(a)
    }
}
