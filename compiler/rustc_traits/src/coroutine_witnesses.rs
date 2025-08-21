use rustc_infer::infer::TyCtxtInferExt;
use rustc_infer::infer::canonical::query_response::make_query_region_constraints;
use rustc_infer::infer::resolve::OpportunisticRegionResolver;
use rustc_infer::traits::{Obligation, ObligationCause};
use rustc_middle::ty::{self, Ty, TyCtxt, TypeFoldable, TypeVisitableExt, fold_regions};
use rustc_span::def_id::DefId;
use rustc_trait_selection::traits::{ObligationCtxt, with_replaced_escaping_bound_vars};

/// Return the set of types that should be taken into account when checking
/// trait bounds on a coroutine's internal state. This properly replaces
/// `ReErased` with new existential bound lifetimes.
pub(crate) fn coroutine_hidden_types<'tcx>(
    tcx: TyCtxt<'tcx>,
    def_id: DefId,
) -> ty::EarlyBinder<'tcx, ty::Binder<'tcx, ty::CoroutineWitnessTypes<TyCtxt<'tcx>>>> {
    let coroutine_layout = tcx.mir_coroutine_witnesses(def_id);
    let mut vars = vec![];
    let bound_tys = tcx.mk_type_list_from_iter(
        coroutine_layout
            .as_ref()
            .map_or_else(|| [].iter(), |l| l.field_tys.iter())
            .filter(|decl| !decl.ignore_for_traits)
            .map(|decl| {
                let ty = fold_regions(tcx, decl.ty, |re, debruijn| {
                    assert_eq!(re, tcx.lifetimes.re_erased);
                    let var = ty::BoundVar::from_usize(vars.len());
                    vars.push(ty::BoundVariableKind::Region(ty::BoundRegionKind::Anon));
                    ty::Region::new_bound(
                        tcx,
                        debruijn,
                        ty::BoundRegion { var, kind: ty::BoundRegionKind::Anon },
                    )
                });
                ty
            }),
    );

    let assumptions = compute_assumptions(tcx, def_id, bound_tys);

    ty::EarlyBinder::bind(ty::Binder::bind_with_vars(
        ty::CoroutineWitnessTypes { types: bound_tys, assumptions },
        tcx.mk_bound_variable_kinds(&vars),
    ))
}

fn compute_assumptions<'tcx>(
    tcx: TyCtxt<'tcx>,
    def_id: DefId,
    bound_tys: &'tcx ty::List<Ty<'tcx>>,
) -> &'tcx ty::List<ty::ArgOutlivesPredicate<'tcx>> {
    let infcx = tcx.infer_ctxt().build(ty::TypingMode::Analysis {
        defining_opaque_types_and_generators: ty::List::empty(),
    });
    with_replaced_escaping_bound_vars(&infcx, &mut vec![None], bound_tys, |bound_tys| {
        let param_env = tcx.param_env(def_id);
        let ocx = ObligationCtxt::new(&infcx);

        ocx.register_obligations(bound_tys.iter().map(|ty| {
            Obligation::new(
                tcx,
                ObligationCause::dummy(),
                param_env,
                ty::ClauseKind::WellFormed(ty.into()),
            )
        }));
        let _errors = ocx.select_all_or_error();

        let region_obligations = infcx.take_registered_region_obligations();
        let region_assumptions = infcx.take_registered_region_assumptions();
        let region_constraints = infcx.take_and_reset_region_constraints();

        let outlives = make_query_region_constraints(
            region_obligations,
            &region_constraints,
            region_assumptions,
        )
        .outlives
        .fold_with(&mut OpportunisticRegionResolver::new(&infcx));

        tcx.mk_outlives_from_iter(
            outlives
                .into_iter()
                .map(|(o, _)| o)
                // FIXME(higher_ranked_auto): We probably should deeply resolve these before
                // filtering out infers which only correspond to unconstrained infer regions
                // which we can sometimes get.
                .filter(|o| !o.has_infer()),
        )
    })
}
