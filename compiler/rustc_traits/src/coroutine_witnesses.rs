use rustc_hir::def_id::DefId;
use rustc_middle::ty::{self, TyCtxt, fold_regions};

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

    ty::EarlyBinder::bind(ty::Binder::bind_with_vars(
        ty::CoroutineWitnessTypes { types: bound_tys },
        tcx.mk_bound_variable_kinds(&vars),
    ))
}
