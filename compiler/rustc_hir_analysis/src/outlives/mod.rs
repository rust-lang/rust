use rustc_hir::def::DefKind;
use rustc_hir::def_id::LocalDefId;
use rustc_middle::ty::{self, CrateInferredOutlivesClausesMap, GenericArgKind, TyCtxt, Upcast};
use rustc_span::{Span, bug};

pub(crate) mod dump;
mod explicit;
mod implicit_infer;
mod utils;

pub(super) fn inferred_outlives_of(
    tcx: TyCtxt<'_>,
    item_def_id: LocalDefId,
) -> &[(ty::Clause<'_>, Span)] {
    match tcx.def_kind(item_def_id) {
        DefKind::Struct | DefKind::Enum | DefKind::Union => {}
        DefKind::TyAlias if tcx.type_alias_is_checked(item_def_id) => {}
        DefKind::AnonConst
            if tcx.features().generic_const_exprs()
                && let id = tcx.local_def_id_to_hir_id(item_def_id)
                && tcx.hir_opt_const_param_default_param_def_id(id).is_some() =>
        {
            // In `generics_of` we set the generics' parent to be our parent's parent which means that
            // we lose out on the clauses of our actual parent if we dont return those clauses here.
            // (See comment in `generics_of` for more information on why the parent shenanigans is necessary)
            //
            // struct Foo<'a, 'b, const N: usize = { ... }>(&'a &'b ());
            //        ^^^                          ^^^^^^^ the def id we are calling
            //        ^^^                                  inferred_outlives_of on
            //        parent item we dont have set as the
            //        parent of generics returned by `generics_of`
            //
            // In the above code we want the anon const to have clauses in its param env for `'b: 'a`
            let item_def_id = tcx.hir_get_parent_item(id);
            // In the above code example we would be calling `inferred_outlives_of(Foo)` here
            return tcx.inferred_outlives_of(item_def_id);
        }
        _ => return &[],
    }

    let crate_map = tcx.inferred_outlives_crate(());
    crate_map.clauses.get(&item_def_id.to_def_id()).copied().unwrap_or(&[])
}

/// Compute a map from each applicable item in the local crate to its
/// inferred / implied outlives-clauses.
pub(super) fn inferred_outlives_crate(
    tcx: TyCtxt<'_>,
    (): (),
) -> CrateInferredOutlivesClausesMap<'_> {
    let global_inferred_clauses = implicit_infer::infer_outlives_clauses(tcx);

    // Convert the inferred clauses into a form the global data structure expects.
    // FIXME: Consider correcting impedance mismatch in some way,
    //        probably by updating the global data structure.
    let clauses = global_inferred_clauses
        .iter()
        .map(|(&def_id, clauses)| {
            let clauses = clauses.as_ref().skip_binder().iter().filter_map(
                |(&ty::OutlivesClause(arg1, region2), &span)| match arg1.kind() {
                    GenericArgKind::Type(ty1) => Some((
                        ty::ClauseKind::TypeOutlives(ty::OutlivesClause(ty1, region2)).upcast(tcx),
                        span,
                    )),
                    GenericArgKind::Lifetime(region1) => Some((
                        ty::ClauseKind::RegionOutlives(ty::OutlivesClause(region1, region2))
                            .upcast(tcx),
                        span,
                    )),
                    GenericArgKind::Const(_) => bug!(),
                },
            );
            (def_id, &*tcx.arena.alloc_from_iter(clauses))
        })
        .collect();

    ty::CrateInferredOutlivesClausesMap { clauses }
}
