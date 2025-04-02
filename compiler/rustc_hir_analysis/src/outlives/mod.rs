use rustc_hir::def::DefKind;
use rustc_hir::def_id::LocalDefId;
use rustc_middle::query::Providers;
use rustc_middle::ty::{self, CratePredicatesMap, GenericArgKind, TyCtxt, Upcast};
use rustc_span::Span;

pub(crate) mod dump;
mod explicit;
mod implicit_infer;
mod utils;

pub(crate) fn provide(providers: &mut Providers) {
    *providers = Providers { inferred_outlives_of, inferred_outlives_crate, ..*providers };
}

fn inferred_outlives_of(tcx: TyCtxt<'_>, item_def_id: LocalDefId) -> &[(ty::Clause<'_>, Span)] {
    match tcx.def_kind(item_def_id) {
        DefKind::Struct | DefKind::Enum | DefKind::Union => {
            let crate_map = tcx.inferred_outlives_crate(());
            crate_map.predicates.get(&item_def_id.to_def_id()).copied().unwrap_or(&[])
        }
        DefKind::TyAlias if tcx.type_alias_is_lazy(item_def_id) => {
            let crate_map = tcx.inferred_outlives_crate(());
            crate_map.predicates.get(&item_def_id.to_def_id()).copied().unwrap_or(&[])
        }
        DefKind::AnonConst if tcx.features().generic_const_exprs() => {
            let id = tcx.local_def_id_to_hir_id(item_def_id);
            if tcx.hir_opt_const_param_default_param_def_id(id).is_some() {
                // In `generics_of` we set the generics' parent to be our parent's parent which means that
                // we lose out on the predicates of our actual parent if we dont return those predicates here.
                // (See comment in `generics_of` for more information on why the parent shenanigans is necessary)
                //
                // struct Foo<'a, 'b, const N: usize = { ... }>(&'a &'b ());
                //        ^^^                          ^^^^^^^ the def id we are calling
                //        ^^^                                  inferred_outlives_of on
                //        parent item we dont have set as the
                //        parent of generics returned by `generics_of`
                //
                // In the above code we want the anon const to have predicates in its param env for `'b: 'a`
                let item_def_id = tcx.hir_get_parent_item(id);
                // In the above code example we would be calling `inferred_outlives_of(Foo)` here
                tcx.inferred_outlives_of(item_def_id)
            } else {
                &[]
            }
        }
        _ => &[],
    }
}

fn inferred_outlives_crate(tcx: TyCtxt<'_>, (): ()) -> CratePredicatesMap<'_> {
    // Compute a map from each ADT (struct/enum/union) and lazy type alias to
    // the **explicit** outlives predicates (`T: 'a`, `'a: 'b`) that the user wrote.
    // Typically there won't be many of these, except in older code where
    // they were mandatory. Nonetheless, we have to ensure that every such
    // predicate is satisfied, so they form a kind of base set of requirements
    // for the type.

    // Compute the inferred predicates
    let global_inferred_outlives = implicit_infer::infer_predicates(tcx);

    // Convert the inferred predicates into the "collected" form the
    // global data structure expects.
    //
    // FIXME -- consider correcting impedance mismatch in some way,
    // probably by updating the global data structure.
    let predicates = global_inferred_outlives
        .iter()
        .map(|(&def_id, set)| {
            let predicates =
                &*tcx.arena.alloc_from_iter(set.as_ref().skip_binder().iter().filter_map(
                    |(ty::OutlivesPredicate(kind1, region2), &span)| {
                        match kind1.unpack() {
                            GenericArgKind::Type(ty1) => Some((
                                ty::ClauseKind::TypeOutlives(ty::OutlivesPredicate(ty1, *region2))
                                    .upcast(tcx),
                                span,
                            )),
                            GenericArgKind::Lifetime(region1) => Some((
                                ty::ClauseKind::RegionOutlives(ty::OutlivesPredicate(
                                    region1, *region2,
                                ))
                                .upcast(tcx),
                                span,
                            )),
                            GenericArgKind::Const(_) => {
                                // Generic consts don't impose any constraints.
                                None
                            }
                        }
                    },
                ));
            (def_id, predicates)
        })
        .collect();

    ty::CratePredicatesMap { predicates }
}
