use hir::Node;
use rustc::hir;
use rustc::hir::def_id::{CrateNum, DefId, LOCAL_CRATE};
use rustc::ty::query::Providers;
use rustc::ty::subst::UnpackedKind;
use rustc::ty::{self, CratePredicatesMap, TyCtxt};
use rustc_data_structures::sync::Lrc;

mod explicit;
mod implicit_infer;
/// Code to write unit test for outlives.
pub mod test;
mod utils;

pub fn provide(providers: &mut Providers) {
    *providers = Providers {
        inferred_outlives_of,
        inferred_outlives_crate,
        ..*providers
    };
}

fn inferred_outlives_of<'a, 'tcx>(
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    item_def_id: DefId,
) -> Lrc<Vec<ty::Predicate<'tcx>>> {
    let id = tcx
        .hir()
        .as_local_node_id(item_def_id)
        .expect("expected local def-id");

    match tcx.hir().get(id) {
        Node::Item(item) => match item.node {
            hir::ItemKind::Struct(..) | hir::ItemKind::Enum(..) | hir::ItemKind::Union(..) => {
                let crate_map = tcx.inferred_outlives_crate(LOCAL_CRATE);

                let predicates = crate_map
                    .predicates
                    .get(&item_def_id)
                    .unwrap_or(&crate_map.empty_predicate)
                    .clone();

                if tcx.has_attr(item_def_id, "rustc_outlives") {
                    let mut pred: Vec<String> = predicates
                        .iter()
                        .map(|out_pred| match out_pred {
                            ty::Predicate::RegionOutlives(p) => p.to_string(),
                            ty::Predicate::TypeOutlives(p) => p.to_string(),
                            err => bug!("unexpected predicate {:?}", err),
                        }).collect();
                    pred.sort();

                    let span = tcx.def_span(item_def_id);
                    let mut err = tcx.sess.struct_span_err(span, "rustc_outlives");
                    for p in &pred {
                        err.note(p);
                    }
                    err.emit();
                }

                debug!("inferred_outlives_of({:?}) = {:?}", item_def_id, predicates);

                predicates
            }

            _ => Lrc::new(Vec::new()),
        },

        _ => Lrc::new(Vec::new()),
    }
}

fn inferred_outlives_crate<'tcx>(
    tcx: TyCtxt<'_, 'tcx, 'tcx>,
    crate_num: CrateNum,
) -> Lrc<CratePredicatesMap<'tcx>> {
    assert_eq!(crate_num, LOCAL_CRATE);

    // Compute a map from each struct/enum/union S to the **explicit**
    // outlives predicates (`T: 'a`, `'a: 'b`) that the user wrote.
    // Typically there won't be many of these, except in older code where
    // they were mandatory. Nonetheless, we have to ensure that every such
    // predicate is satisfied, so they form a kind of base set of requirements
    // for the type.

    // Compute the inferred predicates
    let mut exp_map = explicit::ExplicitPredicatesMap::new();

    let global_inferred_outlives = implicit_infer::infer_predicates(tcx, &mut exp_map);

    // Convert the inferred predicates into the "collected" form the
    // global data structure expects.
    //
    // FIXME -- consider correcting impedance mismatch in some way,
    // probably by updating the global data structure.
    let predicates = global_inferred_outlives
        .iter()
        .map(|(&def_id, set)| {
            let vec: Vec<ty::Predicate<'tcx>> = set
                .iter()
                .map(
                    |ty::OutlivesPredicate(kind1, region2)| match kind1.unpack() {
                        UnpackedKind::Type(ty1) => ty::Predicate::TypeOutlives(ty::Binder::bind(
                            ty::OutlivesPredicate(ty1, region2),
                        )),
                        UnpackedKind::Lifetime(region1) => ty::Predicate::RegionOutlives(
                            ty::Binder::bind(ty::OutlivesPredicate(region1, region2)),
                        ),
                    },
                ).collect();
            (def_id, Lrc::new(vec))
        }).collect();

    let empty_predicate = Lrc::new(Vec::new());

    Lrc::new(ty::CratePredicatesMap {
        predicates,
        empty_predicate,
    })
}
