// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
#![allow(unused)]
#[allow(dead_code)]
use hir::map as hir_map;
use rustc::dep_graph::DepKind;
use rustc::hir;
use rustc::hir::Ty_::*;
use rustc::hir::def_id::{CrateNum, DefId, LOCAL_CRATE};
use rustc::ty::maps::Providers;
use rustc::ty::subst::UnpackedKind;
use rustc::ty::{self, CratePredicatesMap, TyCtxt};
use rustc_data_structures::sync::Lrc;
use util::nodemap::FxHashMap;

mod explicit;
mod implicit_empty;
mod implicit_infer;
/// Code to write unit test for outlives.
pub mod test;

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
    let id = tcx.hir
        .as_local_node_id(item_def_id)
        .expect("expected local def-id");

    match tcx.hir.get(id) {
        hir_map::NodeItem(item) => match item.node {
            hir::ItemStruct(..) | hir::ItemEnum(..) | hir::ItemUnion(..) => {
                let crate_map = tcx.inferred_outlives_crate(LOCAL_CRATE);
                let dep_node = item_def_id.to_dep_node(tcx, DepKind::InferredOutlivesOf);
                tcx.dep_graph.read(dep_node);

                crate_map
                    .predicates
                    .get(&item_def_id)
                    .unwrap_or(&crate_map.empty_predicate)
                    .clone()
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
    // Compute a map from each struct/enum/union S to the **explicit**
    // outlives predicates (`T: 'a`, `'a: 'b`) that the user wrote.
    // Typically there won't be many of these, except in older code where
    // they were mandatory. Nonetheless, we have to ensure that every such
    // predicate is satisfied, so they form a kind of base set of requirements
    // for the type.

    // Compute the inferred predicates
    let exp = explicit::explicit_predicates(tcx, crate_num);
    let mut global_inferred_outlives = implicit_infer::infer_predicates(tcx, &exp);

    // Convert the inferred predicates into the "collected" form the
    // global data structure expects.
    //
    // FIXME -- consider correcting impedance mismatch in some way,
    // probably by updating the global data structure.
    let mut predicates = global_inferred_outlives
        .iter()
        .map(|(&def_id, set)| {
            let vec: Vec<ty::Predicate<'tcx>> = set.iter()
                .map(
                    |ty::OutlivesPredicate(kind1, region2)| match kind1.unpack() {
                        UnpackedKind::Type(ty1) => ty::Predicate::TypeOutlives(ty::Binder::bind(
                            ty::OutlivesPredicate(ty1, region2),
                        )),
                        UnpackedKind::Lifetime(region1) => ty::Predicate::RegionOutlives(
                            ty::Binder::bind(ty::OutlivesPredicate(region1, region2)),
                        ),
                    },
                )
                .collect();
            (def_id, Lrc::new(vec))
        })
        .collect();

    let empty_predicate = Lrc::new(Vec::new());

    Lrc::new(ty::CratePredicatesMap {
        predicates,
        empty_predicate,
    })
}
