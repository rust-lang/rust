// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use hir::map as hir_map;
use rustc::hir;
use rustc::hir::def_id::{self, CrateNum, DefId, LOCAL_CRATE};
use rustc::hir::itemlikevisit::ItemLikeVisitor;
use rustc::ty::maps::Providers;
use rustc::ty::{self, CratePredicatesMap, TyCtxt};
use rustc_data_structures::sync::Lrc;
use util::nodemap::FxHashMap;

// Create the sets of inferred predicates for each type. These sets
// are initially empty but will grow during the inference step.
pub fn empty_predicate_map<'tcx>(
    tcx: TyCtxt<'_, 'tcx, 'tcx>,
) -> FxHashMap<DefId, Lrc<Vec<ty::Predicate<'tcx>>>> {
    let mut predicates = FxHashMap();

    // iterate over the entire crate
    tcx.hir
        .krate()
        .visit_all_item_likes(&mut EmptyImplicitVisitor {
            tcx,
            predicates: &mut predicates,
        });

    predicates
}

pub struct EmptyImplicitVisitor<'cx, 'tcx: 'cx> {
    tcx: TyCtxt<'cx, 'tcx, 'tcx>,
    predicates: &'cx mut FxHashMap<DefId, Lrc<Vec<ty::Predicate<'tcx>>>>,
}

impl<'a, 'p, 'v> ItemLikeVisitor<'v> for EmptyImplicitVisitor<'a, 'p> {
    fn visit_item(&mut self, item: &hir::Item) {
        self.predicates
            .insert(self.tcx.hir.local_def_id(item.id), Lrc::new(Vec::new()));
    }

    fn visit_trait_item(&mut self, trait_item: &hir::TraitItem) {}

    fn visit_impl_item(&mut self, impl_item: &hir::ImplItem) {}
}
