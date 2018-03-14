// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use rustc::hir::itemlikevisit::ItemLikeVisitor;
use rustc::hir;
use rustc::middle::cstore::ForeignModule;
use rustc::ty::TyCtxt;

pub fn collect<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>) -> Vec<ForeignModule> {
    let mut collector = Collector {
        tcx,
        modules: Vec::new(),
    };
    tcx.hir.krate().visit_all_item_likes(&mut collector);
    return collector.modules
}

struct Collector<'a, 'tcx: 'a> {
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    modules: Vec<ForeignModule>,
}

impl<'a, 'tcx> ItemLikeVisitor<'tcx> for Collector<'a, 'tcx> {
    fn visit_item(&mut self, it: &'tcx hir::Item) {
        let fm = match it.node {
            hir::ItemForeignMod(ref fm) => fm,
            _ => return,
        };

        let foreign_items = fm.items.iter()
            .map(|it| self.tcx.hir.local_def_id(it.id))
            .collect();
        self.modules.push(ForeignModule {
            foreign_items,
            def_id: self.tcx.hir.local_def_id(it.id),
        });
    }

    fn visit_trait_item(&mut self, _it: &'tcx hir::TraitItem) {}
    fn visit_impl_item(&mut self, _it: &'tcx hir::ImplItem) {}
}
