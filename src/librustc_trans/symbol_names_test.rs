// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Walks the crate looking for items/impl-items/trait-items that have
//! either a `rustc_symbol_name` or `rustc_item_path` attribute and
//! generates an error giving, respectively, the symbol name or
//! item-path. This is used for unit testing the code that generates
//! paths etc in all kinds of annoying scenarios.

use rustc::hir;
use rustc::hir::intravisit::{self, Visitor, NestedVisitorMap};
use syntax::ast;

use common::SharedCrateContext;
use monomorphize::Instance;

const SYMBOL_NAME: &'static str = "rustc_symbol_name";
const ITEM_PATH: &'static str = "rustc_item_path";

pub fn report_symbol_names(scx: &SharedCrateContext) {
    // if the `rustc_attrs` feature is not enabled, then the
    // attributes we are interested in cannot be present anyway, so
    // skip the walk.
    let tcx = scx.tcx();
    if !tcx.sess.features.borrow().rustc_attrs {
        return;
    }

    let _ignore = tcx.dep_graph.in_ignore();
    let mut visitor = SymbolNamesTest { scx: scx };
    // FIXME(#37712) could use ItemLikeVisitor if trait items were item-like
    tcx.map.krate().visit_all_item_likes(&mut visitor.as_deep_visitor());
}

struct SymbolNamesTest<'a, 'tcx:'a> {
    scx: &'a SharedCrateContext<'a, 'tcx>,
}

impl<'a, 'tcx> SymbolNamesTest<'a, 'tcx> {
    fn process_attrs(&mut self,
                     node_id: ast::NodeId) {
        let tcx = self.scx.tcx();
        let def_id = tcx.map.local_def_id(node_id);
        for attr in tcx.get_attrs(def_id).iter() {
            if attr.check_name(SYMBOL_NAME) {
                // for now, can only use on monomorphic names
                let instance = Instance::mono(self.scx, def_id);
                let name = instance.symbol_name(self.scx);
                tcx.sess.span_err(attr.span, &format!("symbol-name({})", name));
            } else if attr.check_name(ITEM_PATH) {
                let path = tcx.item_path_str(def_id);
                tcx.sess.span_err(attr.span, &format!("item-path({})", path));
            }

            // (*) The formatting of `tag({})` is chosen so that tests can elect
            // to test the entirety of the string, if they choose, or else just
            // some subset.
        }
    }
}

impl<'a, 'tcx> Visitor<'tcx> for SymbolNamesTest<'a, 'tcx> {
    fn nested_visit_map<'this>(&'this mut self) -> NestedVisitorMap<'this, 'tcx> {
        NestedVisitorMap::None
    }

    fn visit_item(&mut self, item: &'tcx hir::Item) {
        self.process_attrs(item.id);
        intravisit::walk_item(self, item);
    }

    fn visit_trait_item(&mut self, ti: &'tcx hir::TraitItem) {
        self.process_attrs(ti.id);
        intravisit::walk_trait_item(self, ti)
    }

    fn visit_impl_item(&mut self, ii: &'tcx hir::ImplItem) {
        self.process_attrs(ii.id);
        intravisit::walk_impl_item(self, ii)
    }
}

