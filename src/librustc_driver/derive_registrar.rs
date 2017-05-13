// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use rustc::dep_graph::DepNode;
use rustc::hir::itemlikevisit::ItemLikeVisitor;
use rustc::hir::map::Map;
use rustc::hir;
use syntax::ast;
use syntax::attr;

pub fn find(hir_map: &Map) -> Option<ast::NodeId> {
    let _task = hir_map.dep_graph.in_task(DepNode::PluginRegistrar);
    let krate = hir_map.krate();

    let mut finder = Finder { registrar: None };
    krate.visit_all_item_likes(&mut finder);
    finder.registrar
}

struct Finder {
    registrar: Option<ast::NodeId>,
}

impl<'v> ItemLikeVisitor<'v> for Finder {
    fn visit_item(&mut self, item: &hir::Item) {
        if attr::contains_name(&item.attrs, "rustc_derive_registrar") {
            self.registrar = Some(item.id);
        }
    }

    fn visit_trait_item(&mut self, _trait_item: &hir::TraitItem) {
    }

    fn visit_impl_item(&mut self, _impl_item: &hir::ImplItem) {
    }
}

