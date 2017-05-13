// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Used by `rustc` when compiling a plugin crate.

use syntax::ast;
use syntax::attr;
use errors;
use syntax_pos::Span;
use rustc::dep_graph::DepNode;
use rustc::hir::map::Map;
use rustc::hir::itemlikevisit::ItemLikeVisitor;
use rustc::hir;

struct RegistrarFinder {
    registrars: Vec<(ast::NodeId, Span)> ,
}

impl<'v> ItemLikeVisitor<'v> for RegistrarFinder {
    fn visit_item(&mut self, item: &hir::Item) {
        if let hir::ItemFn(..) = item.node {
            if attr::contains_name(&item.attrs,
                                   "plugin_registrar") {
                self.registrars.push((item.id, item.span));
            }
        }
    }

    fn visit_trait_item(&mut self, _trait_item: &hir::TraitItem) {
    }

    fn visit_impl_item(&mut self, _impl_item: &hir::ImplItem) {
    }
}

/// Find the function marked with `#[plugin_registrar]`, if any.
pub fn find_plugin_registrar(diagnostic: &errors::Handler,
                             hir_map: &Map)
                             -> Option<ast::NodeId> {
    let _task = hir_map.dep_graph.in_task(DepNode::PluginRegistrar);
    let krate = hir_map.krate();

    let mut finder = RegistrarFinder { registrars: Vec::new() };
    krate.visit_all_item_likes(&mut finder);

    match finder.registrars.len() {
        0 => None,
        1 => {
            let (node_id, _) = finder.registrars.pop().unwrap();
            Some(node_id)
        },
        _ => {
            let mut e = diagnostic.struct_err("multiple plugin registration functions found");
            for &(_, span) in &finder.registrars {
                e.span_note(span, "one is here");
            }
            e.emit();
            diagnostic.abort_if_errors();
            unreachable!();
        }
    }
}
