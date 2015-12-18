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
use syntax::codemap::Span;
use syntax::errors;
use rustc_front::intravisit::Visitor;
use rustc_front::hir;

struct RegistrarFinder {
    registrars: Vec<(ast::NodeId, Span)> ,
}

impl<'v> Visitor<'v> for RegistrarFinder {
    fn visit_item(&mut self, item: &hir::Item) {
        if let hir::ItemFn(..) = item.node {
            if attr::contains_name(&item.attrs,
                                   "plugin_registrar") {
                self.registrars.push((item.id, item.span));
            }
        }
    }
}

/// Find the function marked with `#[plugin_registrar]`, if any.
pub fn find_plugin_registrar(diagnostic: &errors::Handler,
                             krate: &hir::Crate)
                             -> Option<ast::NodeId> {
    let mut finder = RegistrarFinder { registrars: Vec::new() };
    krate.visit_all_items(&mut finder);

    match finder.registrars.len() {
        0 => None,
        1 => {
            let (node_id, _) = finder.registrars.pop().unwrap();
            Some(node_id)
        },
        _ => {
            diagnostic.err("multiple plugin registration functions found");
            for &(_, span) in &finder.registrars {
                diagnostic.span_note(span, "one is here");
            }
            diagnostic.abort_if_errors();
            unreachable!();
        }
    }
}
