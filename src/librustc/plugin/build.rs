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
use syntax::diagnostic;
use syntax::visit;
use syntax::visit::Visitor;

struct RegistrarFinder {
    registrars: Vec<(ast::NodeId, Span)> ,
}

impl<'v> Visitor<'v> for RegistrarFinder {
    fn visit_item(&mut self, item: &ast::Item) {
        match item.node {
            ast::ItemFn(..) => {
                if attr::contains_name(item.attrs.as_slice(),
                                       "plugin_registrar") {
                    self.registrars.push((item.id, item.span));
                }
            }
            _ => {}
        }

        visit::walk_item(self, item);
    }
}

/// Find the function marked with `#[plugin_registrar]`, if any.
pub fn find_plugin_registrar(diagnostic: &diagnostic::SpanHandler,
                             krate: &ast::Crate) -> Option<ast::NodeId> {
    let mut finder = RegistrarFinder { registrars: Vec::new() };
    visit::walk_crate(&mut finder, krate);

    match finder.registrars.len() {
        0 => None,
        1 => {
            let (node_id, _) = finder.registrars.pop().unwrap();
            Some(node_id)
        },
        _ => {
            diagnostic.handler().err("multiple plugin registration functions found");
            for &(_, span) in finder.registrars.iter() {
                diagnostic.span_note(span, "one is here");
            }
            diagnostic.handler().abort_if_errors();
            unreachable!();
        }
    }
}
