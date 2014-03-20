// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use ast;
use attr;
use codemap::Span;
use diagnostic;
use visit;
use visit::Visitor;

use std::vec::Vec;

struct MacroRegistrarContext {
    registrars: Vec<(ast::NodeId, Span)> ,
}

impl Visitor<()> for MacroRegistrarContext {
    fn visit_item(&mut self, item: &ast::Item, _: ()) {
        match item.node {
            ast::ItemFn(..) => {
                if attr::contains_name(item.attrs.as_slice(),
                                       "macro_registrar") {
                    self.registrars.push((item.id, item.span));
                }
            }
            _ => {}
        }

        visit::walk_item(self, item, ());
    }
}

pub fn find_macro_registrar(diagnostic: &diagnostic::SpanHandler,
                            krate: &ast::Crate) -> Option<ast::DefId> {
    let mut ctx = MacroRegistrarContext { registrars: Vec::new() };
    visit::walk_crate(&mut ctx, krate, ());

    match ctx.registrars.len() {
        0 => None,
        1 => {
            let (node_id, _) = ctx.registrars.pop().unwrap();
            Some(ast::DefId {
                krate: ast::LOCAL_CRATE,
                node: node_id
            })
        },
        _ => {
            diagnostic.handler().err("multiple macro registration functions found");
            for &(_, span) in ctx.registrars.iter() {
                diagnostic.span_note(span, "one is here");
            }
            diagnostic.handler().abort_if_errors();
            unreachable!();
        }
    }
}
