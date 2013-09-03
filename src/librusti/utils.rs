// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::io;
use syntax::ast;
use syntax::print::pp;
use syntax::print::pprust;
use syntax::parse::token;
use syntax::visit;

struct EachBindingVisitor {
    f: @fn(&ast::Path, ast::NodeId)
}

impl visit::Visitor<()> for EachBindingVisitor {
    fn visit_pat(&mut self, pat:@ast::Pat, _:()) {
                match pat.node {
                    ast::PatIdent(_, ref path, _) => {
                        (self.f)(path, pat.id);
                    }
                    _ => {}
                }

                visit::walk_pat(self, pat, ());
    }
}

pub fn each_binding(l: @ast::Local, f: @fn(&ast::Path, ast::NodeId)) {
    use syntax::visit::Visitor;

    let mut vt = EachBindingVisitor{ f: f };

    vt.visit_pat(l.pat, ());
}

/// A utility function that hands off a pretty printer to a callback.
pub fn with_pp(intr: @token::ident_interner,
               cb: &fn(@pprust::ps, @io::Writer)) -> ~str {
    do io::with_str_writer |writer| {
        let pp = pprust::rust_printer(writer, intr);

        cb(pp, writer);
        pp::eof(pp.s);
    }
}
