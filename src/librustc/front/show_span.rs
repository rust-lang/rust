// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Span debugger
//!
//! This module shows spans for all expressions in the crate
//! to help with compiler debugging.

use syntax::ast;
use syntax::visit;
use syntax::visit::Visitor;

use driver::session::Session;

struct ShowSpanVisitor<'a> {
    sess: &'a Session
}

impl<'a> Visitor<()> for ShowSpanVisitor<'a> {
    fn visit_expr(&mut self, e: &ast::Expr, _: ()) {
        self.sess.span_note(e.span, "expression");
        visit::walk_expr(self, e, ());
    }

    fn visit_mac(&mut self, macro: &ast::Mac, e: ()) {
        visit::walk_mac(self, macro, e);
    }
}

pub fn run(sess: &Session, krate: &ast::Crate) {
    let mut v = ShowSpanVisitor { sess: sess };
    visit::walk_crate(&mut v, krate, ());
}
