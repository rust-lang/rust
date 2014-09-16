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

use ast;
use diagnostic;
use visit;
use visit::Visitor;

struct ShowSpanVisitor<'a> {
    span_diagnostic: &'a diagnostic::SpanHandler,
}

impl<'a, 'v> Visitor<'v> for ShowSpanVisitor<'a> {
    fn visit_expr(&mut self, e: &ast::Expr) {
        self.span_diagnostic.span_note(e.span, "expression");
        visit::walk_expr(self, e);
    }

    fn visit_mac(&mut self, macro: &ast::Mac) {
        visit::walk_mac(self, macro);
    }
}

pub fn run(span_diagnostic: &diagnostic::SpanHandler, krate: &ast::Crate) {
    let mut v = ShowSpanVisitor { span_diagnostic: span_diagnostic };
    visit::walk_crate(&mut v, krate);
}
