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

use std::str::FromStr;

use ast;
use errors;
use visit;
use visit::Visitor;

enum Mode {
    Expression,
    Pattern,
    Type,
}

impl FromStr for Mode {
    type Err = ();
    fn from_str(s: &str) -> Result<Mode, ()> {
        let mode = match s {
            "expr" => Mode::Expression,
            "pat" => Mode::Pattern,
            "ty" => Mode::Type,
            _ => return Err(())
        };
        Ok(mode)
    }
}

struct ShowSpanVisitor<'a> {
    span_diagnostic: &'a errors::Handler,
    mode: Mode,
}

impl<'a, 'v> Visitor<'v> for ShowSpanVisitor<'a> {
    fn visit_expr(&mut self, e: &ast::Expr) {
        if let Mode::Expression = self.mode {
            self.span_diagnostic.span_note(e.span, "expression");
        }
        visit::walk_expr(self, e);
    }

    fn visit_pat(&mut self, p: &ast::Pat) {
        if let Mode::Pattern = self.mode {
            self.span_diagnostic.span_note(p.span, "pattern");
        }
        visit::walk_pat(self, p);
    }

    fn visit_ty(&mut self, t: &ast::Ty) {
        if let Mode::Type = self.mode {
            self.span_diagnostic.span_note(t.span, "type");
        }
        visit::walk_ty(self, t);
    }

    fn visit_mac(&mut self, mac: &ast::Mac) {
        visit::walk_mac(self, mac);
    }
}

pub fn run(span_diagnostic: &errors::Handler,
           mode: &str,
           krate: &ast::Crate) {
    let mode = match mode.parse().ok() {
        Some(mode) => mode,
        None => return
    };
    let mut v = ShowSpanVisitor {
        span_diagnostic: span_diagnostic,
        mode: mode,
    };
    visit::walk_crate(&mut v, krate);
}
