//! Span debugger
//!
//! This module shows spans for all expressions in the crate
//! to help with compiler debugging.

use std::str::FromStr;

use rustc_ast as ast;
use rustc_ast::visit;
use rustc_ast::visit::Visitor;

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
            _ => return Err(()),
        };
        Ok(mode)
    }
}

struct ShowSpanVisitor<'a> {
    span_diagnostic: &'a rustc_errors::Handler,
    mode: Mode,
}

impl<'a> Visitor<'a> for ShowSpanVisitor<'a> {
    fn visit_expr(&mut self, e: &'a ast::Expr) {
        if let Mode::Expression = self.mode {
            self.span_diagnostic.span_warn(e.span, "expression");
        }
        visit::walk_expr(self, e);
    }

    fn visit_pat(&mut self, p: &'a ast::Pat) {
        if let Mode::Pattern = self.mode {
            self.span_diagnostic.span_warn(p.span, "pattern");
        }
        visit::walk_pat(self, p);
    }

    fn visit_ty(&mut self, t: &'a ast::Ty) {
        if let Mode::Type = self.mode {
            self.span_diagnostic.span_warn(t.span, "type");
        }
        visit::walk_ty(self, t);
    }
}

pub fn run(span_diagnostic: &rustc_errors::Handler, mode: &str, krate: &ast::Crate) {
    let mode = match mode.parse().ok() {
        Some(mode) => mode,
        None => return,
    };
    let mut v = ShowSpanVisitor { span_diagnostic, mode };
    visit::walk_crate(&mut v, krate);
}
