//! Span debugger
//!
//! This module shows spans for all expressions in the crate
//! to help with compiler debugging.

use std::str::FromStr;

use rustc_ast as ast;
use rustc_ast::visit;
use rustc_ast::visit::Visitor;

use crate::errors;

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
    dcx: &'a rustc_errors::DiagCtxt,
    mode: Mode,
}

impl<'a> Visitor<'a> for ShowSpanVisitor<'a> {
    fn visit_expr(&mut self, e: &'a ast::Expr) {
        if let Mode::Expression = self.mode {
            self.dcx.emit_warn(errors::ShowSpan { span: e.span, msg: "expression" });
        }
        visit::walk_expr(self, e);
    }

    fn visit_pat(&mut self, p: &'a ast::Pat) {
        if let Mode::Pattern = self.mode {
            self.dcx.emit_warn(errors::ShowSpan { span: p.span, msg: "pattern" });
        }
        visit::walk_pat(self, p);
    }

    fn visit_ty(&mut self, t: &'a ast::Ty) {
        if let Mode::Type = self.mode {
            self.dcx.emit_warn(errors::ShowSpan { span: t.span, msg: "type" });
        }
        visit::walk_ty(self, t);
    }
}

pub fn run(dcx: &rustc_errors::DiagCtxt, mode: &str, krate: &ast::Crate) {
    let Ok(mode) = mode.parse() else {
        return;
    };
    let mut v = ShowSpanVisitor { dcx, mode };
    visit::walk_crate(&mut v, krate);
}
