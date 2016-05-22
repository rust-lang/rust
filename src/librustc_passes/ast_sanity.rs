// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Sanity check AST before lowering it to HIR
//
// This pass is supposed to catch things that fit into AST data structures,
// but not permitted by the language. It runs after expansion when AST is frozen,
// so it can check for erroneous constructions produced by syntax extensions.
// This pass is supposed to perform only simple checks not requiring name resolution
// or type checking or some other kind of complex analysis.

use rustc::lint;
use rustc::session::Session;
use syntax::ast::*;
use syntax::codemap::Span;
use syntax::errors;
use syntax::parse::token::keywords;
use syntax::visit::{self, Visitor};

struct SanityChecker<'a> {
    session: &'a Session,
}

impl<'a> SanityChecker<'a> {
    fn err_handler(&self) -> &errors::Handler {
        &self.session.parse_sess.span_diagnostic
    }

    fn check_label(&self, label: Ident, span: Span, id: NodeId) {
        if label.name == keywords::StaticLifetime.name() {
            self.err_handler().span_err(span, &format!("invalid label name `{}`", label.name));
        }
        if label.name.as_str() == "'_" {
            self.session.add_lint(
                lint::builtin::LIFETIME_UNDERSCORE, id, span,
                format!("invalid label name `{}`", label.name)
            );
        }
    }
}

impl<'a, 'v> Visitor<'v> for SanityChecker<'a> {
    fn visit_lifetime(&mut self, lt: &Lifetime) {
        if lt.name.as_str() == "'_" {
            self.session.add_lint(
                lint::builtin::LIFETIME_UNDERSCORE, lt.id, lt.span,
                format!("invalid lifetime name `{}`", lt.name)
            );
        }

        visit::walk_lifetime(self, lt)
    }

    fn visit_expr(&mut self, expr: &Expr) {
        match expr.node {
            ExprKind::While(_, _, Some(ident)) | ExprKind::Loop(_, Some(ident)) |
            ExprKind::WhileLet(_, _, _, Some(ident)) | ExprKind::ForLoop(_, _, _, Some(ident)) => {
                self.check_label(ident, expr.span, expr.id);
            }
            ExprKind::Break(Some(ident)) | ExprKind::Again(Some(ident)) => {
                self.check_label(ident.node, ident.span, expr.id);
            }
            _ => {}
        }

        visit::walk_expr(self, expr)
    }
}

pub fn check_crate(session: &Session, krate: &Crate) {
    visit::walk_crate(&mut SanityChecker { session: session }, krate)
}
