// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Verifies that const fn arguments are immutable by value bindings
//! and the const fn body doesn't contain any statements

use rustc::session::{Session, CompileResult};

use syntax::ast::{self, PatKind};
use syntax::visit::{self, Visitor, FnKind};
use syntax::codemap::Span;

pub fn check_crate(sess: &Session, krate: &ast::Crate) -> CompileResult {
    sess.track_errors(|| {
        visit::walk_crate(&mut CheckConstFn{ sess: sess }, krate);
    })
}

struct CheckConstFn<'a> {
    sess: &'a Session,
}

struct CheckBlock<'a> {
    sess: &'a Session,
    kind: &'static str,
}

impl<'a, 'v> Visitor<'v> for CheckBlock<'a> {
    fn visit_block(&mut self, block: &'v ast::Block) {
        check_block(&self.sess, block, self.kind);
        CheckConstFn{ sess: self.sess}.visit_block(block);
    }
    fn visit_expr(&mut self, e: &'v ast::Expr) {
        if let ast::ExprKind::Closure(..) = e.node {
            CheckConstFn{ sess: self.sess}.visit_expr(e);
        } else {
            visit::walk_expr(self, e);
        }
    }
    fn visit_item(&mut self, _i: &'v ast::Item) { bug!("should be handled in CheckConstFn") }
    fn visit_fn(&mut self,
                _fk: FnKind<'v>,
                _fd: &'v ast::FnDecl,
                _b: &'v ast::Block,
                _s: Span,
                _fn_id: ast::NodeId) { bug!("should be handled in CheckConstFn") }
}

fn check_block(sess: &Session, b: &ast::Block, kind: &'static str) {
    // Check all statements in the block
    for stmt in &b.stmts {
        let span = match stmt.node {
            ast::StmtKind::Decl(ref decl, _) => {
                match decl.node {
                    ast::DeclKind::Local(_) => decl.span,

                    // Item statements are allowed
                    ast::DeclKind::Item(_) => continue,
                }
            }
            ast::StmtKind::Expr(ref expr, _) => expr.span,
            ast::StmtKind::Semi(ref semi, _) => semi.span,
            ast::StmtKind::Mac(..) => bug!(),
        };
        span_err!(sess, span, E0016,
                  "blocks in {}s are limited to items and tail expressions", kind);
    }
}

impl<'a, 'v> Visitor<'v> for CheckConstFn<'a> {
    fn visit_item(&mut self, i: &'v ast::Item) {
        visit::walk_item(self, i);
        match i.node {
            ast::ItemKind::Const(_, ref e) => {
                CheckBlock{ sess: self.sess, kind: "constant"}.visit_expr(e)
            },
            ast::ItemKind::Static(_, _, ref e) => {
                CheckBlock{ sess: self.sess, kind: "static"}.visit_expr(e)
            },
            _ => {},
        }
    }

    fn visit_fn(&mut self,
                fk: FnKind<'v>,
                fd: &'v ast::FnDecl,
                b: &'v ast::Block,
                s: Span,
                _fn_id: ast::NodeId) {
        visit::walk_fn(self, fk, fd, b, s);
        match fk {
            FnKind::ItemFn(_, _, _, ast::Constness::Const, _, _) => {},
            FnKind::Method(_, m, _) if m.constness == ast::Constness::Const => {},
            _ => return,
        }

        // Ensure the arguments are simple, not mutable/by-ref or patterns.
        for arg in &fd.inputs {
            match arg.pat.node {
                PatKind::Wild => {}
                PatKind::Ident(ast::BindingMode::ByValue(ast::Mutability::Immutable), _, None) => {}
                _ => {
                    span_err!(self.sess, arg.pat.span, E0022,
                              "arguments of constant functions can only \
                               be immutable by-value bindings");
                }
            }
        }
        check_block(&self.sess, b, "const function");
    }
}
