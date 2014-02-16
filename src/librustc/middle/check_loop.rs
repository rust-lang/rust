// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use middle::ty;

use syntax::ast;
use syntax::codemap::Span;
use syntax::visit::Visitor;
use syntax::visit;

#[deriving(Clone, Eq)]
enum Context {
    Normal, Loop, Closure
}

struct CheckLoopVisitor {
    tcx: ty::ctxt,
}

pub fn check_crate(tcx: ty::ctxt, krate: &ast::Crate) {
    visit::walk_crate(&mut CheckLoopVisitor { tcx: tcx }, krate, Normal)
}

impl Visitor<Context> for CheckLoopVisitor {
    fn visit_item(&mut self, i: &ast::Item, _cx: Context) {
        visit::walk_item(self, i, Normal);
    }

    fn visit_expr(&mut self, e: &ast::Expr, cx:Context) {
        match e.node {
            ast::ExprWhile(e, b) => {
                self.visit_expr(e, cx);
                self.visit_block(b, Loop);
            }
            ast::ExprLoop(b, _) => {
                self.visit_block(b, Loop);
            }
            ast::ExprFnBlock(_, b) | ast::ExprProc(_, b) => {
                self.visit_block(b, Closure);
            }
            ast::ExprBreak(_) => self.require_loop("break", cx, e.span),
            ast::ExprAgain(_) => self.require_loop("continue", cx, e.span),
            _ => visit::walk_expr(self, e, cx)
        }
    }
}

impl CheckLoopVisitor {
    fn require_loop(&self, name: &str, cx: Context, span: Span) {
        match cx {
            Loop => {}
            Closure => {
                self.tcx.sess.span_err(span, format!("`{}` inside of a closure",
                                                     name));
            }
            Normal => {
                self.tcx.sess.span_err(span, format!("`{}` outside of loop",
                                                     name));
            }
        }
    }
}
