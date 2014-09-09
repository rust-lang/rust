// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use driver::session::Session;

use syntax::ast;
use syntax::codemap::Span;
use syntax::visit::Visitor;
use syntax::visit;

#[deriving(Clone, PartialEq)]
enum Context {
    Normal, Loop, Closure
}

struct CheckLoopVisitor<'a> {
    sess: &'a Session,
    cx: Context
}

pub fn check_crate(sess: &Session, krate: &ast::Crate) {
    visit::walk_crate(&mut CheckLoopVisitor { sess: sess, cx: Normal }, krate)
}

impl<'a, 'v> Visitor<'v> for CheckLoopVisitor<'a> {
    fn visit_item(&mut self, i: &ast::Item) {
        self.with_context(Normal, |v| visit::walk_item(v, i));
    }

    fn visit_expr(&mut self, e: &ast::Expr) {
        match e.node {
            ast::ExprWhile(ref e, ref b, _) => {
                self.visit_expr(&**e);
                self.with_context(Loop, |v| v.visit_block(&**b));
            }
            ast::ExprLoop(ref b, _) => {
                self.with_context(Loop, |v| v.visit_block(&**b));
            }
            ast::ExprForLoop(_, ref e, ref b, _) => {
                self.visit_expr(&**e);
                self.with_context(Loop, |v| v.visit_block(&**b));
            }
            ast::ExprFnBlock(_, _, ref b) |
            ast::ExprProc(_, ref b) |
            ast::ExprUnboxedFn(_, _, _, ref b) => {
                self.with_context(Closure, |v| v.visit_block(&**b));
            }
            ast::ExprBreak(_) => self.require_loop("break", e.span),
            ast::ExprAgain(_) => self.require_loop("continue", e.span),
            _ => visit::walk_expr(self, e)
        }
    }
}

impl<'a> CheckLoopVisitor<'a> {
    fn with_context(&mut self, cx: Context, f: |&mut CheckLoopVisitor<'a>|) {
        let old_cx = self.cx;
        self.cx = cx;
        f(self);
        self.cx = old_cx;
    }

    fn require_loop(&self, name: &str, span: Span) {
        match self.cx {
            Loop => {}
            Closure => {
                self.sess.span_err(span,
                                   format!("`{}` inside of a closure",
                                           name).as_slice());
            }
            Normal => {
                self.sess.span_err(span,
                                   format!("`{}` outside of loop",
                                           name).as_slice());
            }
        }
    }
}
