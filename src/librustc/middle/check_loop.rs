// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
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
}

pub fn check_crate(sess: &Session, krate: &ast::Crate) {
    visit::walk_crate(&mut CheckLoopVisitor { sess: sess }, krate, Normal)
}

impl<'a> Visitor<Context> for CheckLoopVisitor<'a> {
    fn visit_item(&mut self, i: &ast::Item, _cx: Context) {
        visit::walk_item(self, i, Normal);
    }

    fn visit_expr(&mut self, e: &ast::Expr, cx:Context) {
        match e.node {
            ast::ExprWhile(ref e, ref b) => {
                self.visit_expr(&**e, cx);
                self.visit_block(&**b, Loop);
            }
            ast::ExprLoop(ref b, _) => {
                self.visit_block(&**b, Loop);
            }
            ast::ExprForLoop(_, ref e, ref b, _) => {
                self.visit_expr(&**e, cx);
                self.visit_block(&**b, Loop);
            }
            ast::ExprFnBlock(_, _, ref b) |
            ast::ExprProc(_, ref b) |
            ast::ExprUnboxedFn(_, _, ref b) => {
                self.visit_block(&**b, Closure);
            }
            ast::ExprBreak(_) => self.require_loop("break", cx, e.span),
            ast::ExprAgain(_) => self.require_loop("continue", cx, e.span),
            _ => visit::walk_expr(self, e, cx)
        }
    }
}

impl<'a> CheckLoopVisitor<'a> {
    fn require_loop(&self, name: &str, cx: Context, span: Span) {
        match cx {
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
