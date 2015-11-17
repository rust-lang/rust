// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
use self::Context::*;

use session::Session;

use syntax::codemap::Span;
use rustc_front::intravisit::{self, Visitor};
use rustc_front::hir;

#[derive(Clone, Copy, PartialEq)]
enum Context {
    Normal, Loop, Closure
}

#[derive(Copy, Clone)]
struct CheckLoopVisitor<'a> {
    sess: &'a Session,
    cx: Context
}

pub fn check_crate(sess: &Session, krate: &hir::Crate) {
    krate.visit_all_items(&mut CheckLoopVisitor { sess: sess, cx: Normal });
}

impl<'a, 'v> Visitor<'v> for CheckLoopVisitor<'a> {
    fn visit_item(&mut self, i: &hir::Item) {
        self.with_context(Normal, |v| intravisit::walk_item(v, i));
    }

    fn visit_expr(&mut self, e: &hir::Expr) {
        match e.node {
            hir::ExprWhile(ref e, ref b, _) => {
                self.visit_expr(&**e);
                self.with_context(Loop, |v| v.visit_block(&**b));
            }
            hir::ExprLoop(ref b, _) => {
                self.with_context(Loop, |v| v.visit_block(&**b));
            }
            hir::ExprClosure(_, _, ref b) => {
                self.with_context(Closure, |v| v.visit_block(&**b));
            }
            hir::ExprBreak(_) => self.require_loop("break", e.span),
            hir::ExprAgain(_) => self.require_loop("continue", e.span),
            _ => intravisit::walk_expr(self, e)
        }
    }
}

impl<'a> CheckLoopVisitor<'a> {
    fn with_context<F>(&mut self, cx: Context, f: F) where
        F: FnOnce(&mut CheckLoopVisitor<'a>),
    {
        let old_cx = self.cx;
        self.cx = cx;
        f(self);
        self.cx = old_cx;
    }

    fn require_loop(&self, name: &str, span: Span) {
        match self.cx {
            Loop => {}
            Closure => {
                span_err!(self.sess, span, E0267,
                                   "`{}` inside of a closure", name);
            }
            Normal => {
                span_err!(self.sess, span, E0268,
                                   "`{}` outside of loop", name);
            }
        }
    }
}
