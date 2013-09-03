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

use syntax::ast::*;
use syntax::visit;
use syntax::visit::Visitor;

#[deriving(Clone)]
pub struct Context {
    in_loop: bool,
    can_ret: bool
}

struct CheckLoopVisitor {
    tcx: ty::ctxt,
}

pub fn check_crate(tcx: ty::ctxt, crate: &Crate) {
    visit::walk_crate(&mut CheckLoopVisitor { tcx: tcx },
                      crate,
                      Context { in_loop: false, can_ret: true });
}

impl Visitor<Context> for CheckLoopVisitor {
    fn visit_item(&mut self, i:@item, _cx:Context) {
        visit::walk_item(self, i, Context {
                                    in_loop: false,
                                    can_ret: true
                                  });
    }

    fn visit_expr(&mut self, e:@Expr, cx:Context) {

            match e.node {
              ExprWhile(e, ref b) => {
                self.visit_expr(e, cx);
                self.visit_block(b, Context { in_loop: true,.. cx });
              }
              ExprLoop(ref b, _) => {
                self.visit_block(b, Context { in_loop: true,.. cx });
              }
              ExprFnBlock(_, ref b) => {
                self.visit_block(b, Context { in_loop: false, can_ret: false });
              }
              ExprBreak(_) => {
                if !cx.in_loop {
                    self.tcx.sess.span_err(e.span, "`break` outside of loop");
                }
              }
              ExprAgain(_) => {
                if !cx.in_loop {
                    self.tcx.sess.span_err(e.span, "`loop` outside of loop");
                }
              }
              ExprRet(oe) => {
                if !cx.can_ret {
                    self.tcx.sess.span_err(e.span, "`return` in block function");
                }
                visit::walk_expr_opt(self, oe, cx);
              }
              _ => visit::walk_expr(self, e, cx)
            }

    }
}
