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
use syntax::visit::Visitor;
use syntax::visit;

#[deriving(Clone)]
pub struct Context {
    in_loop: bool,
    can_ret: bool
}

struct LoopCheckingVisitor {
    tcx: ty::ctxt,
}

impl Visitor<Context> for LoopCheckingVisitor {
    fn visit_item(&mut self, i: @item, _: Context) {
        visit::walk_item(self, i, Context {
            in_loop: false,
            can_ret: true,
        });
    }

    fn visit_expr(&mut self, e: @expr, cx: Context) {
        match e.node {
          expr_while(e, ref b) => {
            self.visit_expr(e, cx);
            self.visit_block(b, Context { in_loop: true,.. cx });
          }
          expr_loop(ref b, _) => {
            self.visit_block(b, Context { in_loop: true,.. cx });
          }
          expr_fn_block(_, ref b) => {
            self.visit_block(b, Context {
                in_loop: false,
                can_ret: false,
            });
          }
          expr_break(_) => {
            if !cx.in_loop {
                self.tcx.sess.span_err(e.span, "`break` outside of loop");
            }
          }
          expr_again(_) => {
            if !cx.in_loop {
                self.tcx.sess.span_err(e.span, "`loop` outside of loop");
            }
          }
          expr_ret(oe) => {
            if !cx.can_ret {
                self.tcx.sess.span_err(e.span, "`return` in block function");
            }
            visit::walk_expr_opt(self, oe, cx);
          }
          _ => visit::walk_expr(self, e, cx)
        }
    }
}

pub fn check_crate(tcx: ty::ctxt, crate: &Crate) {
    let cx = Context {
        in_loop: false,
        can_ret: true,
    };
    let mut visitor = LoopCheckingVisitor {
        tcx: tcx,
    };
    visit::walk_crate(&mut visitor, crate, cx);
}
