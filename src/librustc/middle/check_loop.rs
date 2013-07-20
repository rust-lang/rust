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
use syntax::oldvisit;

#[deriving(Clone)]
pub struct Context {
    in_loop: bool,
    can_ret: bool
}

pub fn check_crate(tcx: ty::ctxt, crate: &Crate) {
    oldvisit::visit_crate(crate,
                          (Context { in_loop: false, can_ret: true },
                          oldvisit::mk_vt(@oldvisit::Visitor {
        visit_item: |i, (_cx, v)| {
            oldvisit::visit_item(i, (Context {
                                    in_loop: false,
                                    can_ret: true
                                 }, v));
        },
        visit_expr: |e: @expr, (cx, v): (Context, oldvisit::vt<Context>)| {
            match e.node {
              expr_while(e, ref b) => {
                (v.visit_expr)(e, (cx, v));
                (v.visit_block)(b, (Context { in_loop: true,.. cx }, v));
              }
              expr_loop(ref b, _) => {
                (v.visit_block)(b, (Context { in_loop: true,.. cx }, v));
              }
              expr_fn_block(_, ref b) => {
                (v.visit_block)(b, (Context {
                                         in_loop: false,
                                         can_ret: false
                                      }, v));
              }
              expr_loop_body(@expr {node: expr_fn_block(_, ref b), _}) => {
                let sigil = ty::ty_closure_sigil(ty::expr_ty(tcx, e));
                let blk = (sigil == BorrowedSigil);
                (v.visit_block)(b, (Context {
                                         in_loop: true,
                                         can_ret: blk
                                     }, v));
              }
              expr_break(_) => {
                if !cx.in_loop {
                    tcx.sess.span_err(e.span, "`break` outside of loop");
                }
              }
              expr_again(_) => {
                if !cx.in_loop {
                    tcx.sess.span_err(e.span, "`again` outside of loop");
                }
              }
              expr_ret(oe) => {
                if !cx.can_ret {
                    tcx.sess.span_err(e.span, "`return` in block function");
                }
                oldvisit::visit_expr_opt(oe, (cx, v));
              }
              _ => oldvisit::visit_expr(e, (cx, v))
            }
        },
        .. *oldvisit::default_visitor()
    })));
}
