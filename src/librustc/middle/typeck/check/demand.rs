// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use check::fn_ctxt;

// Requires that the two types unify, and prints an error message if they
// don't.
fn suptype(fcx: @fn_ctxt, sp: span,
           expected: ty::t, actual: ty::t) {
    suptype_with_fn(fcx, sp, expected, actual,
        |sp, e, a, s| { fcx.report_mismatched_types(sp, e, a, s) })
}

fn suptype_with_fn(fcx: @fn_ctxt, sp: span,
           expected: ty::t, actual: ty::t,
                   handle_err: fn(span, ty::t, ty::t, &ty::type_err)) {

    // n.b.: order of actual, expected is reversed
    match infer::mk_subty(fcx.infcx(), false, sp,
                          actual, expected) {
      result::Ok(()) => { /* ok */ }
      result::Err(ref err) => {
          handle_err(sp, expected, actual, err);
      }
    }
}

fn eqtype(fcx: @fn_ctxt, sp: span,
          expected: ty::t, actual: ty::t) {

    match infer::mk_eqty(fcx.infcx(), false, sp, actual, expected) {
        Ok(()) => { /* ok */ }
        Err(ref err) => {
            fcx.report_mismatched_types(sp, expected, actual, err);
        }
    }
}

// Checks that the type `actual` can be assigned to `expected`.
fn assign(fcx: @fn_ctxt, sp: span, expected: ty::t, expr: @ast::expr) {
    let expr_ty = fcx.expr_ty(expr);
    match fcx.mk_assignty(expr, expr_ty, expected) {
      result::Ok(()) => { /* ok */ }
      result::Err(ref err) => {
        fcx.report_mismatched_types(sp, expected, expr_ty, err);
      }
    }
}


