// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*!

Code related to floating-point type inference.

*/

use to_str::ToStr;
use middle::ty::ty_float;

// Bitvector to represent sets of floating-point types.
pub enum float_ty_set = uint;

// Constants representing singleton sets containing each of the floating-point
// types.
pub const FLOAT_TY_SET_EMPTY: uint = 0b000u;
pub const FLOAT_TY_SET_FLOAT: uint = 0b001u;
pub const FLOAT_TY_SET_F32:   uint = 0b010u;
pub const FLOAT_TY_SET_F64:   uint = 0b100u;

pub fn float_ty_set_all() -> float_ty_set {
    float_ty_set(FLOAT_TY_SET_FLOAT | FLOAT_TY_SET_F32 | FLOAT_TY_SET_F64)
}

pub fn intersection(a: float_ty_set, b: float_ty_set) -> float_ty_set {
    float_ty_set(*a & *b)
}

pub fn single_type_contained_in(tcx: ty::ctxt, a: float_ty_set)
                             -> Option<ty::t> {
    debug!("single_type_contained_in(a=%s)", uint::to_str(*a, 10));

    if *a == FLOAT_TY_SET_FLOAT { return Some(ty::mk_float(tcx)); }
    if *a == FLOAT_TY_SET_F32   { return Some(ty::mk_f32(tcx));   }
    if *a == FLOAT_TY_SET_F64   { return Some(ty::mk_f64(tcx));   }
    return None;
}

pub fn convert_floating_point_ty_to_float_ty_set(tcx: ty::ctxt, t: ty::t)
                                              -> float_ty_set {
    match get(t).sty {
        ty::ty_float(ast::ty_f)     => float_ty_set(FLOAT_TY_SET_FLOAT),
        ty::ty_float(ast::ty_f32)   => float_ty_set(FLOAT_TY_SET_F32),
        ty::ty_float(ast::ty_f64)   => float_ty_set(FLOAT_TY_SET_F64),
        _ => tcx.sess.bug(~"non-floating-point type passed to \
                            convert_floating_point_ty_to_float_ty_set()")
    }
}

