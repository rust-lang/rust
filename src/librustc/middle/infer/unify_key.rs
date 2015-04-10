// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use middle::ty::{self, IntVarValue, Ty};
use rustc_data_structures::unify::UnifyKey;
use syntax::ast;

pub trait ToType<'tcx> {
    fn to_type(&self, tcx: &ty::ctxt<'tcx>) -> Ty<'tcx>;
}

impl UnifyKey for ty::IntVid {
    type Value = Option<IntVarValue>;
    fn index(&self) -> u32 { self.index }
    fn from_index(i: u32) -> ty::IntVid { ty::IntVid { index: i } }
    fn tag(_: Option<ty::IntVid>) -> &'static str { "IntVid" }
}

impl<'tcx> ToType<'tcx> for IntVarValue {
    fn to_type(&self, tcx: &ty::ctxt<'tcx>) -> Ty<'tcx> {
        match *self {
            ty::IntType(i) => ty::mk_mach_int(tcx, i),
            ty::UintType(i) => ty::mk_mach_uint(tcx, i),
        }
    }
}

// Floating point type keys

impl UnifyKey for ty::FloatVid {
    type Value = Option<ast::FloatTy>;
    fn index(&self) -> u32 { self.index }
    fn from_index(i: u32) -> ty::FloatVid { ty::FloatVid { index: i } }
    fn tag(_: Option<ty::FloatVid>) -> &'static str { "FloatVid" }
}

impl<'tcx> ToType<'tcx> for ast::FloatTy {
    fn to_type(&self, tcx: &ty::ctxt<'tcx>) -> Ty<'tcx> {
        ty::mk_mach_float(tcx, *self)
    }
}
