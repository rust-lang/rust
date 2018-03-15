// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use syntax::ast;
use ty::{self, IntVarValue, Ty, TyCtxt};
use rustc_data_structures::unify::{Combine, UnifyKey};

pub trait ToType {
    fn to_type<'a, 'gcx, 'tcx>(&self, tcx: TyCtxt<'a, 'gcx, 'tcx>) -> Ty<'tcx>;
}

impl UnifyKey for ty::IntVid {
    type Value = Option<IntVarValue>;
    fn index(&self) -> u32 { self.index }
    fn from_index(i: u32) -> ty::IntVid { ty::IntVid { index: i } }
    fn tag(_: Option<ty::IntVid>) -> &'static str { "IntVid" }
}

#[derive(PartialEq, Copy, Clone, Debug)]
pub struct RegionVidKey {
    /// The minimum region vid in the unification set. This is needed
    /// to have a canonical name for a type to prevent infinite
    /// recursion.
    pub min_vid: ty::RegionVid
}

impl Combine for RegionVidKey {
    fn combine(&self, other: &RegionVidKey) -> RegionVidKey {
        let min_vid = if self.min_vid.index < other.min_vid.index {
            self.min_vid
        } else {
            other.min_vid
        };

        RegionVidKey { min_vid: min_vid }
    }
}

impl UnifyKey for ty::RegionVid {
    type Value = RegionVidKey;
    fn index(&self) -> u32 { self.index }
    fn from_index(i: u32) -> ty::RegionVid { ty::RegionVid { index: i } }
    fn tag(_: Option<ty::RegionVid>) -> &'static str { "RegionVid" }
}

impl ToType for IntVarValue {
    fn to_type<'a, 'gcx, 'tcx>(&self, tcx: TyCtxt<'a, 'gcx, 'tcx>) -> Ty<'tcx> {
        match *self {
            ty::IntType(i) => tcx.mk_mach_int(i),
            ty::UintType(i) => tcx.mk_mach_uint(i),
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

impl ToType for ast::FloatTy {
    fn to_type<'a, 'gcx, 'tcx>(&self, tcx: TyCtxt<'a, 'gcx, 'tcx>) -> Ty<'tcx> {
        tcx.mk_mach_float(*self)
    }
}

impl UnifyKey for ty::TyVid {
    type Value = ();
    fn index(&self) -> u32 { self.index }
    fn from_index(i: u32) -> ty::TyVid { ty::TyVid { index: i } }
    fn tag(_: Option<ty::TyVid>) -> &'static str { "TyVid" }
}
