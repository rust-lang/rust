// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use rustc::ty::{self, TyCtxt};
use rustc::ty::util::IntTypeExt;
use rustc_const_math::ConstInt;

#[derive(Debug, Eq, PartialEq, Copy, Clone)]
pub struct Disr(pub u64);

impl Disr {
    pub fn for_variant(tcx: TyCtxt, def: &ty::AdtDef, variant_index: usize) -> Self {
        let v = if def.is_enum() {
            def.discriminants(tcx)[variant_index]
        } else {
            def.repr.discr_type().initial_discriminant(tcx.global_tcx())
        };
        Disr::from(v)
    }

    pub fn wrapping_add(self, other: Self) -> Self {
        Disr(self.0.wrapping_add(other.0))
    }
}

impl ::std::ops::BitAnd for Disr {
    type Output = Disr;
    fn bitand(self, other: Self) -> Self {
        Disr(self.0 & other.0)
    }
}

impl From<ConstInt> for Disr {
    fn from(i: ConstInt) -> Disr {
        // FIXME: what if discr has 128 bit discr?
        Disr(i.to_u128_unchecked() as u64)
    }
}

impl From<usize> for Disr {
    fn from(i: usize) -> Disr {
        Disr(i as u64)
    }
}

impl PartialOrd for Disr {
    fn partial_cmp(&self, other: &Disr) -> Option<::std::cmp::Ordering> {
        self.0.partial_cmp(&other.0)
    }
}

impl Ord for Disr {
    fn cmp(&self, other: &Disr) -> ::std::cmp::Ordering {
        self.0.cmp(&other.0)
    }
}
