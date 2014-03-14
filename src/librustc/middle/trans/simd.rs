// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use middle::ty;
use middle::ty::{simd_size};
use middle::trans::build::GEPi;
use middle::trans::common::{Block, C_uint};
use lib::llvm::ValueRef;

pub fn get_base_and_len(bcx: &Block,
                        llval: ValueRef,
                        ty: ty::t)
                        -> (ValueRef, ValueRef) {
    (GEPi(bcx, llval, [0, 0]),
     C_uint(bcx.ccx(), simd_size(bcx.ccx().tcx, ty)))
}
