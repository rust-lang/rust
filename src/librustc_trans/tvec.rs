// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use llvm;
use builder::Builder;
use llvm::ValueRef;
use common::*;
use rustc::ty::Ty;

pub fn slice_for_each<'a, 'tcx, F>(
    bcx: &Builder<'a, 'tcx>,
    data_ptr: ValueRef,
    unit_ty: Ty<'tcx>,
    len: ValueRef,
    f: F
) -> Builder<'a, 'tcx> where F: FnOnce(&Builder<'a, 'tcx>, ValueRef) {
    // Special-case vectors with elements of size 0  so they don't go out of bounds (#9890)
    let zst = type_is_zero_size(bcx.ccx, unit_ty);
    let add = |bcx: &Builder, a, b| if zst {
        bcx.add(a, b)
    } else {
        bcx.inbounds_gep(a, &[b])
    };

    let body_bcx = bcx.build_sibling_block("slice_loop_body");
    let next_bcx = bcx.build_sibling_block("slice_loop_next");
    let header_bcx = bcx.build_sibling_block("slice_loop_header");

    let start = if zst {
        C_uint(bcx.ccx, 0usize)
    } else {
        data_ptr
    };
    let end = add(&bcx, start, len);

    bcx.br(header_bcx.llbb());
    let current = header_bcx.phi(val_ty(start), &[start], &[bcx.llbb()]);

    let keep_going = header_bcx.icmp(llvm::IntNE, current, end);
    header_bcx.cond_br(keep_going, body_bcx.llbb(), next_bcx.llbb());

    f(&body_bcx, if zst { data_ptr } else { current });
    let next = add(&body_bcx, current, C_uint(bcx.ccx, 1usize));
    header_bcx.add_incoming_to_phi(current, next, body_bcx.llbb());
    body_bcx.br(header_bcx.llbb());
    next_bcx
}
