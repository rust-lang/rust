// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(non_camel_case_types)]

use llvm;
use llvm::ValueRef;
use base::*;
use common::*;
use rustc::ty::Ty;

pub fn slice_for_each<'blk, 'tcx, F>(bcx: BlockAndBuilder<'blk, 'tcx>,
                                     data_ptr: ValueRef,
                                     unit_ty: Ty<'tcx>,
                                     len: ValueRef,
                                     f: F)
                                     -> BlockAndBuilder<'blk, 'tcx>
    where F: FnOnce(BlockAndBuilder<'blk, 'tcx>, ValueRef) -> BlockAndBuilder<'blk, 'tcx>,
{
    let _icx = push_ctxt("tvec::slice_for_each");
    let fcx = bcx.fcx();

    // Special-case vectors with elements of size 0  so they don't go out of bounds (#9890)
    let zst = type_is_zero_size(bcx.ccx(), unit_ty);
    let add = |bcx: &BlockAndBuilder, a, b| if zst {
        bcx.add(a, b)
    } else {
        bcx.inbounds_gep(a, &[b])
    };

    let body_bcx = fcx.build_new_block("slice_loop_body");
    let next_bcx = fcx.build_new_block("slice_loop_next");
    let header_bcx = fcx.build_new_block("slice_loop_header");

    let start = if zst {
        C_uint(bcx.ccx(), 0usize)
    } else {
        data_ptr
    };
    let end = add(&bcx, start, len);

    bcx.br(header_bcx.llbb());
    let current = header_bcx.phi(val_ty(start), &[start], &[bcx.llbb()]);

    let keep_going = header_bcx.icmp(llvm::IntNE, current, end);
    header_bcx.cond_br(keep_going, body_bcx.llbb(), next_bcx.llbb());

    let body_bcx = f(body_bcx, if zst { data_ptr } else { current });
    // FIXME(simulacrum): The code below is identical to the closure (add) above, but using the
    // closure doesn't compile due to body_bcx still being borrowed when dropped.
    let next = if zst {
        body_bcx.add(current, C_uint(bcx.ccx(), 1usize))
    } else {
        body_bcx.inbounds_gep(current, &[C_uint(bcx.ccx(), 1usize)])
    };
    body_bcx.add_incoming_to_phi(current, next, body_bcx.llbb());
    body_bcx.br(header_bcx.llbb());
    next_bcx
}
