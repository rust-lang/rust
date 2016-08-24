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
use build::*;
use common::*;
use debuginfo::DebugLoc;
use rustc::ty::Ty;

pub fn slice_for_each<'blk, 'tcx, F>(bcx: Block<'blk, 'tcx>,
                                     data_ptr: ValueRef,
                                     unit_ty: Ty<'tcx>,
                                     len: ValueRef,
                                     f: F)
                                     -> Block<'blk, 'tcx> where
    F: FnOnce(Block<'blk, 'tcx>, ValueRef) -> Block<'blk, 'tcx>,
{
    let _icx = push_ctxt("tvec::slice_for_each");
    let fcx = bcx.fcx;

    // Special-case vectors with elements of size 0  so they don't go out of bounds (#9890)
    let zst = type_is_zero_size(bcx.ccx(), unit_ty);
    let add = |bcx, a, b| if zst {
        Add(bcx, a, b, DebugLoc::None)
    } else {
        InBoundsGEP(bcx, a, &[b])
    };

    let header_bcx = fcx.new_block("slice_loop_header");
    let body_bcx = fcx.new_block("slice_loop_body");
    let next_bcx = fcx.new_block("slice_loop_next");

    let start = if zst {
        C_uint(bcx.ccx(), 0 as usize)
    } else {
        data_ptr
    };
    let end = add(bcx, start, len);

    Br(bcx, header_bcx.llbb, DebugLoc::None);
    let current = Phi(header_bcx, val_ty(start), &[start], &[bcx.llbb]);

    let keep_going =
        ICmp(header_bcx, llvm::IntULT, current, end, DebugLoc::None);
    CondBr(header_bcx, keep_going, body_bcx.llbb, next_bcx.llbb, DebugLoc::None);

    let body_bcx = f(body_bcx, if zst { data_ptr } else { current });
    let next = add(body_bcx, current, C_uint(bcx.ccx(), 1usize));
    AddIncomingToPhi(current, next, body_bcx.llbb);
    Br(body_bcx, header_bcx.llbb, DebugLoc::None);
    next_bcx
}
