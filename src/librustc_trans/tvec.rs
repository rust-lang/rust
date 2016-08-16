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

pub fn iter_vec_raw<'blk, 'tcx, F>(bcx: Block<'blk, 'tcx>,
                                   data_ptr: ValueRef,
                                   unit_ty: Ty<'tcx>,
                                   len: ValueRef,
                                   f: F)
                                   -> Block<'blk, 'tcx> where
    F: FnOnce(Block<'blk, 'tcx>, ValueRef, Ty<'tcx>) -> Block<'blk, 'tcx>,
{
    let _icx = push_ctxt("tvec::iter_vec_raw");
    let fcx = bcx.fcx;

    if type_is_zero_size(bcx.ccx(), unit_ty) {
        // Special-case vectors with elements of size 0  so they don't go out of bounds (#9890)
        if bcx.unreachable.get() {
            return bcx;
        }

        let loop_bcx = fcx.new_block("expr_repeat");
        let next_bcx = fcx.new_block("expr_repeat: next");

        Br(bcx, loop_bcx.llbb, DebugLoc::None);

        let loop_counter = Phi(loop_bcx, bcx.ccx().int_type(),
                            &[C_uint(bcx.ccx(), 0 as usize)], &[bcx.llbb]);

        let bcx = loop_bcx;
        let bcx = f(bcx, data_ptr, unit_ty);
        let plusone = Add(bcx, loop_counter, C_uint(bcx.ccx(), 1usize), DebugLoc::None);
        AddIncomingToPhi(loop_counter, plusone, bcx.llbb);

        let cond_val = ICmp(bcx, llvm::IntULT, plusone, len, DebugLoc::None);
        CondBr(bcx, cond_val, loop_bcx.llbb, next_bcx.llbb, DebugLoc::None);

        next_bcx
    } else {
        // Calculate the last pointer address we want to handle.
        let data_end_ptr = InBoundsGEP(bcx, data_ptr, &[len]);

        // Now perform the iteration.
        let header_bcx = fcx.new_block("iter_vec_loop_header");
        Br(bcx, header_bcx.llbb, DebugLoc::None);
        let data_ptr =
            Phi(header_bcx, val_ty(data_ptr), &[data_ptr], &[bcx.llbb]);
        let not_yet_at_end =
            ICmp(header_bcx, llvm::IntULT, data_ptr, data_end_ptr, DebugLoc::None);
        let body_bcx = fcx.new_block("iter_vec_loop_body");
        let next_bcx = fcx.new_block("iter_vec_next");
        CondBr(header_bcx, not_yet_at_end, body_bcx.llbb, next_bcx.llbb, DebugLoc::None);
        let body_bcx = f(body_bcx, data_ptr, unit_ty);
        AddIncomingToPhi(data_ptr, InBoundsGEP(body_bcx, data_ptr,
                                               &[C_int(bcx.ccx(), 1)]),
                         body_bcx.llbb);
        Br(body_bcx, header_bcx.llbb, DebugLoc::None);
        next_bcx
    }
}
