// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use llvm::ValueRef;
use rustc::middle::ty::Ty;
use rustc::mir::repr as mir;
use trans::adt;
use trans::base;
use trans::build;
use trans::common::{self, Block};
use trans::debuginfo::DebugLoc;
use trans::glue;
use trans::machine;
use trans::type_of;
use trans::type_::Type;

use super::{MirContext, TempRef};

impl<'bcx, 'tcx> MirContext<'bcx, 'tcx> {
    pub fn trans_drop(&mut self,
                      bcx: Block<'bcx, 'tcx>,
                      value: &mir::Lvalue<'tcx>,
                      target: mir::BasicBlock,
                      unwind: Option<mir::BasicBlock>) {
        let lvalue = self.trans_lvalue(bcx, value);
        let ty = lvalue.ty.to_ty(bcx.tcx());
        // Double check for necessity to drop
        if !glue::type_needs_drop(bcx.tcx(), ty) {
            build::Br(bcx, self.llblock(target), DebugLoc::None);
            return;
        }
        let drop_fn = glue::get_drop_glue(bcx.ccx(), ty);
        let drop_ty = glue::get_drop_glue_type(bcx.ccx(), ty);
        let llvalue = if drop_ty != ty {
            build::PointerCast(bcx, lvalue.llval,
                               type_of::type_of(bcx.ccx(), drop_ty).ptr_to())
        } else {
            lvalue.llval
        };
        if let Some(unwind) = unwind {
            // block cannot be cleanup in this case, so a regular block is fine
            let intermediate_bcx = bcx.fcx.new_block("", None);
            let uwbcx = self.bcx(unwind);
            let unwind = self.make_landing_pad(uwbcx);
            // FIXME: it could be possible to do zeroing before invoking here if the drop glue
            // didn’t code in the checks inside itself.
            build::Invoke(bcx,
                          drop_fn,
                          &[llvalue],
                          intermediate_bcx.llbb,
                          unwind.llbb,
                          None,
                          DebugLoc::None);
            // FIXME: perhaps we also should fill inside failed branch? We do not want to re-drop a
            // failed drop again by mistake. (conflicts with MSVC SEH if we don’t want to introduce
            // a heap of hacks)
            self.drop_fill(intermediate_bcx, lvalue.llval, ty);
            build::Br(intermediate_bcx, self.llblock(target), DebugLoc::None);
        } else {
            build::Call(bcx, drop_fn, &[llvalue], None, DebugLoc::None);
            self.drop_fill(bcx, lvalue.llval, ty);
            build::Br(bcx, self.llblock(target), DebugLoc::None);
        }
    }

    pub fn drop_fill(&mut self, bcx: Block<'bcx, 'tcx>, value: ValueRef, ty: Ty<'tcx>) {
        let llty = type_of::type_of(bcx.ccx(), ty);
        let llptr = build::PointerCast(bcx, value, Type::i8(bcx.ccx()).ptr_to());
        let filling = common::C_u8(bcx.ccx(), adt::DTOR_DONE);
        let size = machine::llsize_of(bcx.ccx(), llty);
        let align = common::C_u32(bcx.ccx(), machine::llalign_of_min(bcx.ccx(), llty));
        base::call_memset(&build::B(bcx), llptr, filling, size, align, false);
    }

    pub fn set_operand_dropped(&mut self,
                               bcx: Block<'bcx, 'tcx>,
                               operand: &mir::Operand<'tcx>) {
        match *operand {
            mir::Operand::Constant(_) => return,
            mir::Operand::Consume(ref lvalue) => {
                if let mir::Lvalue::Temp(idx) = *lvalue {
                    if let TempRef::Operand(..) = self.temps[idx as usize] {
                        return // we do not handle these, should we?
                    }
                }
                let lvalue = self.trans_lvalue(bcx, lvalue);
                let ty = lvalue.ty.to_ty(bcx.tcx());
                if !glue::type_needs_drop(bcx.tcx(), ty) ||
                    common::type_is_fat_ptr(bcx.tcx(), ty) {
                    return
                } else {
                    self.drop_fill(bcx, lvalue.llval, ty);
                }
            }
        }
    }
}
