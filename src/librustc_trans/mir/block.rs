// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use llvm::{ValueRef, BasicBlockRef};
use rustc::mir;
use rustc::middle::const_val::ConstInt;
use base::{self, Lifetime};
use builder::Builder;
use common::Funclet;
use machine::llalign_of_min;
use type_::Type;

use rustc_data_structures::indexed_vec::IndexVec;

use super::{MirContext, LocalRef};
use super::analyze::CleanupKind;
use super::constant::Const;
use super::lvalue::Alignment;
use super::operand::OperandRef;
use super::operand::OperandValue::{Pair, Ref, Immediate};

impl<'a, 'tcx> MirContext<'a, 'tcx> {
    pub fn trans_block(&mut self, bb: mir::Block,
        funclets: &IndexVec<mir::Block, Option<Funclet>>) {
        let mut bcx = self.get_builder(bb);
        let data = &self.mir[bb];

        debug!("trans_block({:?}={:?})", bb, data);

        let funclet = match self.cleanup_kinds[bb] {
            CleanupKind::Internal { funclet } => funclets[funclet].as_ref(),
            _ => funclets[bb].as_ref(),
        };

        // Create the cleanup bundle, if needed.
        let cleanup_pad = funclet.map(|lp| lp.cleanuppad());
        let cleanup_bundle = funclet.map(|l| l.bundle());

        let funclet_br = |this: &Self, bcx: Builder, bb: mir::Block| {
            let lltarget = this.blocks[bb];
            if let Some(cp) = cleanup_pad {
                match this.cleanup_kinds[bb] {
                    CleanupKind::Funclet => {
                        // micro-optimization: generate a `ret` rather than a jump
                        // to a return block
                        bcx.cleanup_ret(cp, Some(lltarget));
                    }
                    CleanupKind::Internal { .. } => bcx.br(lltarget),
                    CleanupKind::NotCleanup => bug!("jump from cleanup bb to bb {:?}", bb)
                }
            } else {
                bcx.br(lltarget);
            }
        };

        let llblock = |this: &mut Self, target: mir::Block| {
            let lltarget = this.blocks[target];

            if let Some(cp) = cleanup_pad {
                match this.cleanup_kinds[target] {
                    CleanupKind::Funclet => {
                        // MSVC cross-funclet jump - need a trampoline

                        debug!("llblock: creating cleanup trampoline for {:?}", target);
                        let name = &format!("{:?}_cleanup_trampoline_{:?}", bb, target);
                        let trampoline = this.new_block(name);
                        trampoline.cleanup_ret(cp, Some(lltarget));
                        trampoline.llbb()
                    }
                    CleanupKind::Internal { .. } => lltarget,
                    CleanupKind::NotCleanup =>
                        bug!("jump from cleanup bb {:?} to bb {:?}", bb, target)
                }
            } else {
                if let (CleanupKind::NotCleanup, CleanupKind::Funclet) =
                    (this.cleanup_kinds[bb], this.cleanup_kinds[target])
                {
                    // jump *into* cleanup - need a landing pad if GNU
                    this.landing_pad_to(target)
                } else {
                    lltarget
                }
            }
        };

        for statement in &data.statements {
            bcx = self.trans_statement(bcx, statement, cleanup_bundle);
        }

        let terminator = data.terminator();
        debug!("trans_block: terminator: {:?}", terminator);

        self.set_debug_loc(&bcx, terminator.source_info);
        match terminator.kind {
            mir::TerminatorKind::Resume => {
                if let Some(cleanup_pad) = cleanup_pad {
                    bcx.cleanup_ret(cleanup_pad, None);
                } else {
                    let ps = self.get_personality_slot(&bcx);
                    let lp = bcx.load(ps, None);
                    Lifetime::End.call(&bcx, ps);
                    if !bcx.sess().target.target.options.custom_unwind_resume {
                        bcx.resume(lp);
                    } else {
                        let exc_ptr = bcx.extract_value(lp, 0);
                        bcx.call(bcx.ccx.eh_unwind_resume(), &[exc_ptr], cleanup_bundle);
                        bcx.unreachable();
                    }
                }
            }

            mir::TerminatorKind::Goto { target } => {
                funclet_br(self, bcx, target);
            }

            mir::TerminatorKind::SwitchInt { ref discr, switch_ty, ref values, ref targets } => {
                let discr = self.trans_operand(&bcx, discr);
                if switch_ty == bcx.tcx().types.bool {
                    let lltrue = llblock(self, targets[0]);
                    let llfalse = llblock(self, targets[1]);
                    if let [ConstInt::U8(0)] = values[..] {
                        bcx.cond_br(discr.immediate(), llfalse, lltrue);
                    } else {
                        bcx.cond_br(discr.immediate(), lltrue, llfalse);
                    }
                } else {
                    let (otherwise, targets) = targets.split_last().unwrap();
                    let switch = bcx.switch(discr.immediate(),
                                            llblock(self, *otherwise), values.len());
                    for (value, target) in values.iter().zip(targets) {
                        let val = Const::from_constint(bcx.ccx, value);
                        let llbb = llblock(self, *target);
                        bcx.add_case(switch, val.llval, llbb)
                    }
                }
            }

            mir::TerminatorKind::Return => {
                let ret = self.fn_ty.ret;
                if ret.is_ignore() || ret.is_indirect() {
                    bcx.ret_void();
                    return;
                }

                let llval = if let Some(cast_ty) = ret.cast {
                    let op = match self.locals[mir::RETURN_POINTER] {
                        LocalRef::Operand(Some(op)) => op,
                        LocalRef::Operand(None) => bug!("use of return before def"),
                        LocalRef::Lvalue(tr_lvalue) => {
                            OperandRef {
                                val: Ref(tr_lvalue.llval, tr_lvalue.alignment),
                                ty: tr_lvalue.ty.to_ty(bcx.tcx())
                            }
                        }
                    };
                    let llslot = match op.val {
                        Immediate(_) | Pair(..) => {
                            let llscratch = bcx.alloca(ret.original_ty, "ret");
                            self.store_operand(&bcx, llscratch, None, op);
                            llscratch
                        }
                        Ref(llval, align) => {
                            assert_eq!(align, Alignment::AbiAligned,
                                       "return pointer is unaligned!");
                            llval
                        }
                    };
                    let load = bcx.load(
                        bcx.pointercast(llslot, cast_ty.ptr_to()),
                        Some(llalign_of_min(bcx.ccx, ret.ty)));
                    load
                } else {
                    let op = self.trans_consume(&bcx, &mir::Lvalue::Local(mir::RETURN_POINTER));
                    if let Ref(llval, align) = op.val {
                        base::load_ty(&bcx, llval, align, op.ty)
                    } else {
                        op.pack_if_pair(&bcx).immediate()
                    }
                };
                bcx.ret(llval);
            }

            mir::TerminatorKind::Unreachable => {
                bcx.unreachable();
            }

            mir::TerminatorKind::Drop { ref location, target, unwind } => {
                bcx = self.trans_drop(
                    bcx, location, unwind, cleanup_bundle, terminator.source_info
                );
                funclet_br(self, bcx, target);
            }

            mir::TerminatorKind::DropAndReplace { .. } => {
                bug!("undesugared DropAndReplace in trans: {:?}", data);
            }

        }
    }

    /// Return the landingpad wrapper around the given basic block
    ///
    /// No-op in MSVC SEH scheme.
    pub fn landing_pad_to(&mut self, target_bb: mir::Block) -> BasicBlockRef {
        if let Some(block) = self.landing_pads[target_bb] {
            return block;
        }

        let block = self.blocks[target_bb];
        let landing_pad = self.landing_pad_uncached(block);
        self.landing_pads[target_bb] = Some(landing_pad);
        landing_pad
    }

    fn landing_pad_uncached(&mut self, target_bb: BasicBlockRef) -> BasicBlockRef {
        if base::wants_msvc_seh(self.ccx.sess()) {
            return target_bb;
        }

        let bcx = self.new_block("cleanup");

        let ccx = bcx.ccx;
        let llpersonality = self.ccx.eh_personality();
        let llretty = Type::struct_(ccx, &[Type::i8p(ccx), Type::i32(ccx)], false);
        let llretval = bcx.landing_pad(llretty, llpersonality, 1, self.llfn);
        bcx.set_cleanup(llretval);
        let slot = self.get_personality_slot(&bcx);
        bcx.store(llretval, slot, None);
        bcx.br(target_bb);
        bcx.llbb()
    }

    pub fn new_block(&self, name: &str) -> Builder<'a, 'tcx> {
        Builder::new_block(self.ccx, self.llfn, name)
    }

    pub fn get_builder(&self, bb: mir::Block) -> Builder<'a, 'tcx> {
        let builder = Builder::with_ccx(self.ccx);
        builder.position_at_end(self.blocks[bb]);
        builder
    }

    fn get_personality_slot(&mut self, bcx: &Builder<'a, 'tcx>) -> ValueRef {
        let ccx = bcx.ccx;
        if let Some(slot) = self.llpersonalityslot {
            slot
        } else {
            let llretty = Type::struct_(ccx, &[Type::i8p(ccx), Type::i32(ccx)], false);
            let slot = bcx.alloca(llretty, "personalityslot");
            self.llpersonalityslot = Some(slot);
            Lifetime::Start.call(bcx, slot);
            slot
        }
    }
}
