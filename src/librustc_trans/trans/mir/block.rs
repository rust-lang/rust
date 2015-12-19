// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use llvm::BasicBlockRef;
use rustc::mir::repr as mir;
use trans::adt;
use trans::base;
use trans::build;
use trans::attributes;
use trans::common::{self, Block};
use trans::debuginfo::DebugLoc;
use trans::type_of;
use trans::type_::Type;

use super::MirContext;
use super::operand::OperandValue::{FatPtr, Immediate, Ref};

impl<'bcx, 'tcx> MirContext<'bcx, 'tcx> {
    pub fn trans_block(&mut self, bb: mir::BasicBlock) {
        debug!("trans_block({:?})", bb);

        let mut bcx = self.bcx(bb);
        let data = self.mir.basic_block_data(bb);

        for statement in &data.statements {
            bcx = self.trans_statement(bcx, statement);
        }

        debug!("trans_block: terminator: {:?}", data.terminator());

        match *data.terminator() {
            mir::Terminator::Goto { target } => {
                build::Br(bcx, self.llblock(target), DebugLoc::None)
            }

            mir::Terminator::If { ref cond, targets: (true_bb, false_bb) } => {
                let cond = self.trans_operand(bcx, cond);
                let lltrue = self.llblock(true_bb);
                let llfalse = self.llblock(false_bb);
                build::CondBr(bcx, cond.immediate(), lltrue, llfalse, DebugLoc::None);
            }

            mir::Terminator::Switch { ref discr, ref adt_def, ref targets } => {
                let adt_ty = bcx.tcx().lookup_item_type(adt_def.did).ty;
                let represented_ty = adt::represent_type(bcx.ccx(), adt_ty);

                let discr_lvalue = self.trans_lvalue(bcx, discr);
                let discr = adt::trans_get_discr(bcx, &represented_ty, discr_lvalue.llval, None);

                // The else branch of the Switch can't be hit, so branch to an unreachable
                // instruction so LLVM knows that
                let unreachable_blk = self.unreachable_block();

                let switch = build::Switch(bcx, discr, unreachable_blk.llbb, targets.len());
                assert_eq!(adt_def.variants.len(), targets.len());
                for (adt_variant, target) in adt_def.variants.iter().zip(targets) {
                    let llval = adt::trans_case(bcx, &*represented_ty, adt_variant.disr_val);
                    let llbb = self.llblock(*target);

                    build::AddCase(switch, llval, llbb)
                }
            }

            mir::Terminator::SwitchInt { ref discr, switch_ty, ref values, ref targets } => {
                let (otherwise, targets) = targets.split_last().unwrap();
                let discr = build::Load(bcx, self.trans_lvalue(bcx, discr).llval);
                let switch = build::Switch(bcx, discr, self.llblock(*otherwise), values.len());
                for (value, target) in values.iter().zip(targets) {
                    let llval = self.trans_constval(bcx, value, switch_ty).immediate();
                    let llbb = self.llblock(*target);
                    build::AddCase(switch, llval, llbb)
                }
            }

            mir::Terminator::Resume => {
                if let Some(personalityslot) = self.llpersonalityslot {
                    let lp = build::Load(bcx, personalityslot);
                    base::call_lifetime_end(bcx, personalityslot);
                    build::Resume(bcx, lp);
                } else {
                    panic!("resume terminator without personality slot set")
                }
            }

            mir::Terminator::Return => {
                let return_ty = bcx.monomorphize(&self.mir.return_ty);
                base::build_return_block(bcx.fcx, bcx, return_ty, DebugLoc::None);
            }

            mir::Terminator::Call { ref func, ref args, ref destination, ref targets } => {
                // The location we'll write the result of the call into.
                let call_dest = self.trans_lvalue(bcx, destination);
                let ret_ty = call_dest.ty.to_ty(bcx.tcx());
                // Create the callee. This will always be a fn
                // ptr and hence a kind of scalar.
                let callee = self.trans_operand(bcx, func);

                // Does the fn use an outptr? If so, we have an extra first argument.
                let return_outptr = type_of::return_uses_outptr(bcx.ccx(), ret_ty);
                // The arguments we'll be passing.
                let mut llargs = if return_outptr {
                    let mut vec = Vec::with_capacity(args.len() + 1);
                    vec.push(call_dest.llval);
                    vec
                } else {
                    Vec::with_capacity(args.len())
                };

                // Process the rest of the args.
                for arg in args {
                    let arg_op = self.trans_operand(bcx, arg);
                    match arg_op.val {
                        Ref(llval) | Immediate(llval) => llargs.push(llval),
                        FatPtr(base, extra) => {
                            // The two words in a fat ptr are passed separately
                            llargs.push(base);
                            llargs.push(extra);
                        }
                    }
                }

                let debugloc = DebugLoc::None;
                let attrs = attributes::from_fn_type(bcx.ccx(), callee.ty);
                match *targets {
                    mir::CallTargets::Return(ret) => {
                        let llret = build::Call(bcx,
                                                callee.immediate(),
                                                &llargs[..],
                                                Some(attrs),
                                                debugloc);
                        if !return_outptr && !common::type_is_zero_size(bcx.ccx(), ret_ty) {
                            base::store_ty(bcx, llret, call_dest.llval, ret_ty);
                        }
                        build::Br(bcx, self.llblock(ret), debugloc)
                    }
                    mir::CallTargets::WithCleanup((ret, cleanup)) => {
                        let landingpad = self.make_landing_pad(cleanup);
                        build::Invoke(bcx,
                                      callee.immediate(),
                                      &llargs[..],
                                      self.llblock(ret),
                                      landingpad.llbb,
                                      Some(attrs),
                                      debugloc);
                        if !return_outptr && !common::type_is_zero_size(bcx.ccx(), ret_ty) {
                            // FIXME: What do we do here?
                            unimplemented!()
                        }
                    }
                }
            },

            mir::Terminator::DivergingCall { ref func, ref args, ref cleanup } => {
                let callee = self.trans_operand(bcx, func);
                let mut llargs = Vec::with_capacity(args.len());
                for arg in args {
                    match self.trans_operand(bcx, arg).val {
                        Ref(llval) | Immediate(llval) => llargs.push(llval),
                        FatPtr(b, e) => {
                            llargs.push(b);
                            llargs.push(e);
                        }
                    }
                }
                let debugloc = DebugLoc::None;
                let attrs = attributes::from_fn_type(bcx.ccx(), callee.ty);
                match *cleanup {
                    None => {
                        build::Call(bcx, callee.immediate(), &llargs[..], Some(attrs), debugloc);
                        build::Unreachable(bcx);
                    }
                    Some(cleanup) => {
                        let landingpad = self.make_landing_pad(cleanup);
                        let unreachable = self.unreachable_block();
                        build::Invoke(bcx,
                                      callee.immediate(),
                                      &llargs[..],
                                      unreachable.llbb,
                                      landingpad.llbb,
                                      Some(attrs),
                                      debugloc);
                    }
                }
            }
        }
    }

    fn make_landing_pad(&mut self, cleanup: mir::BasicBlock) -> Block<'bcx, 'tcx> {
        let bcx = self.bcx(cleanup).fcx.new_block(true, "cleanup", None);
        let ccx = bcx.ccx();
        let llpersonality = bcx.fcx.eh_personality();
        let llretty = Type::struct_(ccx, &[Type::i8p(ccx), Type::i32(ccx)], false);
        let llretval = build::LandingPad(bcx, llretty, llpersonality, 1);
        build::SetCleanup(bcx, llretval);
        match self.llpersonalityslot {
            Some(slot) => build::Store(bcx, llretval, slot),
            None => {
                let personalityslot = base::alloca(bcx, llretty, "personalityslot");
                self.llpersonalityslot = Some(personalityslot);
                base::call_lifetime_start(bcx, personalityslot);
                build::Store(bcx, llretval, personalityslot)
            }
        };
        build::Br(bcx, self.llblock(cleanup), DebugLoc::None);
        bcx
    }

    fn unreachable_block(&mut self) -> Block<'bcx, 'tcx> {
        match self.unreachable_block {
            Some(b) => b,
            None => {
                let bl = self.fcx.new_block(false, "unreachable", None);
                build::Unreachable(bl);
                self.unreachable_block = Some(bl);
                bl
            }
        }
    }

    fn bcx(&self, bb: mir::BasicBlock) -> Block<'bcx, 'tcx> {
        self.blocks[bb.index()]
    }

    fn llblock(&self, bb: mir::BasicBlock) -> BasicBlockRef {
        self.blocks[bb.index()].llbb
    }
}
