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
use middle::infer;
use middle::ty;
use rustc::mir::repr as mir;
use trans::adt;
use trans::base;
use trans::build;
use trans::common::{self, Block};
use trans::debuginfo::DebugLoc;
use trans::foreign;
use trans::type_of;

use syntax::abi as synabi;

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

        debug!("trans_block: terminator: {:?}", data.terminator);

        match data.terminator {
            mir::Terminator::Goto { target } => {
                build::Br(bcx, self.llblock(target), DebugLoc::None)
            }

            mir::Terminator::Panic { .. } => {
                unimplemented!()
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
                // FIXME it might be nice to have just one such block (created lazilly), we could
                // store it in the "MIR trans" state.
                let unreachable_blk = bcx.fcx.new_temp_block("enum-variant-unreachable");
                build::Unreachable(unreachable_blk);

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

            mir::Terminator::Diverge => {
                if let Some(llpersonalityslot) = self.llpersonalityslot {
                    let lp = build::Load(bcx, llpersonalityslot);
                    // FIXME(lifetime) base::call_lifetime_end(bcx, self.personality);
                    build::Resume(bcx, lp);
                } else {
                    // This fn never encountered anything fallible, so
                    // a Diverge cannot actually happen. Note that we
                    // do a total hack to ensure that we visit the
                    // DIVERGE block last.
                    build::Unreachable(bcx);
                }
            }

            mir::Terminator::Return => {
                let return_ty = bcx.monomorphize(&self.mir.return_ty);
                base::build_return_block(bcx.fcx, bcx, return_ty, DebugLoc::None);
            }

            mir::Terminator::Call { ref data, targets } => {
                // The location we'll write the result of the call into.
                let call_dest = self.trans_lvalue(bcx, &data.destination);

                // Create the callee. This will always be a fn
                // ptr and hence a kind of scalar.
                let callee = self.trans_operand(bcx, &data.func);
                let (abi, ret_ty) = if let ty::TyBareFn(_, ref f) = callee.ty.sty {
                    let sig = bcx.tcx().erase_late_bound_regions(&f.sig);
                    let sig = infer::normalize_associated_type(bcx.tcx(), &sig);
                    (f.abi, sig.output)
                } else {
                    panic!("trans_block: expected TyBareFn as callee");
                };

                // Have we got a 'Rust' function?
                let is_rust_fn = abi == synabi::Rust || abi == synabi::RustCall;

                // The arguments we'll be passing
                let mut llargs = Vec::with_capacity(data.args.len() + 1);
                // and their Rust types (formal args only so not outptr)
                let mut arg_tys = Vec::with_capacity(data.args.len());

                // Does the fn use an outptr? If so, that's the first arg.
                if let (true, ty::FnConverging(ret_ty)) = (is_rust_fn, ret_ty) {
                    if type_of::return_uses_outptr(bcx.ccx(), ret_ty) {
                        llargs.push(call_dest.llval);
                    }
                }

                // Process the rest of the args.
                for arg in &data.args {
                    let arg_op = self.trans_operand(bcx, arg);
                    arg_tys.push(arg_op.ty);
                    match arg_op.val {
                        Ref(llval) | Immediate(llval) => llargs.push(llval),
                        FatPtr(base, extra) => {
                            // The two words in a fat ptr are passed separately
                            llargs.push(base);
                            llargs.push(extra);
                        }
                    }
                }

                // FIXME: Handle panics
                //let panic_bb = self.llblock(targets.1);
                //self.make_landing_pad(panic_bb);

                // Do the actual call.
                if is_rust_fn {
                    let (llret, b) = base::invoke(bcx,
                                                  callee.immediate(),
                                                  &llargs,
                                                  callee.ty,
                                                  DebugLoc::None);
                    bcx = b;

                    // Copy the return value into the destination.
                    if let ty::FnConverging(ret_ty) = ret_ty {
                        if !type_of::return_uses_outptr(bcx.ccx(), ret_ty) &&
                           !common::type_is_zero_size(bcx.ccx(), ret_ty) {
                            base::store_ty(bcx, llret, call_dest.llval, ret_ty);
                        }
                    }
                } else {
                    bcx = foreign::trans_native_call(bcx,
                                                     callee.ty,
                                                     callee.immediate(),
                                                     call_dest.llval,
                                                     &llargs,
                                                     arg_tys,
                                                     DebugLoc::None);
                }

                build::Br(bcx, self.llblock(targets.0), DebugLoc::None)
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
