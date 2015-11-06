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
use rustc_mir::repr as mir;
use trans::base;
use trans::build;
use trans::common::Block;
use trans::debuginfo::DebugLoc;

use super::MirContext;

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

            mir::Terminator::If { ref cond, targets: [true_bb, false_bb] } => {
                let cond = self.trans_operand(bcx, cond);
                let lltrue = self.llblock(true_bb);
                let llfalse = self.llblock(false_bb);
                build::CondBr(bcx, cond.llval, lltrue, llfalse, DebugLoc::None);
            }

            mir::Terminator::Switch { .. } => {
                unimplemented!()
            }

            mir::Terminator::SwitchInt { .. } => {
                unimplemented!()
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

            mir::Terminator::Call { .. } => {
                unimplemented!()
                //let llbb = unimplemented!(); // self.make_landing_pad(panic_bb);
                //
                //let tr_dest = self.trans_lvalue(bcx, &data.destination);
                //
                //// Create the callee. This will always be a fn
                //// ptr and hence a kind of scalar.
                //let callee = self.trans_operand(bcx, &data.func);
                //
                //// Process the arguments.
                //
                //let args = unimplemented!();
                //
                //callee::trans_call_inner(bcx,
                //                         DebugLoc::None,
                //                         |bcx, _| Callee {
                //                             bcx: bcx,
                //                             data: CalleeData::Fn(callee.llval),
                //                             ty: callee.ty,
                //                         },
                //                         args,
                //                         Some(Dest::SaveIn(tr_dest.llval)));
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
