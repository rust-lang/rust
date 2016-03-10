// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use llvm::{BasicBlockRef, ValueRef, OperandBundleDef};
use rustc::middle::ty;
use rustc::mir::repr as mir;
use syntax::abi::Abi;
use trans::adt;
use trans::attributes;
use trans::base;
use trans::build;
use trans::callee::{Callee, Fn, Virtual};
use trans::common::{self, Block, BlockAndBuilder};
use trans::debuginfo::DebugLoc;
use trans::Disr;
use trans::foreign;
use trans::meth;
use trans::type_of;
use trans::glue;
use trans::type_::Type;

use super::{MirContext, drop};
use super::operand::OperandValue::{FatPtr, Immediate, Ref};

impl<'bcx, 'tcx> MirContext<'bcx, 'tcx> {
    pub fn trans_block(&mut self, bb: mir::BasicBlock) {
        debug!("trans_block({:?})", bb);

        let mut bcx = self.bcx(bb);
        let data = self.mir.basic_block_data(bb);

        // MSVC SEH bits
        let (cleanup_pad, cleanup_bundle) = if let Some((cp, cb)) = self.make_cleanup_pad(bb) {
            (Some(cp), Some(cb))
        } else {
            (None, None)
        };
        let funclet_br = |bcx: BlockAndBuilder, llbb: BasicBlockRef| if let Some(cp) = cleanup_pad {
            bcx.cleanup_ret(cp, Some(llbb));
        } else {
            bcx.br(llbb);
        };

        for statement in &data.statements {
            bcx = self.trans_statement(bcx, statement);
        }

        debug!("trans_block: terminator: {:?}", data.terminator());

        match *data.terminator() {
            mir::Terminator::Resume => {
                if let Some(cleanup_pad) = cleanup_pad {
                    bcx.cleanup_ret(cleanup_pad, None);
                } else {
                    let ps = self.get_personality_slot(&bcx);
                    let lp = bcx.load(ps);
                    bcx.with_block(|bcx| {
                        base::call_lifetime_end(bcx, ps);
                        base::trans_unwind_resume(bcx, lp);
                    });
                }
            }

            mir::Terminator::Goto { target } => {
                funclet_br(bcx, self.llblock(target));
            }

            mir::Terminator::If { ref cond, targets: (true_bb, false_bb) } => {
                let cond = self.trans_operand(&bcx, cond);
                let lltrue = self.llblock(true_bb);
                let llfalse = self.llblock(false_bb);
                bcx.cond_br(cond.immediate(), lltrue, llfalse);
            }

            mir::Terminator::Switch { ref discr, ref adt_def, ref targets } => {
                let discr_lvalue = self.trans_lvalue(&bcx, discr);
                let ty = discr_lvalue.ty.to_ty(bcx.tcx());
                let repr = adt::represent_type(bcx.ccx(), ty);
                let discr = bcx.with_block(|bcx|
                    adt::trans_get_discr(bcx, &repr, discr_lvalue.llval, None, true)
                );

                // The else branch of the Switch can't be hit, so branch to an unreachable
                // instruction so LLVM knows that
                let unreachable_blk = self.unreachable_block();
                let switch = bcx.switch(discr, unreachable_blk.llbb, targets.len());
                assert_eq!(adt_def.variants.len(), targets.len());
                for (adt_variant, target) in adt_def.variants.iter().zip(targets) {
                    let llval = bcx.with_block(|bcx|
                        adt::trans_case(bcx, &repr, Disr::from(adt_variant.disr_val))
                    );
                    let llbb = self.llblock(*target);
                    build::AddCase(switch, llval, llbb)
                }
            }

            mir::Terminator::SwitchInt { ref discr, switch_ty, ref values, ref targets } => {
                let (otherwise, targets) = targets.split_last().unwrap();
                let discr = bcx.load(self.trans_lvalue(&bcx, discr).llval);
                let switch = bcx.switch(discr, self.llblock(*otherwise), values.len());
                for (value, target) in values.iter().zip(targets) {
                    let llval = self.trans_constval(&bcx, value, switch_ty).immediate();
                    let llbb = self.llblock(*target);
                    build::AddCase(switch, llval, llbb)
                }
            }

            mir::Terminator::Return => {
                let return_ty = bcx.monomorphize(&self.mir.return_ty);
                bcx.with_block(|bcx| {
                    base::build_return_block(self.fcx, bcx, return_ty, DebugLoc::None);
                })
            }

            mir::Terminator::Drop { ref value, target, unwind } => {
                let lvalue = self.trans_lvalue(&bcx, value);
                let ty = lvalue.ty.to_ty(bcx.tcx());
                // Double check for necessity to drop
                if !glue::type_needs_drop(bcx.tcx(), ty) {
                    funclet_br(bcx, self.llblock(target));
                    return;
                }
                let drop_fn = glue::get_drop_glue(bcx.ccx(), ty);
                let drop_ty = glue::get_drop_glue_type(bcx.ccx(), ty);
                let llvalue = if drop_ty != ty {
                    bcx.pointercast(lvalue.llval, type_of::type_of(bcx.ccx(), drop_ty).ptr_to())
                } else {
                    lvalue.llval
                };
                if let Some(unwind) = unwind {
                    let uwbcx = self.bcx(unwind);
                    let unwind = self.make_landing_pad(uwbcx);
                    bcx.invoke(drop_fn,
                               &[llvalue],
                               self.llblock(target),
                               unwind.llbb(),
                               cleanup_bundle.as_ref(),
                               None);
                    self.bcx(target).at_start(|bcx| drop::drop_fill(bcx, lvalue.llval, ty));
                } else {
                    bcx.call(drop_fn, &[llvalue], cleanup_bundle.as_ref(), None);
                    drop::drop_fill(&bcx, lvalue.llval, ty);
                    funclet_br(bcx, self.llblock(target));
                }
            }

            mir::Terminator::Call { ref func, ref args, ref destination, ref cleanup } => {
                // Create the callee. This is a fn ptr or zero-sized and hence a kind of scalar.
                let callee = self.trans_operand(&bcx, func);
                let debugloc = DebugLoc::None;
                // The arguments we'll be passing. Plus one to account for outptr, if used.
                let mut llargs = Vec::with_capacity(args.len() + 1);
                // Types of the arguments. We do not preallocate, because this vector is only
                // filled when `is_foreign` is `true` and foreign calls are minority of the cases.
                let mut arg_tys = Vec::new();

                let (callee, fty) = match callee.ty.sty {
                    ty::TyFnDef(def_id, substs, f) => {
                        (Callee::def(bcx.ccx(), def_id, substs, callee.ty), f)
                    }
                    ty::TyFnPtr(f) => {
                        (Callee {
                            data: Fn(callee.immediate()),
                            ty: callee.ty
                        }, f)
                    }
                    _ => unreachable!("{} is not callable", callee.ty)
                };

                // We do not translate intrinsics here (they shouldn’t be functions)
                assert!(fty.abi != Abi::RustIntrinsic && fty.abi != Abi::PlatformIntrinsic);
                // Foreign-ABI functions are translated differently
                let is_foreign = fty.abi != Abi::Rust && fty.abi != Abi::RustCall;

                // Prepare the return value destination
                let (ret_dest_ty, must_copy_dest) = if let Some((ref d, _)) = *destination {
                    let dest = self.trans_lvalue(&bcx, d);
                    let ret_ty = dest.ty.to_ty(bcx.tcx());
                    if !is_foreign && type_of::return_uses_outptr(bcx.ccx(), ret_ty) {
                        llargs.push(dest.llval);
                        (Some((dest, ret_ty)), false)
                    } else {
                        (Some((dest, ret_ty)), !common::type_is_zero_size(bcx.ccx(), ret_ty))
                    }
                } else {
                    (None, false)
                };

                // Split the rust-call tupled arguments off.
                let (args, rest) = if fty.abi == Abi::RustCall && !args.is_empty() {
                    let (tup, args) = args.split_last().unwrap();
                    // we can reorder safely because of MIR
                    (args, self.trans_operand_untupled(&bcx, tup))
                } else {
                    (&args[..], vec![])
                };

                let datum = {
                    let mut arg_ops = args.iter().map(|arg| {
                        self.trans_operand(&bcx, arg)
                    }).chain(rest.into_iter());

                    // Get the actual pointer we can call.
                    // This can involve vtable accesses or reification.
                    let datum = if let Virtual(idx) = callee.data {
                        assert!(!is_foreign);

                        // Grab the first argument which is a trait object.
                        let vtable = match arg_ops.next().unwrap().val {
                            FatPtr(data, vtable) => {
                                llargs.push(data);
                                vtable
                            }
                            _ => unreachable!("expected FatPtr for Virtual call")
                        };

                        bcx.with_block(|bcx| {
                            meth::get_virtual_method(bcx, vtable, idx, callee.ty)
                        })
                    } else {
                        callee.reify(bcx.ccx())
                    };

                    // Process the rest of the args.
                    for operand in arg_ops {
                        match operand.val {
                            Ref(llval) | Immediate(llval) => llargs.push(llval),
                            FatPtr(b, e) => {
                                llargs.push(b);
                                llargs.push(e);
                            }
                        }
                        if is_foreign {
                            arg_tys.push(operand.ty);
                        }
                    }

                    datum
                };
                let attrs = attributes::from_fn_type(bcx.ccx(), datum.ty);

                // Many different ways to call a function handled here
                match (is_foreign, cleanup, destination) {
                    // The two cases below are the only ones to use LLVM’s `invoke`.
                    (false, &Some(cleanup), &None) => {
                        let cleanup = self.bcx(cleanup);
                        let landingpad = self.make_landing_pad(cleanup);
                        let unreachable_blk = self.unreachable_block();
                        bcx.invoke(datum.val,
                                   &llargs[..],
                                   unreachable_blk.llbb,
                                   landingpad.llbb(),
                                   cleanup_bundle.as_ref(),
                                   Some(attrs));
                        landingpad.at_start(|bcx| for op in args {
                            self.set_operand_dropped(bcx, op);
                        });
                    },
                    (false, &Some(cleanup), &Some((_, success))) => {
                        let cleanup = self.bcx(cleanup);
                        let landingpad = self.make_landing_pad(cleanup);
                        let invokeret = bcx.invoke(datum.val,
                                                   &llargs[..],
                                                   self.llblock(success),
                                                   landingpad.llbb(),
                                                   cleanup_bundle.as_ref(),
                                                   Some(attrs));
                        if must_copy_dest {
                            let (ret_dest, ret_ty) = ret_dest_ty
                                .expect("return destination and type not set");
                            // We translate the copy straight into the beginning of the target
                            // block.
                            self.bcx(success).at_start(|bcx| bcx.with_block( |bcx| {
                                base::store_ty(bcx, invokeret, ret_dest.llval, ret_ty);
                            }));
                        }
                        self.bcx(success).at_start(|bcx| for op in args {
                            self.set_operand_dropped(bcx, op);
                        });
                        landingpad.at_start(|bcx| for op in args {
                            self.set_operand_dropped(bcx, op);
                        });
                    },
                    (false, _, &None) => {
                        bcx.call(datum.val,
                                 &llargs[..],
                                 cleanup_bundle.as_ref(),
                                 Some(attrs));
                        // no need to drop args, because the call never returns
                        bcx.unreachable();
                    }
                    (false, _, &Some((_, target))) => {
                        let llret = bcx.call(datum.val,
                                             &llargs[..],
                                             cleanup_bundle.as_ref(),
                                             Some(attrs));
                        if must_copy_dest {
                            let (ret_dest, ret_ty) = ret_dest_ty
                                .expect("return destination and type not set");
                            bcx.with_block(|bcx| {
                                base::store_ty(bcx, llret, ret_dest.llval, ret_ty);
                            });
                        }
                        for op in args {
                            self.set_operand_dropped(&bcx, op);
                        }
                        funclet_br(bcx, self.llblock(target));
                    }
                    // Foreign functions
                    (true, _, destination) => {
                        let (dest, _) = ret_dest_ty
                            .expect("return destination is not set");
                        bcx = bcx.map_block(|bcx| {
                            foreign::trans_native_call(bcx,
                                                       datum.ty,
                                                       datum.val,
                                                       dest.llval,
                                                       &llargs[..],
                                                       arg_tys,
                                                       debugloc)
                        });
                        if let Some((_, target)) = *destination {
                            for op in args {
                                self.set_operand_dropped(&bcx, op);
                            }
                            funclet_br(bcx, self.llblock(target));
                        }
                    },
                }
            }
        }
    }

    fn get_personality_slot(&mut self, bcx: &BlockAndBuilder<'bcx, 'tcx>) -> ValueRef {
        let ccx = bcx.ccx();
        if let Some(slot) = self.llpersonalityslot {
            slot
        } else {
            let llretty = Type::struct_(ccx, &[Type::i8p(ccx), Type::i32(ccx)], false);
            bcx.with_block(|bcx| {
                let slot = base::alloca(bcx, llretty, "personalityslot");
                self.llpersonalityslot = Some(slot);
                base::call_lifetime_start(bcx, slot);
                slot
            })
        }
    }

    /// Create a landingpad wrapper around the given Block.
    ///
    /// No-op in MSVC SEH scheme.
    fn make_landing_pad(&mut self,
                        cleanup: BlockAndBuilder<'bcx, 'tcx>)
                        -> BlockAndBuilder<'bcx, 'tcx>
    {
        if base::wants_msvc_seh(cleanup.sess()) {
            return cleanup;
        }
        let bcx = self.fcx.new_block("cleanup", None).build();
        let ccx = bcx.ccx();
        let llpersonality = self.fcx.eh_personality();
        let llretty = Type::struct_(ccx, &[Type::i8p(ccx), Type::i32(ccx)], false);
        let llretval = bcx.landing_pad(llretty, llpersonality, 1, self.fcx.llfn);
        bcx.set_cleanup(llretval);
        let slot = self.get_personality_slot(&bcx);
        bcx.store(llretval, slot);
        bcx.br(cleanup.llbb());
        bcx
    }

    /// Create prologue cleanuppad instruction under MSVC SEH handling scheme.
    ///
    /// Also handles setting some state for the original trans and creating an operand bundle for
    /// function calls.
    fn make_cleanup_pad(&mut self, bb: mir::BasicBlock) -> Option<(ValueRef, OperandBundleDef)> {
        let bcx = self.bcx(bb);
        let data = self.mir.basic_block_data(bb);
        let use_funclets = base::wants_msvc_seh(bcx.sess()) && data.is_cleanup;
        let cleanup_pad = if use_funclets {
            bcx.set_personality_fn(self.fcx.eh_personality());
            bcx.at_start(|bcx| Some(bcx.cleanup_pad(None, &[])))
        } else {
            None
        };
        // Set the landingpad global-state for old translator, so it knows about the SEH used.
        bcx.set_lpad(if let Some(cleanup_pad) = cleanup_pad {
            Some(common::LandingPad::msvc(cleanup_pad))
        } else if data.is_cleanup {
            Some(common::LandingPad::gnu())
        } else {
            None
        });
        cleanup_pad.map(|f| (f, OperandBundleDef::new("funclet", &[f])))
    }

    fn unreachable_block(&mut self) -> Block<'bcx, 'tcx> {
        self.unreachable_block.unwrap_or_else(|| {
            let bl = self.fcx.new_block("unreachable", None);
            bl.build().unreachable();
            self.unreachable_block = Some(bl);
            bl
        })
    }

    fn bcx(&self, bb: mir::BasicBlock) -> BlockAndBuilder<'bcx, 'tcx> {
        self.blocks[bb.index()].build()
    }

    pub fn llblock(&self, bb: mir::BasicBlock) -> BasicBlockRef {
        self.blocks[bb.index()].llbb
    }
}
