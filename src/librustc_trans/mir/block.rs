// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use llvm::{self, BasicBlockRef, ValueRef, OperandBundleDef};
use rustc::ty;
use rustc::mir::repr as mir;
use abi::{Abi, FnType, ArgType};
use adt;
use base;
use build;
use callee::{Callee, CalleeData, Fn, Intrinsic, NamedTupleConstructor, Virtual};
use common::{self, type_is_fat_ptr, Block, BlockAndBuilder, C_undef};
use debuginfo::DebugLoc;
use Disr;
use machine::{llalign_of_min, llbitsize_of_real};
use meth;
use type_of;
use glue;
use type_::Type;
use rustc_data_structures::fnv::FnvHashMap;

use super::{MirContext, TempRef, drop};
use super::constant::Const;
use super::lvalue::{LvalueRef, load_fat_ptr};
use super::operand::OperandRef;
use super::operand::OperandValue::{self, FatPtr, Immediate, Ref};

impl<'bcx, 'tcx> MirContext<'bcx, 'tcx> {
    pub fn trans_block(&mut self, bb: mir::BasicBlock) {
        debug!("trans_block({:?})", bb);

        let mut bcx = self.bcx(bb);
        let mir = self.mir.clone();
        let data = mir.basic_block_data(bb);

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

        let terminator = data.terminator();
        debug!("trans_block: terminator: {:?}", terminator);

        let debug_loc = DebugLoc::ScopeAt(self.scopes[terminator.scope.index()],
                                          terminator.span);
        debug_loc.apply_to_bcx(&bcx);
        debug_loc.apply(bcx.fcx());
        match terminator.kind {
            mir::TerminatorKind::Resume => {
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

            mir::TerminatorKind::Goto { target } => {
                funclet_br(bcx, self.llblock(target));
            }

            mir::TerminatorKind::If { ref cond, targets: (true_bb, false_bb) } => {
                let cond = self.trans_operand(&bcx, cond);
                let lltrue = self.llblock(true_bb);
                let llfalse = self.llblock(false_bb);
                bcx.cond_br(cond.immediate(), lltrue, llfalse);
            }

            mir::TerminatorKind::Switch { ref discr, ref adt_def, ref targets } => {
                let discr_lvalue = self.trans_lvalue(&bcx, discr);
                let ty = discr_lvalue.ty.to_ty(bcx.tcx());
                let repr = adt::represent_type(bcx.ccx(), ty);
                let discr = bcx.with_block(|bcx|
                    adt::trans_get_discr(bcx, &repr, discr_lvalue.llval, None, true)
                );

                let mut bb_hist = FnvHashMap();
                for target in targets {
                    *bb_hist.entry(target).or_insert(0) += 1;
                }
                let (default_bb, default_blk) = match bb_hist.iter().max_by_key(|&(_, c)| c) {
                    // If a single target basic blocks is predominant, promote that to be the
                    // default case for the switch instruction to reduce the size of the generated
                    // code. This is especially helpful in cases like an if-let on a huge enum.
                    // Note: This optimization is only valid for exhaustive matches.
                    Some((&&bb, &c)) if c > targets.len() / 2 => {
                        (Some(bb), self.blocks[bb.index()])
                    }
                    // We're generating an exhaustive switch, so the else branch
                    // can't be hit.  Branching to an unreachable instruction
                    // lets LLVM know this
                    _ => (None, self.unreachable_block())
                };
                let switch = bcx.switch(discr, default_blk.llbb, targets.len());
                assert_eq!(adt_def.variants.len(), targets.len());
                for (adt_variant, &target) in adt_def.variants.iter().zip(targets) {
                    if default_bb != Some(target) {
                        let llbb = self.llblock(target);
                        let llval = bcx.with_block(|bcx| adt::trans_case(
                                bcx, &repr, Disr::from(adt_variant.disr_val)));
                        build::AddCase(switch, llval, llbb)
                    }
                }
            }

            mir::TerminatorKind::SwitchInt { ref discr, switch_ty, ref values, ref targets } => {
                let (otherwise, targets) = targets.split_last().unwrap();
                let discr = bcx.load(self.trans_lvalue(&bcx, discr).llval);
                let discr = bcx.with_block(|bcx| base::to_immediate(bcx, discr, switch_ty));
                let switch = bcx.switch(discr, self.llblock(*otherwise), values.len());
                for (value, target) in values.iter().zip(targets) {
                    let val = Const::from_constval(bcx.ccx(), value.clone(), switch_ty);
                    let llbb = self.llblock(*target);
                    build::AddCase(switch, val.llval, llbb)
                }
            }

            mir::TerminatorKind::Return => {
                bcx.with_block(|bcx| {
                    self.fcx.build_return_block(bcx, debug_loc);
                })
            }

            mir::TerminatorKind::Drop { ref value, target, unwind } => {
                let lvalue = self.trans_lvalue(&bcx, value);
                let ty = lvalue.ty.to_ty(bcx.tcx());
                // Double check for necessity to drop
                if !glue::type_needs_drop(bcx.tcx(), ty) {
                    funclet_br(bcx, self.llblock(target));
                    return;
                }
                let drop_fn = glue::get_drop_glue(bcx.ccx(), ty);
                let drop_ty = glue::get_drop_glue_type(bcx.tcx(), ty);
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
                               cleanup_bundle.as_ref());
                    self.bcx(target).at_start(|bcx| {
                        debug_loc.apply_to_bcx(bcx);
                        drop::drop_fill(bcx, lvalue.llval, ty)
                    });
                } else {
                    bcx.call(drop_fn, &[llvalue], cleanup_bundle.as_ref());
                    drop::drop_fill(&bcx, lvalue.llval, ty);
                    funclet_br(bcx, self.llblock(target));
                }
            }

            mir::TerminatorKind::Call { ref func, ref args, ref destination, ref cleanup } => {
                // Create the callee. This is a fn ptr or zero-sized and hence a kind of scalar.
                let callee = self.trans_operand(&bcx, func);

                let (mut callee, abi, sig) = match callee.ty.sty {
                    ty::TyFnDef(def_id, substs, f) => {
                        (Callee::def(bcx.ccx(), def_id, substs), f.abi, &f.sig)
                    }
                    ty::TyFnPtr(f) => {
                        (Callee {
                            data: Fn(callee.immediate()),
                            ty: callee.ty
                        }, f.abi, &f.sig)
                    }
                    _ => bug!("{} is not callable", callee.ty)
                };

                let sig = bcx.tcx().erase_late_bound_regions(sig);

                // Handle intrinsics old trans wants Expr's for, ourselves.
                let intrinsic = match (&callee.ty.sty, &callee.data) {
                    (&ty::TyFnDef(def_id, _, _), &Intrinsic) => {
                        Some(bcx.tcx().item_name(def_id).as_str())
                    }
                    _ => None
                };
                let intrinsic = intrinsic.as_ref().map(|s| &s[..]);

                if intrinsic == Some("move_val_init") {
                    let &(_, target) = destination.as_ref().unwrap();
                    // The first argument is a thin destination pointer.
                    let llptr = self.trans_operand(&bcx, &args[0]).immediate();
                    let val = self.trans_operand(&bcx, &args[1]);
                    self.store_operand(&bcx, llptr, val);
                    self.set_operand_dropped(&bcx, &args[1]);
                    funclet_br(bcx, self.llblock(target));
                    return;
                }

                if intrinsic == Some("transmute") {
                    let &(ref dest, target) = destination.as_ref().unwrap();
                    self.with_lvalue_ref(&bcx, dest, |this, dest| {
                        this.trans_transmute(&bcx, &args[0], dest);
                    });

                    self.set_operand_dropped(&bcx, &args[0]);
                    funclet_br(bcx, self.llblock(target));
                    return;
                }

                let extra_args = &args[sig.inputs.len()..];
                let extra_args = extra_args.iter().map(|op_arg| {
                    let op_ty = self.mir.operand_ty(bcx.tcx(), op_arg);
                    bcx.monomorphize(&op_ty)
                }).collect::<Vec<_>>();
                let fn_ty = callee.direct_fn_type(bcx.ccx(), &extra_args);

                // The arguments we'll be passing. Plus one to account for outptr, if used.
                let arg_count = fn_ty.args.len() + fn_ty.ret.is_indirect() as usize;
                let mut llargs = Vec::with_capacity(arg_count);

                // Prepare the return value destination
                let ret_dest = if let Some((ref dest, _)) = *destination {
                    let is_intrinsic = if let Intrinsic = callee.data {
                        true
                    } else {
                        false
                    };
                    self.make_return_dest(&bcx, dest, &fn_ty.ret, &mut llargs, is_intrinsic)
                } else {
                    ReturnDest::Nothing
                };

                // Split the rust-call tupled arguments off.
                let (first_args, untuple) = if abi == Abi::RustCall && !args.is_empty() {
                    let (tup, args) = args.split_last().unwrap();
                    (args, Some(tup))
                } else {
                    (&args[..], None)
                };

                let is_shuffle = intrinsic.map_or(false, |name| {
                    name.starts_with("simd_shuffle")
                });
                let mut idx = 0;
                for arg in first_args {
                    // The indices passed to simd_shuffle* in the
                    // third argument must be constant. This is
                    // checked by const-qualification, which also
                    // promotes any complex rvalues to constants.
                    if is_shuffle && idx == 2 {
                        match *arg {
                            mir::Operand::Consume(_) => {
                                span_bug!(terminator.span,
                                          "shuffle indices must be constant");
                            }
                            mir::Operand::Constant(ref constant) => {
                                let val = self.trans_constant(&bcx, constant);
                                llargs.push(val.llval);
                                idx += 1;
                                continue;
                            }
                        }
                    }

                    let val = self.trans_operand(&bcx, arg).val;
                    self.trans_argument(&bcx, val, &mut llargs, &fn_ty,
                                        &mut idx, &mut callee.data);
                }
                if let Some(tup) = untuple {
                    self.trans_arguments_untupled(&bcx, tup, &mut llargs, &fn_ty,
                                                  &mut idx, &mut callee.data)
                }

                let fn_ptr = match callee.data {
                    NamedTupleConstructor(_) => {
                        // FIXME translate this like mir::Rvalue::Aggregate.
                        callee.reify(bcx.ccx()).val
                    }
                    Intrinsic => {
                        use callee::ArgVals;
                        use expr::{Ignore, SaveIn};
                        use intrinsic::trans_intrinsic_call;

                        let (dest, llargs) = match ret_dest {
                            _ if fn_ty.ret.is_indirect() => {
                                (SaveIn(llargs[0]), &llargs[1..])
                            }
                            ReturnDest::Nothing => (Ignore, &llargs[..]),
                            ReturnDest::IndirectOperand(dst, _) |
                            ReturnDest::Store(dst) => (SaveIn(dst), &llargs[..]),
                            ReturnDest::DirectOperand(_) =>
                                bug!("Cannot use direct operand with an intrinsic call")
                        };

                        bcx.with_block(|bcx| {
                            trans_intrinsic_call(bcx, callee.ty, &fn_ty,
                                                           ArgVals(llargs), dest,
                                                           debug_loc);
                        });

                        if let ReturnDest::IndirectOperand(dst, _) = ret_dest {
                            // Make a fake operand for store_return
                            let op = OperandRef {
                                val: OperandValue::Ref(dst),
                                ty: sig.output.unwrap()
                            };
                            self.store_return(&bcx, ret_dest, fn_ty.ret, op);
                        }

                        if let Some((_, target)) = *destination {
                            for op in args {
                                self.set_operand_dropped(&bcx, op);
                            }
                            funclet_br(bcx, self.llblock(target));
                        } else {
                            // trans_intrinsic_call already used Unreachable.
                            // bcx.unreachable();
                        }

                        return;
                    }
                    Fn(f) => f,
                    Virtual(_) => bug!("Virtual fn ptr not extracted")
                };

                // Many different ways to call a function handled here
                if let Some(cleanup) = cleanup.map(|bb| self.bcx(bb)) {
                    let ret_bcx = if let Some((_, target)) = *destination {
                        self.blocks[target.index()]
                    } else {
                        self.unreachable_block()
                    };
                    let landingpad = self.make_landing_pad(cleanup);

                    let invokeret = bcx.invoke(fn_ptr,
                                               &llargs,
                                               ret_bcx.llbb,
                                               landingpad.llbb(),
                                               cleanup_bundle.as_ref());
                    fn_ty.apply_attrs_callsite(invokeret);

                    landingpad.at_start(|bcx| {
                        debug_loc.apply_to_bcx(bcx);
                        for op in args {
                            self.set_operand_dropped(bcx, op);
                        }
                    });

                    if destination.is_some() {
                        let ret_bcx = ret_bcx.build();
                        ret_bcx.at_start(|ret_bcx| {
                            debug_loc.apply_to_bcx(ret_bcx);
                            let op = OperandRef {
                                val: OperandValue::Immediate(invokeret),
                                ty: sig.output.unwrap()
                            };
                            self.store_return(&ret_bcx, ret_dest, fn_ty.ret, op);
                            for op in args {
                                self.set_operand_dropped(&ret_bcx, op);
                            }
                        });
                    }
                } else {
                    let llret = bcx.call(fn_ptr, &llargs, cleanup_bundle.as_ref());
                    fn_ty.apply_attrs_callsite(llret);
                    if let Some((_, target)) = *destination {
                        let op = OperandRef {
                            val: OperandValue::Immediate(llret),
                            ty: sig.output.unwrap()
                        };
                        self.store_return(&bcx, ret_dest, fn_ty.ret, op);
                        for op in args {
                            self.set_operand_dropped(&bcx, op);
                        }
                        funclet_br(bcx, self.llblock(target));
                    } else {
                        // no need to drop args, because the call never returns
                        bcx.unreachable();
                    }
                }
            }
        }
    }

    fn trans_argument(&mut self,
                      bcx: &BlockAndBuilder<'bcx, 'tcx>,
                      val: OperandValue,
                      llargs: &mut Vec<ValueRef>,
                      fn_ty: &FnType,
                      next_idx: &mut usize,
                      callee: &mut CalleeData) {
        // Treat the values in a fat pointer separately.
        if let FatPtr(ptr, meta) = val {
            if *next_idx == 0 {
                if let Virtual(idx) = *callee {
                    let llfn = bcx.with_block(|bcx| {
                        meth::get_virtual_method(bcx, meta, idx)
                    });
                    let llty = fn_ty.llvm_type(bcx.ccx()).ptr_to();
                    *callee = Fn(bcx.pointercast(llfn, llty));
                }
            }
            self.trans_argument(bcx, Immediate(ptr), llargs, fn_ty, next_idx, callee);
            self.trans_argument(bcx, Immediate(meta), llargs, fn_ty, next_idx, callee);
            return;
        }

        let arg = &fn_ty.args[*next_idx];
        *next_idx += 1;

        // Fill padding with undef value, where applicable.
        if let Some(ty) = arg.pad {
            llargs.push(C_undef(ty));
        }

        if arg.is_ignore() {
            return;
        }

        // Force by-ref if we have to load through a cast pointer.
        let (mut llval, by_ref) = match val {
            Immediate(llval) if arg.is_indirect() || arg.cast.is_some() => {
                let llscratch = build::AllocaFcx(bcx.fcx(), arg.original_ty, "arg");
                bcx.store(llval, llscratch);
                (llscratch, true)
            }
            Immediate(llval) => (llval, false),
            Ref(llval) => (llval, true),
            FatPtr(_, _) => bug!("fat pointers handled above")
        };

        if by_ref && !arg.is_indirect() {
            // Have to load the argument, maybe while casting it.
            if arg.original_ty == Type::i1(bcx.ccx()) {
                // We store bools as i8 so we need to truncate to i1.
                llval = bcx.load_range_assert(llval, 0, 2, llvm::False);
                llval = bcx.trunc(llval, arg.original_ty);
            } else if let Some(ty) = arg.cast {
                llval = bcx.load(bcx.pointercast(llval, ty.ptr_to()));
                let llalign = llalign_of_min(bcx.ccx(), arg.ty);
                unsafe {
                    llvm::LLVMSetAlignment(llval, llalign);
                }
            } else {
                llval = bcx.load(llval);
            }
        }

        llargs.push(llval);
    }

    fn trans_arguments_untupled(&mut self,
                                bcx: &BlockAndBuilder<'bcx, 'tcx>,
                                operand: &mir::Operand<'tcx>,
                                llargs: &mut Vec<ValueRef>,
                                fn_ty: &FnType,
                                next_idx: &mut usize,
                                callee: &mut CalleeData) {
        let tuple = self.trans_operand(bcx, operand);

        let arg_types = match tuple.ty.sty {
            ty::TyTuple(ref tys) => tys,
            _ => span_bug!(self.mir.span,
                           "bad final argument to \"rust-call\" fn {:?}", tuple.ty)
        };

        // Handle both by-ref and immediate tuples.
        match tuple.val {
            Ref(llval) => {
                let base_repr = adt::represent_type(bcx.ccx(), tuple.ty);
                let base = adt::MaybeSizedValue::sized(llval);
                for (n, &ty) in arg_types.iter().enumerate() {
                    let ptr = adt::trans_field_ptr_builder(bcx, &base_repr, base, Disr(0), n);
                    let val = if common::type_is_fat_ptr(bcx.tcx(), ty) {
                        let (lldata, llextra) = load_fat_ptr(bcx, ptr);
                        FatPtr(lldata, llextra)
                    } else {
                        // trans_argument will load this if it needs to
                        Ref(ptr)
                    };
                    self.trans_argument(bcx, val, llargs, fn_ty, next_idx, callee);
                }

            }
            Immediate(llval) => {
                for (n, &ty) in arg_types.iter().enumerate() {
                    let mut elem = bcx.extract_value(llval, n);
                    // Truncate bools to i1, if needed
                    if ty.is_bool() && common::val_ty(elem) != Type::i1(bcx.ccx()) {
                        elem = bcx.trunc(elem, Type::i1(bcx.ccx()));
                    }
                    // If the tuple is immediate, the elements are as well
                    let val = Immediate(elem);
                    self.trans_argument(bcx, val, llargs, fn_ty, next_idx, callee);
                }
            }
            FatPtr(_, _) => bug!("tuple is a fat pointer?!")
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
            bcx.at_start(|bcx| {
                DebugLoc::None.apply_to_bcx(bcx);
                Some(bcx.cleanup_pad(None, &[]))
            })
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

    fn make_return_dest(&mut self, bcx: &BlockAndBuilder<'bcx, 'tcx>,
                        dest: &mir::Lvalue<'tcx>, fn_ret_ty: &ArgType,
                        llargs: &mut Vec<ValueRef>, is_intrinsic: bool) -> ReturnDest {
        // If the return is ignored, we can just return a do-nothing ReturnDest
        if fn_ret_ty.is_ignore() {
            return ReturnDest::Nothing;
        }
        let dest = match *dest {
            mir::Lvalue::Temp(idx) => {
                let lvalue_ty = self.mir.lvalue_ty(bcx.tcx(), dest);
                let lvalue_ty = bcx.monomorphize(&lvalue_ty);
                let ret_ty = lvalue_ty.to_ty(bcx.tcx());
                match self.temps[idx as usize] {
                    TempRef::Lvalue(dest) => dest,
                    TempRef::Operand(None) => {
                        // Handle temporary lvalues, specifically Operand ones, as
                        // they don't have allocas
                        return if fn_ret_ty.is_indirect() {
                            // Odd, but possible, case, we have an operand temporary,
                            // but the calling convention has an indirect return.
                            let tmp = bcx.with_block(|bcx| {
                                base::alloc_ty(bcx, ret_ty, "tmp_ret")
                            });
                            llargs.push(tmp);
                            ReturnDest::IndirectOperand(tmp, idx)
                        } else if is_intrinsic {
                            // Currently, intrinsics always need a location to store
                            // the result. so we create a temporary alloca for the
                            // result
                            let tmp = bcx.with_block(|bcx| {
                                base::alloc_ty(bcx, ret_ty, "tmp_ret")
                            });
                            ReturnDest::IndirectOperand(tmp, idx)
                        } else {
                            ReturnDest::DirectOperand(idx)
                        };
                    }
                    TempRef::Operand(Some(_)) => {
                        bug!("lvalue temp already assigned to");
                    }
                }
            }
            _ => self.trans_lvalue(bcx, dest)
        };
        if fn_ret_ty.is_indirect() {
            llargs.push(dest.llval);
            ReturnDest::Nothing
        } else {
            ReturnDest::Store(dest.llval)
        }
    }

    fn trans_transmute(&mut self, bcx: &BlockAndBuilder<'bcx, 'tcx>,
                       src: &mir::Operand<'tcx>, dst: LvalueRef<'tcx>) {
        let mut val = self.trans_operand(bcx, src);
        if let ty::TyFnDef(def_id, substs, _) = val.ty.sty {
            let llouttype = type_of::type_of(bcx.ccx(), dst.ty.to_ty(bcx.tcx()));
            let out_type_size = llbitsize_of_real(bcx.ccx(), llouttype);
            if out_type_size != 0 {
                // FIXME #19925 Remove this hack after a release cycle.
                let f = Callee::def(bcx.ccx(), def_id, substs);
                let datum = f.reify(bcx.ccx());
                val = OperandRef {
                    val: OperandValue::Immediate(datum.val),
                    ty: datum.ty
                };
            }
        }

        let llty = type_of::type_of(bcx.ccx(), val.ty);
        let cast_ptr = bcx.pointercast(dst.llval, llty.ptr_to());
        self.store_operand(bcx, cast_ptr, val);
    }

    // Stores the return value of a function call into it's final location.
    fn store_return(&mut self,
                    bcx: &BlockAndBuilder<'bcx, 'tcx>,
                    dest: ReturnDest,
                    ret_ty: ArgType,
                    op: OperandRef<'tcx>) {
        use self::ReturnDest::*;

        match dest {
            Nothing => (),
            Store(dst) => ret_ty.store(bcx, op.immediate(), dst),
            IndirectOperand(tmp, idx) => {
                let op = self.trans_load(bcx, tmp, op.ty);
                self.temps[idx as usize] = TempRef::Operand(Some(op));
            }
            DirectOperand(idx) => {
                let op = if type_is_fat_ptr(bcx.tcx(), op.ty) {
                    let llval = op.immediate();
                    let ptr = bcx.extract_value(llval, 0);
                    let meta = bcx.extract_value(llval, 1);

                    OperandRef {
                        val: OperandValue::FatPtr(ptr, meta),
                        ty: op.ty
                    }
                } else {
                    op
                };
                self.temps[idx as usize] = TempRef::Operand(Some(op));
            }
        }
    }
}

enum ReturnDest {
    // Do nothing, the return value is indirect or ignored
    Nothing,
    // Store the return value to the pointer
    Store(ValueRef),
    // Stores an indirect return value to an operand temporary lvalue
    IndirectOperand(ValueRef, u32),
    // Stores a direct return value to an operand temporary lvalue
    DirectOperand(u32)
}
