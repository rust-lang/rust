// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use llvm::{self, ValueRef};
use rustc_const_eval::ErrKind;
use rustc::middle::lang_items;
use rustc::ty;
use rustc::mir::repr as mir;
use abi::{Abi, FnType, ArgType};
use adt;
use base;
use build;
use callee::{Callee, CalleeData, Fn, Intrinsic, NamedTupleConstructor, Virtual};
use common::{self, Block, BlockAndBuilder, LandingPad};
use common::{C_bool, C_str_slice, C_struct, C_u32, C_undef};
use consts;
use debuginfo::DebugLoc;
use Disr;
use machine::{llalign_of_min, llbitsize_of_real};
use meth;
use type_of;
use glue;
use type_::Type;

use rustc_data_structures::fnv::FnvHashMap;
use syntax::parse::token;

use super::{MirContext, LocalRef};
use super::analyze::CleanupKind;
use super::constant::Const;
use super::lvalue::{LvalueRef, load_fat_ptr};
use super::operand::OperandRef;
use super::operand::OperandValue::*;

impl<'bcx, 'tcx> MirContext<'bcx, 'tcx> {
    pub fn trans_block(&mut self, bb: mir::BasicBlock) {
        let mut bcx = self.bcx(bb);
        let mir = self.mir.clone();
        let data = &mir[bb];

        debug!("trans_block({:?}={:?})", bb, data);

        // Create the cleanup bundle, if needed.
        let cleanup_pad = bcx.lpad().and_then(|lp| lp.cleanuppad());
        let cleanup_bundle = bcx.lpad().and_then(|l| l.bundle());

        let funclet_br = |this: &Self, bcx: BlockAndBuilder, bb: mir::BasicBlock| {
            let lltarget = this.blocks[bb].llbb;
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

        let llblock = |this: &mut Self, target: mir::BasicBlock| {
            let lltarget = this.blocks[target].llbb;

            if let Some(cp) = cleanup_pad {
                match this.cleanup_kinds[target] {
                    CleanupKind::Funclet => {
                        // MSVC cross-funclet jump - need a trampoline

                        debug!("llblock: creating cleanup trampoline for {:?}", target);
                        let name = &format!("{:?}_cleanup_trampoline_{:?}", bb, target);
                        let trampoline = this.fcx.new_block(name, None).build();
                        trampoline.set_personality_fn(this.fcx.eh_personality());
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
                    this.landing_pad_to(target).llbb
                } else {
                    lltarget
                }
            }
        };

        for statement in &data.statements {
            bcx = self.trans_statement(bcx, statement);
        }

        let terminator = data.terminator();
        debug!("trans_block: terminator: {:?}", terminator);

        let span = terminator.source_info.span;
        let debug_loc = self.debug_loc(terminator.source_info);
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
                funclet_br(self, bcx, target);
            }

            mir::TerminatorKind::If { ref cond, targets: (true_bb, false_bb) } => {
                let cond = self.trans_operand(&bcx, cond);

                let lltrue = llblock(self, true_bb);
                let llfalse = llblock(self, false_bb);
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
                        (Some(bb), llblock(self, bb))
                    }
                    // We're generating an exhaustive switch, so the else branch
                    // can't be hit.  Branching to an unreachable instruction
                    // lets LLVM know this
                    _ => (None, self.unreachable_block().llbb)
                };
                let switch = bcx.switch(discr, default_blk, targets.len());
                assert_eq!(adt_def.variants.len(), targets.len());
                for (adt_variant, &target) in adt_def.variants.iter().zip(targets) {
                    if default_bb != Some(target) {
                        let llbb = llblock(self, target);
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
                let switch = bcx.switch(discr, llblock(self, *otherwise), values.len());
                for (value, target) in values.iter().zip(targets) {
                    let val = Const::from_constval(bcx.ccx(), value.clone(), switch_ty);
                    let llbb = llblock(self, *target);
                    build::AddCase(switch, val.llval, llbb)
                }
            }

            mir::TerminatorKind::Return => {
                let ret = bcx.fcx().fn_ty.ret;
                if ret.is_ignore() || ret.is_indirect() {
                    bcx.ret_void();
                    return;
                }

                let llval = if let Some(cast_ty) = ret.cast {
                    let index = mir.local_index(&mir::Lvalue::ReturnPointer).unwrap();
                    let op = match self.locals[index] {
                        LocalRef::Operand(Some(op)) => op,
                        LocalRef::Operand(None) => bug!("use of return before def"),
                        LocalRef::Lvalue(tr_lvalue) => {
                            OperandRef {
                                val: Ref(tr_lvalue.llval),
                                ty: tr_lvalue.ty.to_ty(bcx.tcx())
                            }
                        }
                    };
                    let llslot = match op.val {
                        Immediate(_) | Pair(..) => {
                            let llscratch = build::AllocaFcx(bcx.fcx(), ret.original_ty, "ret");
                            self.store_operand(&bcx, llscratch, op);
                            llscratch
                        }
                        Ref(llval) => llval
                    };
                    let load = bcx.load(bcx.pointercast(llslot, cast_ty.ptr_to()));
                    let llalign = llalign_of_min(bcx.ccx(), ret.ty);
                    unsafe {
                        llvm::LLVMSetAlignment(load, llalign);
                    }
                    load
                } else {
                    let op = self.trans_consume(&bcx, &mir::Lvalue::ReturnPointer);
                    op.pack_if_pair(&bcx).immediate()
                };
                bcx.ret(llval);
            }

            mir::TerminatorKind::Unreachable => {
                bcx.unreachable();
            }

            mir::TerminatorKind::Drop { ref location, target, unwind } => {
                let ty = mir.lvalue_ty(bcx.tcx(), location).to_ty(bcx.tcx());
                let ty = bcx.monomorphize(&ty);

                // Double check for necessity to drop
                if !glue::type_needs_drop(bcx.tcx(), ty) {
                    funclet_br(self, bcx, target);
                    return;
                }

                let lvalue = self.trans_lvalue(&bcx, location);
                let drop_fn = glue::get_drop_glue(bcx.ccx(), ty);
                let drop_ty = glue::get_drop_glue_type(bcx.tcx(), ty);
                let llvalue = if drop_ty != ty {
                    bcx.pointercast(lvalue.llval, type_of::type_of(bcx.ccx(), drop_ty).ptr_to())
                } else {
                    lvalue.llval
                };
                if let Some(unwind) = unwind {
                    bcx.invoke(drop_fn,
                               &[llvalue],
                               self.blocks[target].llbb,
                               llblock(self, unwind),
                               cleanup_bundle);
                } else {
                    bcx.call(drop_fn, &[llvalue], cleanup_bundle);
                    funclet_br(self, bcx, target);
                }
            }

            mir::TerminatorKind::Assert { ref cond, expected, ref msg, target, cleanup } => {
                let cond = self.trans_operand(&bcx, cond).immediate();
                let const_cond = common::const_to_opt_uint(cond).map(|c| c == 1);

                // Don't translate the panic block if success if known.
                if const_cond == Some(expected) {
                    funclet_br(self, bcx, target);
                    return;
                }

                // Pass the condition through llvm.expect for branch hinting.
                let expect = bcx.ccx().get_intrinsic(&"llvm.expect.i1");
                let cond = bcx.call(expect, &[cond, C_bool(bcx.ccx(), expected)], None);

                // Create the failure block and the conditional branch to it.
                let lltarget = llblock(self, target);
                let panic_block = self.fcx.new_block("panic", None);
                if expected {
                    bcx.cond_br(cond, lltarget, panic_block.llbb);
                } else {
                    bcx.cond_br(cond, panic_block.llbb, lltarget);
                }

                // After this point, bcx is the block for the call to panic.
                bcx = panic_block.build();
                debug_loc.apply_to_bcx(&bcx);

                // Get the location information.
                let loc = bcx.sess().codemap().lookup_char_pos(span.lo);
                let filename = token::intern_and_get_ident(&loc.file.name);
                let filename = C_str_slice(bcx.ccx(), filename);
                let line = C_u32(bcx.ccx(), loc.line as u32);

                // Put together the arguments to the panic entry point.
                let (lang_item, args, const_err) = match *msg {
                    mir::AssertMessage::BoundsCheck { ref len, ref index } => {
                        let len = self.trans_operand(&mut bcx, len).immediate();
                        let index = self.trans_operand(&mut bcx, index).immediate();

                        let const_err = common::const_to_opt_uint(len).and_then(|len| {
                            common::const_to_opt_uint(index).map(|index| {
                                ErrKind::IndexOutOfBounds {
                                    len: len,
                                    index: index
                                }
                            })
                        });

                        let file_line = C_struct(bcx.ccx(), &[filename, line], false);
                        let align = llalign_of_min(bcx.ccx(), common::val_ty(file_line));
                        let file_line = consts::addr_of(bcx.ccx(),
                                                        file_line,
                                                        align,
                                                        "panic_bounds_check_loc");
                        (lang_items::PanicBoundsCheckFnLangItem,
                         vec![file_line, index, len],
                         const_err)
                    }
                    mir::AssertMessage::Math(ref err) => {
                        let msg_str = token::intern_and_get_ident(err.description());
                        let msg_str = C_str_slice(bcx.ccx(), msg_str);
                        let msg_file_line = C_struct(bcx.ccx(),
                                                     &[msg_str, filename, line],
                                                     false);
                        let align = llalign_of_min(bcx.ccx(), common::val_ty(msg_file_line));
                        let msg_file_line = consts::addr_of(bcx.ccx(),
                                                            msg_file_line,
                                                            align,
                                                            "panic_loc");
                        (lang_items::PanicFnLangItem,
                         vec![msg_file_line],
                         Some(ErrKind::Math(err.clone())))
                    }
                };

                // If we know we always panic, and the error message
                // is also constant, then we can produce a warning.
                if const_cond == Some(!expected) {
                    if let Some(err) = const_err {
                        let _ = consts::const_err(bcx.ccx(), span,
                                                  Err::<(), _>(err),
                                                  consts::TrueConst::No);
                    }
                }

                // Obtain the panic entry point.
                let def_id = common::langcall(bcx.tcx(), Some(span), "", lang_item);
                let callee = Callee::def(bcx.ccx(), def_id,
                    bcx.ccx().empty_substs_for_def_id(def_id));
                let llfn = callee.reify(bcx.ccx()).val;

                // Translate the actual panic invoke/call.
                if let Some(unwind) = cleanup {
                    bcx.invoke(llfn,
                               &args,
                               self.unreachable_block().llbb,
                               llblock(self, unwind),
                               cleanup_bundle);
                } else {
                    bcx.call(llfn, &args, cleanup_bundle);
                    bcx.unreachable();
                }
            }

            mir::TerminatorKind::DropAndReplace { .. } => {
                bug!("undesugared DropAndReplace in trans: {:?}", data);
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
                    funclet_br(self, bcx, target);
                    return;
                }

                if intrinsic == Some("transmute") {
                    let &(ref dest, target) = destination.as_ref().unwrap();
                    self.with_lvalue_ref(&bcx, dest, |this, dest| {
                        this.trans_transmute(&bcx, &args[0], dest);
                    });

                    funclet_br(self, bcx, target);
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
                                span_bug!(span, "shuffle indices must be constant");
                            }
                            mir::Operand::Constant(ref constant) => {
                                let val = self.trans_constant(&bcx, constant);
                                llargs.push(val.llval);
                                idx += 1;
                                continue;
                            }
                        }
                    }

                    let op = self.trans_operand(&bcx, arg);
                    self.trans_argument(&bcx, op, &mut llargs, &fn_ty,
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
                                val: Ref(dst),
                                ty: sig.output.unwrap()
                            };
                            self.store_return(&bcx, ret_dest, fn_ty.ret, op);
                        }

                        if let Some((_, target)) = *destination {
                            funclet_br(self, bcx, target);
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
                if let &Some(cleanup) = cleanup {
                    let ret_bcx = if let Some((_, target)) = *destination {
                        self.blocks[target]
                    } else {
                        self.unreachable_block()
                    };
                    let invokeret = bcx.invoke(fn_ptr,
                                               &llargs,
                                               ret_bcx.llbb,
                                               llblock(self, cleanup),
                                               cleanup_bundle);
                    fn_ty.apply_attrs_callsite(invokeret);

                    if destination.is_some() {
                        let ret_bcx = ret_bcx.build();
                        ret_bcx.at_start(|ret_bcx| {
                            debug_loc.apply_to_bcx(ret_bcx);
                            let op = OperandRef {
                                val: Immediate(invokeret),
                                ty: sig.output.unwrap()
                            };
                            self.store_return(&ret_bcx, ret_dest, fn_ty.ret, op);
                        });
                    }
                } else {
                    let llret = bcx.call(fn_ptr, &llargs, cleanup_bundle);
                    fn_ty.apply_attrs_callsite(llret);
                    if let Some((_, target)) = *destination {
                        let op = OperandRef {
                            val: Immediate(llret),
                            ty: sig.output.unwrap()
                        };
                        self.store_return(&bcx, ret_dest, fn_ty.ret, op);
                        funclet_br(self, bcx, target);
                    } else {
                        bcx.unreachable();
                    }
                }
            }
        }
    }

    fn trans_argument(&mut self,
                      bcx: &BlockAndBuilder<'bcx, 'tcx>,
                      op: OperandRef<'tcx>,
                      llargs: &mut Vec<ValueRef>,
                      fn_ty: &FnType,
                      next_idx: &mut usize,
                      callee: &mut CalleeData) {
        if let Pair(a, b) = op.val {
            // Treat the values in a fat pointer separately.
            if common::type_is_fat_ptr(bcx.tcx(), op.ty) {
                let (ptr, meta) = (a, b);
                if *next_idx == 0 {
                    if let Virtual(idx) = *callee {
                        let llfn = bcx.with_block(|bcx| {
                            meth::get_virtual_method(bcx, meta, idx)
                        });
                        let llty = fn_ty.llvm_type(bcx.ccx()).ptr_to();
                        *callee = Fn(bcx.pointercast(llfn, llty));
                    }
                }

                let imm_op = |x| OperandRef {
                    val: Immediate(x),
                    // We won't be checking the type again.
                    ty: bcx.tcx().types.err
                };
                self.trans_argument(bcx, imm_op(ptr), llargs, fn_ty, next_idx, callee);
                self.trans_argument(bcx, imm_op(meta), llargs, fn_ty, next_idx, callee);
                return;
            }
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
        let (mut llval, by_ref) = match op.val {
            Immediate(_) | Pair(..) => {
                if arg.is_indirect() || arg.cast.is_some() {
                    let llscratch = build::AllocaFcx(bcx.fcx(), arg.original_ty, "arg");
                    self.store_operand(bcx, llscratch, op);
                    (llscratch, true)
                } else {
                    (op.pack_if_pair(bcx).immediate(), false)
                }
            }
            Ref(llval) => (llval, true)
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
                        Pair(lldata, llextra)
                    } else {
                        // trans_argument will load this if it needs to
                        Ref(ptr)
                    };
                    let op = OperandRef {
                        val: val,
                        ty: ty
                    };
                    self.trans_argument(bcx, op, llargs, fn_ty, next_idx, callee);
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
                    let op = OperandRef {
                        val: Immediate(elem),
                        ty: ty
                    };
                    self.trans_argument(bcx, op, llargs, fn_ty, next_idx, callee);
                }
            }
            Pair(a, b) => {
                let elems = [a, b];
                for (n, &ty) in arg_types.iter().enumerate() {
                    let mut elem = elems[n];
                    // Truncate bools to i1, if needed
                    if ty.is_bool() && common::val_ty(elem) != Type::i1(bcx.ccx()) {
                        elem = bcx.trunc(elem, Type::i1(bcx.ccx()));
                    }
                    // Pair is always made up of immediates
                    let op = OperandRef {
                        val: Immediate(elem),
                        ty: ty
                    };
                    self.trans_argument(bcx, op, llargs, fn_ty, next_idx, callee);
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

    /// Return the landingpad wrapper around the given basic block
    ///
    /// No-op in MSVC SEH scheme.
    fn landing_pad_to(&mut self, target_bb: mir::BasicBlock) -> Block<'bcx, 'tcx>
    {
        if let Some(block) = self.landing_pads[target_bb] {
            return block;
        }

        if base::wants_msvc_seh(self.fcx.ccx.sess()) {
            return self.blocks[target_bb];
        }

        let target = self.bcx(target_bb);

        let block = self.fcx.new_block("cleanup", None);
        self.landing_pads[target_bb] = Some(block);

        let bcx = block.build();
        let ccx = bcx.ccx();
        let llpersonality = self.fcx.eh_personality();
        let llretty = Type::struct_(ccx, &[Type::i8p(ccx), Type::i32(ccx)], false);
        let llretval = bcx.landing_pad(llretty, llpersonality, 1, self.fcx.llfn);
        bcx.set_cleanup(llretval);
        let slot = self.get_personality_slot(&bcx);
        bcx.store(llretval, slot);
        bcx.br(target.llbb());
        block
    }

    pub fn init_cpad(&mut self, bb: mir::BasicBlock) {
        let bcx = self.bcx(bb);
        let data = &self.mir[bb];
        debug!("init_cpad({:?})", data);

        match self.cleanup_kinds[bb] {
            CleanupKind::NotCleanup => {
                bcx.set_lpad(None)
            }
            _ if !base::wants_msvc_seh(bcx.sess()) => {
                bcx.set_lpad(Some(LandingPad::gnu()))
            }
            CleanupKind::Internal { funclet } => {
                // FIXME: is this needed?
                bcx.set_personality_fn(self.fcx.eh_personality());
                bcx.set_lpad_ref(self.bcx(funclet).lpad());
            }
            CleanupKind::Funclet => {
                bcx.set_personality_fn(self.fcx.eh_personality());
                DebugLoc::None.apply_to_bcx(&bcx);
                let cleanup_pad = bcx.cleanup_pad(None, &[]);
                bcx.set_lpad(Some(LandingPad::msvc(cleanup_pad)));
            }
        };
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
        self.blocks[bb].build()
    }

    fn make_return_dest(&mut self, bcx: &BlockAndBuilder<'bcx, 'tcx>,
                        dest: &mir::Lvalue<'tcx>, fn_ret_ty: &ArgType,
                        llargs: &mut Vec<ValueRef>, is_intrinsic: bool) -> ReturnDest {
        // If the return is ignored, we can just return a do-nothing ReturnDest
        if fn_ret_ty.is_ignore() {
            return ReturnDest::Nothing;
        }
        let dest = if let Some(index) = self.mir.local_index(dest) {
            let ret_ty = self.lvalue_ty(dest);
            match self.locals[index] {
                LocalRef::Lvalue(dest) => dest,
                LocalRef::Operand(None) => {
                    // Handle temporary lvalues, specifically Operand ones, as
                    // they don't have allocas
                    return if fn_ret_ty.is_indirect() {
                        // Odd, but possible, case, we have an operand temporary,
                        // but the calling convention has an indirect return.
                        let tmp = bcx.with_block(|bcx| {
                            base::alloc_ty(bcx, ret_ty, "tmp_ret")
                        });
                        llargs.push(tmp);
                        ReturnDest::IndirectOperand(tmp, index)
                    } else if is_intrinsic {
                        // Currently, intrinsics always need a location to store
                        // the result. so we create a temporary alloca for the
                        // result
                        let tmp = bcx.with_block(|bcx| {
                            base::alloc_ty(bcx, ret_ty, "tmp_ret")
                        });
                        ReturnDest::IndirectOperand(tmp, index)
                    } else {
                        ReturnDest::DirectOperand(index)
                    };
                }
                LocalRef::Operand(Some(_)) => {
                    bug!("lvalue local already assigned to");
                }
            }
        } else {
            self.trans_lvalue(bcx, dest)
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
                    val: Immediate(datum.val),
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
            IndirectOperand(tmp, index) => {
                let op = self.trans_load(bcx, tmp, op.ty);
                self.locals[index] = LocalRef::Operand(Some(op));
            }
            DirectOperand(index) => {
                // If there is a cast, we have to store and reload.
                let op = if ret_ty.cast.is_some() {
                    let tmp = bcx.with_block(|bcx| {
                        base::alloc_ty(bcx, op.ty, "tmp_ret")
                    });
                    ret_ty.store(bcx, op.immediate(), tmp);
                    self.trans_load(bcx, tmp, op.ty)
                } else {
                    op.unpack_if_pair(bcx)
                };
                self.locals[index] = LocalRef::Operand(Some(op));
            }
        }
    }
}

enum ReturnDest {
    // Do nothing, the return value is indirect or ignored
    Nothing,
    // Store the return value to the pointer
    Store(ValueRef),
    // Stores an indirect return value to an operand local lvalue
    IndirectOperand(ValueRef, mir::Local),
    // Stores a direct return value to an operand local lvalue
    DirectOperand(mir::Local)
}
