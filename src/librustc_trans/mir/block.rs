// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use llvm::{self, ValueRef, BasicBlockRef};
use rustc_const_eval::{ErrKind, ConstEvalErr, note_const_eval_err};
use rustc::middle::lang_items;
use rustc::middle::const_val::ConstInt;
use rustc::ty::{self, layout, TypeFoldable};
use rustc::mir;
use abi::{Abi, FnType, ArgType};
use base::{self, Lifetime};
use callee::{Callee, CalleeData, Fn, Intrinsic, NamedTupleConstructor, Virtual};
use builder::Builder;
use common::{self, Funclet};
use common::{C_bool, C_str_slice, C_struct, C_u32, C_undef};
use consts;
use machine::llalign_of_min;
use meth;
use type_of::{self, align_of};
use glue;
use type_::Type;

use rustc_data_structures::indexed_vec::IndexVec;
use syntax::symbol::Symbol;

use std::cmp;

use super::{MirContext, LocalRef};
use super::analyze::CleanupKind;
use super::constant::Const;
use super::lvalue::{Alignment, LvalueRef};
use super::operand::OperandRef;
use super::operand::OperandValue::{Pair, Ref, Immediate};

impl<'a, 'tcx> MirContext<'a, 'tcx> {
    pub fn trans_block(&mut self, bb: mir::BasicBlock,
        funclets: &IndexVec<mir::BasicBlock, Option<Funclet>>) {
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

        let funclet_br = |this: &Self, bcx: Builder, bb: mir::BasicBlock| {
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

        let llblock = |this: &mut Self, target: mir::BasicBlock| {
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
            bcx = self.trans_statement(bcx, statement);
        }

        let terminator = data.terminator();
        debug!("trans_block: terminator: {:?}", terminator);

        let span = terminator.source_info.span;
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
                let ty = location.ty(&self.mir, bcx.tcx()).to_ty(bcx.tcx());
                let ty = self.monomorphize(&ty);

                // Double check for necessity to drop
                if !bcx.ccx.shared().type_needs_drop(ty) {
                    funclet_br(self, bcx, target);
                    return;
                }

                let mut lvalue = self.trans_lvalue(&bcx, location);
                let drop_fn = glue::get_drop_glue(bcx.ccx, ty);
                let drop_ty = glue::get_drop_glue_type(bcx.ccx.shared(), ty);
                if bcx.ccx.shared().type_is_sized(ty) && drop_ty != ty {
                    lvalue.llval = bcx.pointercast(
                        lvalue.llval, type_of::type_of(bcx.ccx, drop_ty).ptr_to());
                }
                let args = &[lvalue.llval, lvalue.llextra][..1 + lvalue.has_extra() as usize];
                if let Some(unwind) = unwind {
                    bcx.invoke(
                        drop_fn,
                        args,
                        self.blocks[target],
                        llblock(self, unwind),
                        cleanup_bundle
                    );
                } else {
                    bcx.call(drop_fn, args, cleanup_bundle);
                    funclet_br(self, bcx, target);
                }
            }

            mir::TerminatorKind::Assert { ref cond, expected, ref msg, target, cleanup } => {
                let cond = self.trans_operand(&bcx, cond).immediate();
                let mut const_cond = common::const_to_opt_u128(cond, false).map(|c| c == 1);

                // This case can currently arise only from functions marked
                // with #[rustc_inherit_overflow_checks] and inlined from
                // another crate (mostly core::num generic/#[inline] fns),
                // while the current crate doesn't use overflow checks.
                // NOTE: Unlike binops, negation doesn't have its own
                // checked operation, just a comparison with the minimum
                // value, so we have to check for the assert message.
                if !bcx.ccx.check_overflow() {
                    use rustc_const_math::ConstMathErr::Overflow;
                    use rustc_const_math::Op::Neg;

                    if let mir::AssertMessage::Math(Overflow(Neg)) = *msg {
                        const_cond = Some(expected);
                    }
                }

                // Don't translate the panic block if success if known.
                if const_cond == Some(expected) {
                    funclet_br(self, bcx, target);
                    return;
                }

                // Pass the condition through llvm.expect for branch hinting.
                let expect = bcx.ccx.get_intrinsic(&"llvm.expect.i1");
                let cond = bcx.call(expect, &[cond, C_bool(bcx.ccx, expected)], None);

                // Create the failure block and the conditional branch to it.
                let lltarget = llblock(self, target);
                let panic_block = self.new_block("panic");
                if expected {
                    bcx.cond_br(cond, lltarget, panic_block.llbb());
                } else {
                    bcx.cond_br(cond, panic_block.llbb(), lltarget);
                }

                // After this point, bcx is the block for the call to panic.
                bcx = panic_block;
                self.set_debug_loc(&bcx, terminator.source_info);

                // Get the location information.
                let loc = bcx.sess().codemap().lookup_char_pos(span.lo);
                let filename = Symbol::intern(&loc.file.name).as_str();
                let filename = C_str_slice(bcx.ccx, filename);
                let line = C_u32(bcx.ccx, loc.line as u32);

                // Put together the arguments to the panic entry point.
                let (lang_item, args, const_err) = match *msg {
                    mir::AssertMessage::BoundsCheck { ref len, ref index } => {
                        let len = self.trans_operand(&mut bcx, len).immediate();
                        let index = self.trans_operand(&mut bcx, index).immediate();

                        let const_err = common::const_to_opt_u128(len, false)
                            .and_then(|len| common::const_to_opt_u128(index, false)
                                .map(|index| ErrKind::IndexOutOfBounds {
                                    len: len as u64,
                                    index: index as u64
                                }));

                        let file_line = C_struct(bcx.ccx, &[filename, line], false);
                        let align = llalign_of_min(bcx.ccx, common::val_ty(file_line));
                        let file_line = consts::addr_of(bcx.ccx,
                                                        file_line,
                                                        align,
                                                        "panic_bounds_check_loc");
                        (lang_items::PanicBoundsCheckFnLangItem,
                         vec![file_line, index, len],
                         const_err)
                    }
                    mir::AssertMessage::Math(ref err) => {
                        let msg_str = Symbol::intern(err.description()).as_str();
                        let msg_str = C_str_slice(bcx.ccx, msg_str);
                        let msg_file_line = C_struct(bcx.ccx,
                                                     &[msg_str, filename, line],
                                                     false);
                        let align = llalign_of_min(bcx.ccx, common::val_ty(msg_file_line));
                        let msg_file_line = consts::addr_of(bcx.ccx,
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
                        let err = ConstEvalErr{ span: span, kind: err };
                        let mut diag = bcx.tcx().sess.struct_span_warn(
                            span, "this expression will panic at run-time");
                        note_const_eval_err(bcx.tcx(), &err, span, "expression", &mut diag);
                        diag.emit();
                    }
                }

                // Obtain the panic entry point.
                let def_id = common::langcall(bcx.tcx(), Some(span), "", lang_item);
                let callee = Callee::def(bcx.ccx, def_id,
                    bcx.ccx.empty_substs_for_def_id(def_id));
                let llfn = callee.reify(bcx.ccx);

                // Translate the actual panic invoke/call.
                if let Some(unwind) = cleanup {
                    bcx.invoke(llfn,
                               &args,
                               self.unreachable_block(),
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

                let (mut callee, sig) = match callee.ty.sty {
                    ty::TyFnDef(def_id, substs, sig) => {
                        (Callee::def(bcx.ccx, def_id, substs), sig)
                    }
                    ty::TyFnPtr(sig) => {
                        (Callee {
                            data: Fn(callee.immediate()),
                            ty: callee.ty
                        }, sig)
                    }
                    _ => bug!("{} is not callable", callee.ty)
                };

                let sig = bcx.tcx().erase_late_bound_regions_and_normalize(&sig);
                let abi = sig.abi;

                // Handle intrinsics old trans wants Expr's for, ourselves.
                let intrinsic = match (&callee.ty.sty, &callee.data) {
                    (&ty::TyFnDef(def_id, ..), &Intrinsic) => {
                        Some(bcx.tcx().item_name(def_id).as_str())
                    }
                    _ => None
                };
                let mut intrinsic = intrinsic.as_ref().map(|s| &s[..]);

                if intrinsic == Some("move_val_init") {
                    let &(_, target) = destination.as_ref().unwrap();
                    // The first argument is a thin destination pointer.
                    let llptr = self.trans_operand(&bcx, &args[0]).immediate();
                    let val = self.trans_operand(&bcx, &args[1]);
                    self.store_operand(&bcx, llptr, None, val);
                    funclet_br(self, bcx, target);
                    return;
                }

                if intrinsic == Some("transmute") {
                    let &(ref dest, target) = destination.as_ref().unwrap();
                    self.trans_transmute(&bcx, &args[0], dest);
                    funclet_br(self, bcx, target);
                    return;
                }

                let extra_args = &args[sig.inputs().len()..];
                let extra_args = extra_args.iter().map(|op_arg| {
                    let op_ty = op_arg.ty(&self.mir, bcx.tcx());
                    self.monomorphize(&op_ty)
                }).collect::<Vec<_>>();
                let fn_ty = callee.direct_fn_type(bcx.ccx, &extra_args);

                if intrinsic == Some("drop_in_place") {
                    let &(_, target) = destination.as_ref().unwrap();
                    let ty = if let ty::TyFnDef(_, substs, _) = callee.ty.sty {
                        substs.type_at(0)
                    } else {
                        bug!("Unexpected ty: {}", callee.ty);
                    };

                    // Double check for necessity to drop
                    if !bcx.ccx.shared().type_needs_drop(ty) {
                        funclet_br(self, bcx, target);
                        return;
                    }

                    let drop_fn = glue::get_drop_glue(bcx.ccx, ty);
                    let llty = fn_ty.llvm_type(bcx.ccx).ptr_to();
                    callee.data = Fn(bcx.pointercast(drop_fn, llty));
                    intrinsic = None;
                }

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
                        callee.reify(bcx.ccx)
                    }
                    Intrinsic => {
                        use intrinsic::trans_intrinsic_call;

                        let (dest, llargs) = match ret_dest {
                            _ if fn_ty.ret.is_indirect() => {
                                (llargs[0], &llargs[1..])
                            }
                            ReturnDest::Nothing => {
                                (C_undef(fn_ty.ret.original_ty.ptr_to()), &llargs[..])
                            }
                            ReturnDest::IndirectOperand(dst, _) |
                            ReturnDest::Store(dst) => (dst, &llargs[..]),
                            ReturnDest::DirectOperand(_) =>
                                bug!("Cannot use direct operand with an intrinsic call")
                        };

                        trans_intrinsic_call(&bcx, callee.ty, &fn_ty, &llargs, dest,
                            terminator.source_info.span);

                        if let ReturnDest::IndirectOperand(dst, _) = ret_dest {
                            // Make a fake operand for store_return
                            let op = OperandRef {
                                val: Ref(dst, Alignment::AbiAligned),
                                ty: sig.output(),
                            };
                            self.store_return(&bcx, ret_dest, fn_ty.ret, op);
                        }

                        if let Some((_, target)) = *destination {
                            funclet_br(self, bcx, target);
                        } else {
                            bcx.unreachable();
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
                                               ret_bcx,
                                               llblock(self, cleanup),
                                               cleanup_bundle);
                    fn_ty.apply_attrs_callsite(invokeret);

                    if let Some((_, target)) = *destination {
                        let ret_bcx = self.get_builder(target);
                        self.set_debug_loc(&ret_bcx, terminator.source_info);
                        let op = OperandRef {
                            val: Immediate(invokeret),
                            ty: sig.output(),
                        };
                        self.store_return(&ret_bcx, ret_dest, fn_ty.ret, op);
                    }
                } else {
                    let llret = bcx.call(fn_ptr, &llargs, cleanup_bundle);
                    fn_ty.apply_attrs_callsite(llret);
                    if let Some((_, target)) = *destination {
                        let op = OperandRef {
                            val: Immediate(llret),
                            ty: sig.output(),
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
                      bcx: &Builder<'a, 'tcx>,
                      op: OperandRef<'tcx>,
                      llargs: &mut Vec<ValueRef>,
                      fn_ty: &FnType,
                      next_idx: &mut usize,
                      callee: &mut CalleeData) {
        if let Pair(a, b) = op.val {
            // Treat the values in a fat pointer separately.
            if common::type_is_fat_ptr(bcx.ccx, op.ty) {
                let (ptr, meta) = (a, b);
                if *next_idx == 0 {
                    if let Virtual(idx) = *callee {
                        let llfn = meth::get_virtual_method(bcx, meta, idx);
                        let llty = fn_ty.llvm_type(bcx.ccx).ptr_to();
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
        let (mut llval, align, by_ref) = match op.val {
            Immediate(_) | Pair(..) => {
                if arg.is_indirect() || arg.cast.is_some() {
                    let llscratch = bcx.alloca(arg.original_ty, "arg");
                    self.store_operand(bcx, llscratch, None, op);
                    (llscratch, Alignment::AbiAligned, true)
                } else {
                    (op.pack_if_pair(bcx).immediate(), Alignment::AbiAligned, false)
                }
            }
            Ref(llval, Alignment::Packed) if arg.is_indirect() => {
                // `foo(packed.large_field)`. We can't pass the (unaligned) field directly. I
                // think that ATM (Rust 1.16) we only pass temporaries, but we shouldn't
                // have scary latent bugs around.

                let llscratch = bcx.alloca(arg.original_ty, "arg");
                base::memcpy_ty(bcx, llscratch, llval, op.ty, Some(1));
                (llscratch, Alignment::AbiAligned, true)
            }
            Ref(llval, align) => (llval, align, true)
        };

        if by_ref && !arg.is_indirect() {
            // Have to load the argument, maybe while casting it.
            if arg.original_ty == Type::i1(bcx.ccx) {
                // We store bools as i8 so we need to truncate to i1.
                llval = bcx.load_range_assert(llval, 0, 2, llvm::False, None);
                llval = bcx.trunc(llval, arg.original_ty);
            } else if let Some(ty) = arg.cast {
                llval = bcx.load(bcx.pointercast(llval, ty.ptr_to()),
                                 align.min_with(llalign_of_min(bcx.ccx, arg.ty)));
            } else {
                llval = bcx.load(llval, align.to_align());
            }
        }

        llargs.push(llval);
    }

    fn trans_arguments_untupled(&mut self,
                                bcx: &Builder<'a, 'tcx>,
                                operand: &mir::Operand<'tcx>,
                                llargs: &mut Vec<ValueRef>,
                                fn_ty: &FnType,
                                next_idx: &mut usize,
                                callee: &mut CalleeData) {
        let tuple = self.trans_operand(bcx, operand);

        let arg_types = match tuple.ty.sty {
            ty::TyTuple(ref tys, _) => tys,
            _ => span_bug!(self.mir.span,
                           "bad final argument to \"rust-call\" fn {:?}", tuple.ty)
        };

        // Handle both by-ref and immediate tuples.
        match tuple.val {
            Ref(llval, align) => {
                for (n, &ty) in arg_types.iter().enumerate() {
                    let ptr = LvalueRef::new_sized_ty(llval, tuple.ty, align);
                    let (ptr, align) = ptr.trans_field_ptr(bcx, n);
                    let val = if common::type_is_fat_ptr(bcx.ccx, ty) {
                        let (lldata, llextra) = base::load_fat_ptr(bcx, ptr, align, ty);
                        Pair(lldata, llextra)
                    } else {
                        // trans_argument will load this if it needs to
                        Ref(ptr, align)
                    };
                    let op = OperandRef {
                        val: val,
                        ty: ty
                    };
                    self.trans_argument(bcx, op, llargs, fn_ty, next_idx, callee);
                }

            }
            Immediate(llval) => {
                let l = bcx.ccx.layout_of(tuple.ty);
                let v = if let layout::Univariant { ref variant, .. } = *l {
                    variant
                } else {
                    bug!("Not a tuple.");
                };
                for (n, &ty) in arg_types.iter().enumerate() {
                    let mut elem = bcx.extract_value(llval, v.memory_index[n] as usize);
                    // Truncate bools to i1, if needed
                    if ty.is_bool() && common::val_ty(elem) != Type::i1(bcx.ccx) {
                        elem = bcx.trunc(elem, Type::i1(bcx.ccx));
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
                    if ty.is_bool() && common::val_ty(elem) != Type::i1(bcx.ccx) {
                        elem = bcx.trunc(elem, Type::i1(bcx.ccx));
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

    /// Return the landingpad wrapper around the given basic block
    ///
    /// No-op in MSVC SEH scheme.
    fn landing_pad_to(&mut self, target_bb: mir::BasicBlock) -> BasicBlockRef {
        if let Some(block) = self.landing_pads[target_bb] {
            return block;
        }

        if base::wants_msvc_seh(self.ccx.sess()) {
            return self.blocks[target_bb];
        }

        let target = self.get_builder(target_bb);

        let bcx = self.new_block("cleanup");
        self.landing_pads[target_bb] = Some(bcx.llbb());

        let ccx = bcx.ccx;
        let llpersonality = self.ccx.eh_personality();
        let llretty = Type::struct_(ccx, &[Type::i8p(ccx), Type::i32(ccx)], false);
        let llretval = bcx.landing_pad(llretty, llpersonality, 1, self.llfn);
        bcx.set_cleanup(llretval);
        let slot = self.get_personality_slot(&bcx);
        bcx.store(llretval, slot, None);
        bcx.br(target.llbb());
        bcx.llbb()
    }

    fn unreachable_block(&mut self) -> BasicBlockRef {
        self.unreachable_block.unwrap_or_else(|| {
            let bl = self.new_block("unreachable");
            bl.unreachable();
            self.unreachable_block = Some(bl.llbb());
            bl.llbb()
        })
    }

    pub fn new_block(&self, name: &str) -> Builder<'a, 'tcx> {
        Builder::new_block(self.ccx, self.llfn, name)
    }

    pub fn get_builder(&self, bb: mir::BasicBlock) -> Builder<'a, 'tcx> {
        let builder = Builder::with_ccx(self.ccx);
        builder.position_at_end(self.blocks[bb]);
        builder
    }

    fn make_return_dest(&mut self, bcx: &Builder<'a, 'tcx>,
                        dest: &mir::Lvalue<'tcx>, fn_ret_ty: &ArgType,
                        llargs: &mut Vec<ValueRef>, is_intrinsic: bool) -> ReturnDest {
        // If the return is ignored, we can just return a do-nothing ReturnDest
        if fn_ret_ty.is_ignore() {
            return ReturnDest::Nothing;
        }
        let dest = if let mir::Lvalue::Local(index) = *dest {
            let ret_ty = self.monomorphized_lvalue_ty(dest);
            match self.locals[index] {
                LocalRef::Lvalue(dest) => dest,
                LocalRef::Operand(None) => {
                    // Handle temporary lvalues, specifically Operand ones, as
                    // they don't have allocas
                    return if fn_ret_ty.is_indirect() {
                        // Odd, but possible, case, we have an operand temporary,
                        // but the calling convention has an indirect return.
                        let tmp = LvalueRef::alloca(bcx, ret_ty, "tmp_ret");
                        llargs.push(tmp.llval);
                        ReturnDest::IndirectOperand(tmp.llval, index)
                    } else if is_intrinsic {
                        // Currently, intrinsics always need a location to store
                        // the result. so we create a temporary alloca for the
                        // result
                        let tmp = LvalueRef::alloca(bcx, ret_ty, "tmp_ret");
                        ReturnDest::IndirectOperand(tmp.llval, index)
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

    fn trans_transmute(&mut self, bcx: &Builder<'a, 'tcx>,
                       src: &mir::Operand<'tcx>,
                       dst: &mir::Lvalue<'tcx>) {
        if let mir::Lvalue::Local(index) = *dst {
            match self.locals[index] {
                LocalRef::Lvalue(lvalue) => self.trans_transmute_into(bcx, src, &lvalue),
                LocalRef::Operand(None) => {
                    let lvalue_ty = self.monomorphized_lvalue_ty(dst);
                    assert!(!lvalue_ty.has_erasable_regions());
                    let lvalue = LvalueRef::alloca(bcx, lvalue_ty, "transmute_temp");
                    self.trans_transmute_into(bcx, src, &lvalue);
                    let op = self.trans_load(bcx, lvalue.llval, lvalue.alignment, lvalue_ty);
                    self.locals[index] = LocalRef::Operand(Some(op));
                }
                LocalRef::Operand(Some(_)) => {
                    let ty = self.monomorphized_lvalue_ty(dst);
                    assert!(common::type_is_zero_size(bcx.ccx, ty),
                            "assigning to initialized SSAtemp");
                }
            }
        } else {
            let dst = self.trans_lvalue(bcx, dst);
            self.trans_transmute_into(bcx, src, &dst);
        }
    }

    fn trans_transmute_into(&mut self, bcx: &Builder<'a, 'tcx>,
                            src: &mir::Operand<'tcx>,
                            dst: &LvalueRef<'tcx>) {
        let val = self.trans_operand(bcx, src);
        let llty = type_of::type_of(bcx.ccx, val.ty);
        let cast_ptr = bcx.pointercast(dst.llval, llty.ptr_to());
        let in_type = val.ty;
        let out_type = dst.ty.to_ty(bcx.tcx());;
        let llalign = cmp::min(align_of(bcx.ccx, in_type), align_of(bcx.ccx, out_type));
        self.store_operand(bcx, cast_ptr, Some(llalign), val);
    }


    // Stores the return value of a function call into it's final location.
    fn store_return(&mut self,
                    bcx: &Builder<'a, 'tcx>,
                    dest: ReturnDest,
                    ret_ty: ArgType,
                    op: OperandRef<'tcx>) {
        use self::ReturnDest::*;

        match dest {
            Nothing => (),
            Store(dst) => ret_ty.store(bcx, op.immediate(), dst),
            IndirectOperand(tmp, index) => {
                let op = self.trans_load(bcx, tmp, Alignment::AbiAligned, op.ty);
                self.locals[index] = LocalRef::Operand(Some(op));
            }
            DirectOperand(index) => {
                // If there is a cast, we have to store and reload.
                let op = if ret_ty.cast.is_some() {
                    let tmp = LvalueRef::alloca(bcx, op.ty, "tmp_ret");
                    ret_ty.store(bcx, op.immediate(), tmp.llval);
                    self.trans_load(bcx, tmp.llval, tmp.alignment, op.ty)
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
