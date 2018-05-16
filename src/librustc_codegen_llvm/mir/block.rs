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
use rustc::middle::lang_items;
use rustc::ty::{self, Ty, TypeFoldable};
use rustc::ty::layout::{self, LayoutOf};
use rustc::mir;
use rustc::mir::interpret::EvalErrorKind;
use abi::{Abi, ArgType, ArgTypeExt, FnType, FnTypeExt, LlvmType, PassMode};
use base;
use callee;
use builder::{Builder, MemFlags};
use common::{self, C_bool, C_str_slice, C_struct, C_u32, C_uint_big, C_undef};
use consts;
use meth;
use monomorphize;
use type_of::LayoutLlvmExt;
use type_::Type;

use syntax::symbol::Symbol;
use syntax_pos::Pos;

use super::{FunctionCx, LocalRef};
use super::place::PlaceRef;
use super::operand::OperandRef;
use super::operand::OperandValue::{Pair, Ref, Immediate};

impl<'a, 'tcx> FunctionCx<'a, 'tcx> {
    pub fn codegen_block(&mut self, bb: mir::BasicBlock) {
        let mut bx = self.build_block(bb);
        let data = &self.mir[bb];

        debug!("codegen_block({:?}={:?})", bb, data);

        for statement in &data.statements {
            bx = self.codegen_statement(bx, statement);
        }

        self.codegen_terminator(bx, bb, data.terminator());
    }

    fn codegen_terminator(&mut self,
                        mut bx: Builder<'a, 'tcx>,
                        bb: mir::BasicBlock,
                        terminator: &mir::Terminator<'tcx>)
    {
        debug!("codegen_terminator: {:?}", terminator);

        // Create the cleanup bundle, if needed.
        let tcx = bx.tcx();
        let span = terminator.source_info.span;
        let funclet_bb = self.cleanup_kinds[bb].funclet_bb(bb);
        let funclet = funclet_bb.and_then(|funclet_bb| self.funclets[funclet_bb].as_ref());

        let cleanup_pad = funclet.map(|lp| lp.cleanuppad());
        let cleanup_bundle = funclet.map(|l| l.bundle());

        let lltarget = |this: &mut Self, target: mir::BasicBlock| {
            let lltarget = this.blocks[target];
            let target_funclet = this.cleanup_kinds[target].funclet_bb(target);
            match (funclet_bb, target_funclet) {
                (None, None) => (lltarget, false),
                (Some(f), Some(t_f))
                    if f == t_f || !base::wants_msvc_seh(tcx.sess)
                    => (lltarget, false),
                (None, Some(_)) => {
                    // jump *into* cleanup - need a landing pad if GNU
                    (this.landing_pad_to(target), false)
                }
                (Some(_), None) => span_bug!(span, "{:?} - jump out of cleanup?", terminator),
                (Some(_), Some(_)) => {
                    (this.landing_pad_to(target), true)
                }
            }
        };

        let llblock = |this: &mut Self, target: mir::BasicBlock| {
            let (lltarget, is_cleanupret) = lltarget(this, target);
            if is_cleanupret {
                // MSVC cross-funclet jump - need a trampoline

                debug!("llblock: creating cleanup trampoline for {:?}", target);
                let name = &format!("{:?}_cleanup_trampoline_{:?}", bb, target);
                let trampoline = this.new_block(name);
                trampoline.cleanup_ret(cleanup_pad.unwrap(), Some(lltarget));
                trampoline.llbb()
            } else {
                lltarget
            }
        };

        let funclet_br = |this: &mut Self, bx: Builder, target: mir::BasicBlock| {
            let (lltarget, is_cleanupret) = lltarget(this, target);
            if is_cleanupret {
                // micro-optimization: generate a `ret` rather than a jump
                // to a trampoline.
                bx.cleanup_ret(cleanup_pad.unwrap(), Some(lltarget));
            } else {
                bx.br(lltarget);
            }
        };

        let do_call = |
            this: &mut Self,
            bx: Builder<'a, 'tcx>,
            fn_ty: FnType<'tcx, Ty<'tcx>>,
            fn_ptr: ValueRef,
            llargs: &[ValueRef],
            destination: Option<(ReturnDest<'tcx>, mir::BasicBlock)>,
            cleanup: Option<mir::BasicBlock>
        | {
            if let Some(cleanup) = cleanup {
                let ret_bx = if let Some((_, target)) = destination {
                    this.blocks[target]
                } else {
                    this.unreachable_block()
                };
                let invokeret = bx.invoke(fn_ptr,
                                           &llargs,
                                           ret_bx,
                                           llblock(this, cleanup),
                                           cleanup_bundle);
                fn_ty.apply_attrs_callsite(&bx, invokeret);

                if let Some((ret_dest, target)) = destination {
                    let ret_bx = this.build_block(target);
                    this.set_debug_loc(&ret_bx, terminator.source_info);
                    this.store_return(&ret_bx, ret_dest, &fn_ty.ret, invokeret);
                }
            } else {
                let llret = bx.call(fn_ptr, &llargs, cleanup_bundle);
                fn_ty.apply_attrs_callsite(&bx, llret);
                if this.mir[bb].is_cleanup {
                    // Cleanup is always the cold path. Don't inline
                    // drop glue. Also, when there is a deeply-nested
                    // struct, there are "symmetry" issues that cause
                    // exponential inlining - see issue #41696.
                    llvm::Attribute::NoInline.apply_callsite(llvm::AttributePlace::Function, llret);
                }

                if let Some((ret_dest, target)) = destination {
                    this.store_return(&bx, ret_dest, &fn_ty.ret, llret);
                    funclet_br(this, bx, target);
                } else {
                    bx.unreachable();
                }
            }
        };

        self.set_debug_loc(&bx, terminator.source_info);
        match terminator.kind {
            mir::TerminatorKind::Resume => {
                if let Some(cleanup_pad) = cleanup_pad {
                    bx.cleanup_ret(cleanup_pad, None);
                } else {
                    let slot = self.get_personality_slot(&bx);
                    let lp0 = slot.project_field(&bx, 0).load(&bx).immediate();
                    let lp1 = slot.project_field(&bx, 1).load(&bx).immediate();
                    slot.storage_dead(&bx);

                    if !bx.sess().target.target.options.custom_unwind_resume {
                        let mut lp = C_undef(self.landing_pad_type());
                        lp = bx.insert_value(lp, lp0, 0);
                        lp = bx.insert_value(lp, lp1, 1);
                        bx.resume(lp);
                    } else {
                        bx.call(bx.cx.eh_unwind_resume(), &[lp0], cleanup_bundle);
                        bx.unreachable();
                    }
                }
            }

            mir::TerminatorKind::Abort => {
                // Call core::intrinsics::abort()
                let fnname = bx.cx.get_intrinsic(&("llvm.trap"));
                bx.call(fnname, &[], None);
                bx.unreachable();
            }

            mir::TerminatorKind::Goto { target } => {
                funclet_br(self, bx, target);
            }

            mir::TerminatorKind::SwitchInt { ref discr, switch_ty, ref values, ref targets } => {
                let discr = self.codegen_operand(&bx, discr);
                if switch_ty == bx.tcx().types.bool {
                    let lltrue = llblock(self, targets[0]);
                    let llfalse = llblock(self, targets[1]);
                    if let [0] = values[..] {
                        bx.cond_br(discr.immediate(), llfalse, lltrue);
                    } else {
                        assert_eq!(&values[..], &[1]);
                        bx.cond_br(discr.immediate(), lltrue, llfalse);
                    }
                } else {
                    let (otherwise, targets) = targets.split_last().unwrap();
                    let switch = bx.switch(discr.immediate(),
                                            llblock(self, *otherwise), values.len());
                    let switch_llty = bx.cx.layout_of(switch_ty).immediate_llvm_type(bx.cx);
                    for (&value, target) in values.iter().zip(targets) {
                        let llval = C_uint_big(switch_llty, value);
                        let llbb = llblock(self, *target);
                        bx.add_case(switch, llval, llbb)
                    }
                }
            }

            mir::TerminatorKind::Return => {
                let llval = match self.fn_ty.ret.mode {
                    PassMode::Ignore | PassMode::Indirect(_) => {
                        bx.ret_void();
                        return;
                    }

                    PassMode::Direct(_) | PassMode::Pair(..) => {
                        let op = self.codegen_consume(&bx, &mir::Place::Local(mir::RETURN_PLACE));
                        if let Ref(llval, align) = op.val {
                            bx.load(llval, align)
                        } else {
                            op.immediate_or_packed_pair(&bx)
                        }
                    }

                    PassMode::Cast(cast_ty) => {
                        let op = match self.locals[mir::RETURN_PLACE] {
                            LocalRef::Operand(Some(op)) => op,
                            LocalRef::Operand(None) => bug!("use of return before def"),
                            LocalRef::Place(cg_place) => {
                                OperandRef {
                                    val: Ref(cg_place.llval, cg_place.align),
                                    layout: cg_place.layout
                                }
                            }
                        };
                        let llslot = match op.val {
                            Immediate(_) | Pair(..) => {
                                let scratch = PlaceRef::alloca(&bx, self.fn_ty.ret.layout, "ret");
                                op.val.store(&bx, scratch);
                                scratch.llval
                            }
                            Ref(llval, align) => {
                                assert_eq!(align.abi(), op.layout.align.abi(),
                                           "return place is unaligned!");
                                llval
                            }
                        };
                        bx.load(
                            bx.pointercast(llslot, cast_ty.llvm_type(bx.cx).ptr_to()),
                            self.fn_ty.ret.layout.align)
                    }
                };
                bx.ret(llval);
            }

            mir::TerminatorKind::Unreachable => {
                bx.unreachable();
            }

            mir::TerminatorKind::Drop { ref location, target, unwind } => {
                let ty = location.ty(self.mir, bx.tcx()).to_ty(bx.tcx());
                let ty = self.monomorphize(&ty);
                let drop_fn = monomorphize::resolve_drop_in_place(bx.cx.tcx, ty);

                if let ty::InstanceDef::DropGlue(_, None) = drop_fn.def {
                    // we don't actually need to drop anything.
                    funclet_br(self, bx, target);
                    return
                }

                let place = self.codegen_place(&bx, location);
                let mut args: &[_] = &[place.llval, place.llextra];
                args = &args[..1 + place.has_extra() as usize];
                let (drop_fn, fn_ty) = match ty.sty {
                    ty::TyDynamic(..) => {
                        let fn_ty = drop_fn.ty(bx.cx.tcx);
                        let sig = common::ty_fn_sig(bx.cx, fn_ty);
                        let sig = bx.tcx().normalize_erasing_late_bound_regions(
                            ty::ParamEnv::reveal_all(),
                            &sig,
                        );
                        let fn_ty = FnType::new_vtable(bx.cx, sig, &[]);
                        args = &args[..1];
                        (meth::DESTRUCTOR.get_fn(&bx, place.llextra, &fn_ty), fn_ty)
                    }
                    _ => {
                        (callee::get_fn(bx.cx, drop_fn),
                         FnType::of_instance(bx.cx, &drop_fn))
                    }
                };
                do_call(self, bx, fn_ty, drop_fn, args,
                        Some((ReturnDest::Nothing, target)),
                        unwind);
            }

            mir::TerminatorKind::Assert { ref cond, expected, ref msg, target, cleanup } => {
                let cond = self.codegen_operand(&bx, cond).immediate();
                let mut const_cond = common::const_to_opt_u128(cond, false).map(|c| c == 1);

                // This case can currently arise only from functions marked
                // with #[rustc_inherit_overflow_checks] and inlined from
                // another crate (mostly core::num generic/#[inline] fns),
                // while the current crate doesn't use overflow checks.
                // NOTE: Unlike binops, negation doesn't have its own
                // checked operation, just a comparison with the minimum
                // value, so we have to check for the assert message.
                if !bx.cx.check_overflow {
                    if let mir::interpret::EvalErrorKind::OverflowNeg = *msg {
                        const_cond = Some(expected);
                    }
                }

                // Don't codegen the panic block if success if known.
                if const_cond == Some(expected) {
                    funclet_br(self, bx, target);
                    return;
                }

                // Pass the condition through llvm.expect for branch hinting.
                let expect = bx.cx.get_intrinsic(&"llvm.expect.i1");
                let cond = bx.call(expect, &[cond, C_bool(bx.cx, expected)], None);

                // Create the failure block and the conditional branch to it.
                let lltarget = llblock(self, target);
                let panic_block = self.new_block("panic");
                if expected {
                    bx.cond_br(cond, lltarget, panic_block.llbb());
                } else {
                    bx.cond_br(cond, panic_block.llbb(), lltarget);
                }

                // After this point, bx is the block for the call to panic.
                bx = panic_block;
                self.set_debug_loc(&bx, terminator.source_info);

                // Get the location information.
                let loc = bx.sess().codemap().lookup_char_pos(span.lo());
                let filename = Symbol::intern(&loc.file.name.to_string()).as_str();
                let filename = C_str_slice(bx.cx, filename);
                let line = C_u32(bx.cx, loc.line as u32);
                let col = C_u32(bx.cx, loc.col.to_usize() as u32 + 1);
                let align = tcx.data_layout.aggregate_align
                    .max(tcx.data_layout.i32_align)
                    .max(tcx.data_layout.pointer_align);

                // Put together the arguments to the panic entry point.
                let (lang_item, args) = match *msg {
                    EvalErrorKind::BoundsCheck { ref len, ref index } => {
                        let len = self.codegen_operand(&mut bx, len).immediate();
                        let index = self.codegen_operand(&mut bx, index).immediate();

                        let file_line_col = C_struct(bx.cx, &[filename, line, col], false);
                        let file_line_col = consts::addr_of(bx.cx,
                                                            file_line_col,
                                                            align,
                                                            "panic_bounds_check_loc");
                        (lang_items::PanicBoundsCheckFnLangItem,
                         vec![file_line_col, index, len])
                    }
                    _ => {
                        let str = msg.description();
                        let msg_str = Symbol::intern(str).as_str();
                        let msg_str = C_str_slice(bx.cx, msg_str);
                        let msg_file_line_col = C_struct(bx.cx,
                                                     &[msg_str, filename, line, col],
                                                     false);
                        let msg_file_line_col = consts::addr_of(bx.cx,
                                                                msg_file_line_col,
                                                                align,
                                                                "panic_loc");
                        (lang_items::PanicFnLangItem,
                         vec![msg_file_line_col])
                    }
                };

                // Obtain the panic entry point.
                let def_id = common::langcall(bx.tcx(), Some(span), "", lang_item);
                let instance = ty::Instance::mono(bx.tcx(), def_id);
                let fn_ty = FnType::of_instance(bx.cx, &instance);
                let llfn = callee::get_fn(bx.cx, instance);

                // Codegen the actual panic invoke/call.
                do_call(self, bx, fn_ty, llfn, &args, None, cleanup);
            }

            mir::TerminatorKind::DropAndReplace { .. } => {
                bug!("undesugared DropAndReplace in codegen: {:?}", terminator);
            }

            mir::TerminatorKind::Call { ref func, ref args, ref destination, cleanup } => {
                // Create the callee. This is a fn ptr or zero-sized and hence a kind of scalar.
                let callee = self.codegen_operand(&bx, func);

                let (instance, mut llfn) = match callee.layout.ty.sty {
                    ty::TyFnDef(def_id, substs) => {
                        (Some(ty::Instance::resolve(bx.cx.tcx,
                                                    ty::ParamEnv::reveal_all(),
                                                    def_id,
                                                    substs).unwrap()),
                         None)
                    }
                    ty::TyFnPtr(_) => {
                        (None, Some(callee.immediate()))
                    }
                    _ => bug!("{} is not callable", callee.layout.ty)
                };
                let def = instance.map(|i| i.def);
                let sig = callee.layout.ty.fn_sig(bx.tcx());
                let sig = bx.tcx().normalize_erasing_late_bound_regions(
                    ty::ParamEnv::reveal_all(),
                    &sig,
                );
                let abi = sig.abi;

                // Handle intrinsics old codegen wants Expr's for, ourselves.
                let intrinsic = match def {
                    Some(ty::InstanceDef::Intrinsic(def_id))
                        => Some(bx.tcx().item_name(def_id).as_str()),
                    _ => None
                };
                let intrinsic = intrinsic.as_ref().map(|s| &s[..]);

                if intrinsic == Some("transmute") {
                    if let Some(destination_ref) = destination.as_ref() {
                        let &(ref dest, target) = destination_ref;
                        self.trans_transmute(&bx, &args[0], dest);
                        funclet_br(self, bx, target);
                    } else {
                        // If we are trying to transmute to an uninhabited type,
                        // it is likely there is no allotted destination. In fact,
                        // transmuting to an uninhabited type is UB, which means
                        // we can do what we like. Here, we declare that transmuting
                        // into an uninhabited type is impossible, so anything following
                        // it must be unreachable.
                        bx.unreachable();
                    }
                    return;
                }

                let extra_args = &args[sig.inputs().len()..];
                let extra_args = extra_args.iter().map(|op_arg| {
                    let op_ty = op_arg.ty(self.mir, bx.tcx());
                    self.monomorphize(&op_ty)
                }).collect::<Vec<_>>();

                let fn_ty = match def {
                    Some(ty::InstanceDef::Virtual(..)) => {
                        FnType::new_vtable(bx.cx, sig, &extra_args)
                    }
                    Some(ty::InstanceDef::DropGlue(_, None)) => {
                        // empty drop glue - a nop.
                        let &(_, target) = destination.as_ref().unwrap();
                        funclet_br(self, bx, target);
                        return;
                    }
                    _ => FnType::new(bx.cx, sig, &extra_args)
                };

                // The arguments we'll be passing. Plus one to account for outptr, if used.
                let arg_count = fn_ty.args.len() + fn_ty.ret.is_indirect() as usize;
                let mut llargs = Vec::with_capacity(arg_count);

                // Prepare the return value destination
                let ret_dest = if let Some((ref dest, _)) = *destination {
                    let is_intrinsic = intrinsic.is_some();
                    self.make_return_dest(&bx, dest, &fn_ty.ret, &mut llargs,
                                          is_intrinsic)
                } else {
                    ReturnDest::Nothing
                };

                if intrinsic.is_some() && intrinsic != Some("drop_in_place") {
                    use intrinsic::codegen_intrinsic_call;

                    let dest = match ret_dest {
                        _ if fn_ty.ret.is_indirect() => llargs[0],
                        ReturnDest::Nothing => {
                            C_undef(fn_ty.ret.memory_ty(bx.cx).ptr_to())
                        }
                        ReturnDest::IndirectOperand(dst, _) |
                        ReturnDest::Store(dst) => dst.llval,
                        ReturnDest::DirectOperand(_) =>
                            bug!("Cannot use direct operand with an intrinsic call")
                    };

                    let args: Vec<_> = args.iter().enumerate().map(|(i, arg)| {
                        // The indices passed to simd_shuffle* in the
                        // third argument must be constant. This is
                        // checked by const-qualification, which also
                        // promotes any complex rvalues to constants.
                        if i == 2 && intrinsic.unwrap().starts_with("simd_shuffle") {
                            match *arg {
                                mir::Operand::Copy(_) |
                                mir::Operand::Move(_) => {
                                    span_bug!(span, "shuffle indices must be constant");
                                }
                                mir::Operand::Constant(ref constant) => {
                                    let (llval, ty) = self.simd_shuffle_indices(
                                        &bx,
                                        constant,
                                    );
                                    return OperandRef {
                                        val: Immediate(llval),
                                        layout: bx.cx.layout_of(ty)
                                    };
                                }
                            }
                        }

                        self.codegen_operand(&bx, arg)
                    }).collect();


                    let callee_ty = instance.as_ref().unwrap().ty(bx.cx.tcx);
                    codegen_intrinsic_call(&bx, callee_ty, &fn_ty, &args, dest,
                                         terminator.source_info.span);

                    if let ReturnDest::IndirectOperand(dst, _) = ret_dest {
                        self.store_return(&bx, ret_dest, &fn_ty.ret, dst.llval);
                    }

                    if let Some((_, target)) = *destination {
                        funclet_br(self, bx, target);
                    } else {
                        bx.unreachable();
                    }

                    return;
                }

                // Split the rust-call tupled arguments off.
                let (first_args, untuple) = if abi == Abi::RustCall && !args.is_empty() {
                    let (tup, args) = args.split_last().unwrap();
                    (args, Some(tup))
                } else {
                    (&args[..], None)
                };

                for (i, arg) in first_args.iter().enumerate() {
                    let mut op = self.codegen_operand(&bx, arg);
                    if let (0, Some(ty::InstanceDef::Virtual(_, idx))) = (i, def) {
                        if let Pair(data_ptr, meta) = op.val {
                            llfn = Some(meth::VirtualIndex::from_index(idx)
                                .get_fn(&bx, meta, &fn_ty));
                            llargs.push(data_ptr);
                            continue;
                        }
                    }

                    // The callee needs to own the argument memory if we pass it
                    // by-ref, so make a local copy of non-immediate constants.
                    match (arg, op.val) {
                        (&mir::Operand::Copy(_), Ref(..)) |
                        (&mir::Operand::Constant(_), Ref(..)) => {
                            let tmp = PlaceRef::alloca(&bx, op.layout, "const");
                            op.val.store(&bx, tmp);
                            op.val = Ref(tmp.llval, tmp.align);
                        }
                        _ => {}
                    }

                    self.codegen_argument(&bx, op, &mut llargs, &fn_ty.args[i]);
                }
                if let Some(tup) = untuple {
                    self.codegen_arguments_untupled(&bx, tup, &mut llargs,
                        &fn_ty.args[first_args.len()..])
                }

                let fn_ptr = match (llfn, instance) {
                    (Some(llfn), _) => llfn,
                    (None, Some(instance)) => callee::get_fn(bx.cx, instance),
                    _ => span_bug!(span, "no llfn for call"),
                };

                do_call(self, bx, fn_ty, fn_ptr, &llargs,
                        destination.as_ref().map(|&(_, target)| (ret_dest, target)),
                        cleanup);
            }
            mir::TerminatorKind::GeneratorDrop |
            mir::TerminatorKind::Yield { .. } => bug!("generator ops in codegen"),
            mir::TerminatorKind::FalseEdges { .. } |
            mir::TerminatorKind::FalseUnwind { .. } => bug!("borrowck false edges in codegen"),
        }
    }

    fn codegen_argument(&mut self,
                      bx: &Builder<'a, 'tcx>,
                      op: OperandRef<'tcx>,
                      llargs: &mut Vec<ValueRef>,
                      arg: &ArgType<'tcx, Ty<'tcx>>) {
        // Fill padding with undef value, where applicable.
        if let Some(ty) = arg.pad {
            llargs.push(C_undef(ty.llvm_type(bx.cx)));
        }

        if arg.is_ignore() {
            return;
        }

        if let PassMode::Pair(..) = arg.mode {
            match op.val {
                Pair(a, b) => {
                    llargs.push(a);
                    llargs.push(b);
                    return;
                }
                _ => bug!("codegen_argument: {:?} invalid for pair arugment", op)
            }
        }

        // Force by-ref if we have to load through a cast pointer.
        let (mut llval, align, by_ref) = match op.val {
            Immediate(_) | Pair(..) => {
                match arg.mode {
                    PassMode::Indirect(_) | PassMode::Cast(_) => {
                        let scratch = PlaceRef::alloca(bx, arg.layout, "arg");
                        op.val.store(bx, scratch);
                        (scratch.llval, scratch.align, true)
                    }
                    _ => {
                        (op.immediate_or_packed_pair(bx), arg.layout.align, false)
                    }
                }
            }
            Ref(llval, align) => {
                if arg.is_indirect() && align.abi() < arg.layout.align.abi() {
                    // `foo(packed.large_field)`. We can't pass the (unaligned) field directly. I
                    // think that ATM (Rust 1.16) we only pass temporaries, but we shouldn't
                    // have scary latent bugs around.

                    let scratch = PlaceRef::alloca(bx, arg.layout, "arg");
                    base::memcpy_ty(bx, scratch.llval, llval, op.layout, align, MemFlags::empty());
                    (scratch.llval, scratch.align, true)
                } else {
                    (llval, align, true)
                }
            }
        };

        if by_ref && !arg.is_indirect() {
            // Have to load the argument, maybe while casting it.
            if let PassMode::Cast(ty) = arg.mode {
                llval = bx.load(bx.pointercast(llval, ty.llvm_type(bx.cx).ptr_to()),
                                 align.min(arg.layout.align));
            } else {
                // We can't use `PlaceRef::load` here because the argument
                // may have a type we don't treat as immediate, but the ABI
                // used for this call is passing it by-value. In that case,
                // the load would just produce `OperandValue::Ref` instead
                // of the `OperandValue::Immediate` we need for the call.
                llval = bx.load(llval, align);
                if let layout::Abi::Scalar(ref scalar) = arg.layout.abi {
                    if scalar.is_bool() {
                        bx.range_metadata(llval, 0..2);
                    }
                }
                // We store bools as i8 so we need to truncate to i1.
                llval = base::to_immediate(bx, llval, arg.layout);
            }
        }

        llargs.push(llval);
    }

    fn codegen_arguments_untupled(&mut self,
                                bx: &Builder<'a, 'tcx>,
                                operand: &mir::Operand<'tcx>,
                                llargs: &mut Vec<ValueRef>,
                                args: &[ArgType<'tcx, Ty<'tcx>>]) {
        let tuple = self.codegen_operand(bx, operand);

        // Handle both by-ref and immediate tuples.
        if let Ref(llval, align) = tuple.val {
            let tuple_ptr = PlaceRef::new_sized(llval, tuple.layout, align);
            for i in 0..tuple.layout.fields.count() {
                let field_ptr = tuple_ptr.project_field(bx, i);
                self.codegen_argument(bx, field_ptr.load(bx), llargs, &args[i]);
            }
        } else {
            // If the tuple is immediate, the elements are as well.
            for i in 0..tuple.layout.fields.count() {
                let op = tuple.extract_field(bx, i);
                self.codegen_argument(bx, op, llargs, &args[i]);
            }
        }
    }

    fn get_personality_slot(&mut self, bx: &Builder<'a, 'tcx>) -> PlaceRef<'tcx> {
        let cx = bx.cx;
        if let Some(slot) = self.personality_slot {
            slot
        } else {
            let layout = cx.layout_of(cx.tcx.intern_tup(&[
                cx.tcx.mk_mut_ptr(cx.tcx.types.u8),
                cx.tcx.types.i32
            ]));
            let slot = PlaceRef::alloca(bx, layout, "personalityslot");
            self.personality_slot = Some(slot);
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

        let block = self.blocks[target_bb];
        let landing_pad = self.landing_pad_uncached(block);
        self.landing_pads[target_bb] = Some(landing_pad);
        landing_pad
    }

    fn landing_pad_uncached(&mut self, target_bb: BasicBlockRef) -> BasicBlockRef {
        if base::wants_msvc_seh(self.cx.sess()) {
            span_bug!(self.mir.span, "landing pad was not inserted?")
        }

        let bx = self.new_block("cleanup");

        let llpersonality = self.cx.eh_personality();
        let llretty = self.landing_pad_type();
        let lp = bx.landing_pad(llretty, llpersonality, 1);
        bx.set_cleanup(lp);

        let slot = self.get_personality_slot(&bx);
        slot.storage_live(&bx);
        Pair(bx.extract_value(lp, 0), bx.extract_value(lp, 1)).store(&bx, slot);

        bx.br(target_bb);
        bx.llbb()
    }

    fn landing_pad_type(&self) -> Type {
        let cx = self.cx;
        Type::struct_(cx, &[Type::i8p(cx), Type::i32(cx)], false)
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
        Builder::new_block(self.cx, self.llfn, name)
    }

    pub fn build_block(&self, bb: mir::BasicBlock) -> Builder<'a, 'tcx> {
        let bx = Builder::with_cx(self.cx);
        bx.position_at_end(self.blocks[bb]);
        bx
    }

    fn make_return_dest(&mut self, bx: &Builder<'a, 'tcx>,
                        dest: &mir::Place<'tcx>, fn_ret: &ArgType<'tcx, Ty<'tcx>>,
                        llargs: &mut Vec<ValueRef>, is_intrinsic: bool)
                        -> ReturnDest<'tcx> {
        // If the return is ignored, we can just return a do-nothing ReturnDest
        if fn_ret.is_ignore() {
            return ReturnDest::Nothing;
        }
        let dest = if let mir::Place::Local(index) = *dest {
            match self.locals[index] {
                LocalRef::Place(dest) => dest,
                LocalRef::Operand(None) => {
                    // Handle temporary places, specifically Operand ones, as
                    // they don't have allocas
                    return if fn_ret.is_indirect() {
                        // Odd, but possible, case, we have an operand temporary,
                        // but the calling convention has an indirect return.
                        let tmp = PlaceRef::alloca(bx, fn_ret.layout, "tmp_ret");
                        tmp.storage_live(bx);
                        llargs.push(tmp.llval);
                        ReturnDest::IndirectOperand(tmp, index)
                    } else if is_intrinsic {
                        // Currently, intrinsics always need a location to store
                        // the result. so we create a temporary alloca for the
                        // result
                        let tmp = PlaceRef::alloca(bx, fn_ret.layout, "tmp_ret");
                        tmp.storage_live(bx);
                        ReturnDest::IndirectOperand(tmp, index)
                    } else {
                        ReturnDest::DirectOperand(index)
                    };
                }
                LocalRef::Operand(Some(_)) => {
                    bug!("place local already assigned to");
                }
            }
        } else {
            self.codegen_place(bx, dest)
        };
        if fn_ret.is_indirect() {
            if dest.align.abi() < dest.layout.align.abi() {
                // Currently, MIR code generation does not create calls
                // that store directly to fields of packed structs (in
                // fact, the calls it creates write only to temps),
                //
                // If someone changes that, please update this code path
                // to create a temporary.
                span_bug!(self.mir.span, "can't directly store to unaligned value");
            }
            llargs.push(dest.llval);
            ReturnDest::Nothing
        } else {
            ReturnDest::Store(dest)
        }
    }

    fn codegen_transmute(&mut self, bx: &Builder<'a, 'tcx>,
                       src: &mir::Operand<'tcx>,
                       dst: &mir::Place<'tcx>) {
        if let mir::Place::Local(index) = *dst {
            match self.locals[index] {
                LocalRef::Place(place) => self.codegen_transmute_into(bx, src, place),
                LocalRef::Operand(None) => {
                    let dst_layout = bx.cx.layout_of(self.monomorphized_place_ty(dst));
                    assert!(!dst_layout.ty.has_erasable_regions());
                    let place = PlaceRef::alloca(bx, dst_layout, "transmute_temp");
                    place.storage_live(bx);
                    self.codegen_transmute_into(bx, src, place);
                    let op = place.load(bx);
                    place.storage_dead(bx);
                    self.locals[index] = LocalRef::Operand(Some(op));
                }
                LocalRef::Operand(Some(op)) => {
                    assert!(op.layout.is_zst(),
                            "assigning to initialized SSAtemp");
                }
            }
        } else {
            let dst = self.codegen_place(bx, dst);
            self.codegen_transmute_into(bx, src, dst);
        }
    }

    fn codegen_transmute_into(&mut self, bx: &Builder<'a, 'tcx>,
                            src: &mir::Operand<'tcx>,
                            dst: PlaceRef<'tcx>) {
        let src = self.codegen_operand(bx, src);
        let llty = src.layout.llvm_type(bx.cx);
        let cast_ptr = bx.pointercast(dst.llval, llty.ptr_to());
        let align = src.layout.align.min(dst.layout.align);
        src.val.store(bx, PlaceRef::new_sized(cast_ptr, src.layout, align));
    }


    // Stores the return value of a function call into it's final location.
    fn store_return(&mut self,
                    bx: &Builder<'a, 'tcx>,
                    dest: ReturnDest<'tcx>,
                    ret_ty: &ArgType<'tcx, Ty<'tcx>>,
                    llval: ValueRef) {
        use self::ReturnDest::*;

        match dest {
            Nothing => (),
            Store(dst) => ret_ty.store(bx, llval, dst),
            IndirectOperand(tmp, index) => {
                let op = tmp.load(bx);
                tmp.storage_dead(bx);
                self.locals[index] = LocalRef::Operand(Some(op));
            }
            DirectOperand(index) => {
                // If there is a cast, we have to store and reload.
                let op = if let PassMode::Cast(_) = ret_ty.mode {
                    let tmp = PlaceRef::alloca(bx, ret_ty.layout, "tmp_ret");
                    tmp.storage_live(bx);
                    ret_ty.store(bx, llval, tmp);
                    let op = tmp.load(bx);
                    tmp.storage_dead(bx);
                    op
                } else {
                    OperandRef::from_immediate_or_packed_pair(bx, llval, ret_ty.layout)
                };
                self.locals[index] = LocalRef::Operand(Some(op));
            }
        }
    }
}

enum ReturnDest<'tcx> {
    // Do nothing, the return value is indirect or ignored
    Nothing,
    // Store the return value to the pointer
    Store(PlaceRef<'tcx>),
    // Stores an indirect return value to an operand local place
    IndirectOperand(PlaceRef<'tcx>, mir::Local),
    // Stores a direct return value to an operand local place
    DirectOperand(mir::Local)
}
