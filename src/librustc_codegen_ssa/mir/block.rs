use rustc::middle::lang_items;
use rustc::ty::{self, Ty, TypeFoldable, Instance};
use rustc::ty::layout::{self, LayoutOf, HasTyCtxt, FnTypeExt};
use rustc::mir::{self, Place, PlaceBase, Static, StaticKind};
use rustc::mir::interpret::InterpError;
use rustc_target::abi::call::{ArgType, FnType, PassMode, IgnoreMode};
use rustc_target::spec::abi::Abi;
use crate::base;
use crate::MemFlags;
use crate::common::{self, IntPredicate};
use crate::meth;

use crate::traits::*;

use std::borrow::Cow;

use syntax::symbol::LocalInternedString;
use syntax_pos::Pos;

use super::{FunctionCx, LocalRef};
use super::place::PlaceRef;
use super::operand::OperandRef;
use super::operand::OperandValue::{Pair, Ref, Immediate};

/// Used by `FunctionCx::codegen_terminator` for emitting common patterns
/// e.g., creating a basic block, calling a function, etc.
struct TerminatorCodegenHelper<'a, 'tcx> {
    bb: &'a mir::BasicBlock,
    terminator: &'a mir::Terminator<'tcx>,
    funclet_bb: Option<mir::BasicBlock>,
}

impl<'a, 'tcx> TerminatorCodegenHelper<'a, 'tcx> {
    /// Returns the associated funclet from `FunctionCx::funclets` for the
    /// `funclet_bb` member if it is not `None`.
    fn funclet<'c, 'b, Bx: BuilderMethods<'b, 'tcx>>(
        &self,
        fx: &'c mut FunctionCx<'b, 'tcx, Bx>,
    ) -> Option<&'c Bx::Funclet> {
        match self.funclet_bb {
            Some(funcl) => fx.funclets[funcl].as_ref(),
            None => None,
        }
    }

    fn lltarget<'b, 'c, Bx: BuilderMethods<'b, 'tcx>>(
        &self,
        fx: &'c mut FunctionCx<'b, 'tcx, Bx>,
        target: mir::BasicBlock,
    ) -> (Bx::BasicBlock, bool) {
        let span = self.terminator.source_info.span;
        let lltarget = fx.blocks[target];
        let target_funclet = fx.cleanup_kinds[target].funclet_bb(target);
        match (self.funclet_bb, target_funclet) {
            (None, None) => (lltarget, false),
            (Some(f), Some(t_f)) if f == t_f || !base::wants_msvc_seh(fx.cx.tcx().sess) =>
                (lltarget, false),
            // jump *into* cleanup - need a landing pad if GNU
            (None, Some(_)) => (fx.landing_pad_to(target), false),
            (Some(_), None) => span_bug!(span, "{:?} - jump out of cleanup?", self.terminator),
            (Some(_), Some(_)) => (fx.landing_pad_to(target), true),
        }
    }

    /// Create a basic block.
    fn llblock<'c, 'b, Bx: BuilderMethods<'b, 'tcx>>(
        &self,
        fx: &'c mut FunctionCx<'b, 'tcx, Bx>,
        target: mir::BasicBlock,
    ) -> Bx::BasicBlock {
        let (lltarget, is_cleanupret) = self.lltarget(fx, target);
        if is_cleanupret {
            // MSVC cross-funclet jump - need a trampoline

            debug!("llblock: creating cleanup trampoline for {:?}", target);
            let name = &format!("{:?}_cleanup_trampoline_{:?}", self.bb, target);
            let mut trampoline = fx.new_block(name);
            trampoline.cleanup_ret(self.funclet(fx).unwrap(),
                                   Some(lltarget));
            trampoline.llbb()
        } else {
            lltarget
        }
    }

    fn funclet_br<'c, 'b, Bx: BuilderMethods<'b, 'tcx>>(
        &self,
        fx: &'c mut FunctionCx<'b, 'tcx, Bx>,
        bx: &mut Bx,
        target: mir::BasicBlock,
    ) {
        let (lltarget, is_cleanupret) = self.lltarget(fx, target);
        if is_cleanupret {
            // micro-optimization: generate a `ret` rather than a jump
            // to a trampoline.
            bx.cleanup_ret(self.funclet(fx).unwrap(), Some(lltarget));
        } else {
            bx.br(lltarget);
        }
    }

    /// Call `fn_ptr` of `fn_ty` with the arguments `llargs`, the optional
    /// return destination `destination` and the cleanup function `cleanup`.
    fn do_call<'c, 'b, Bx: BuilderMethods<'b, 'tcx>>(
        &self,
        fx: &'c mut FunctionCx<'b, 'tcx, Bx>,
        bx: &mut Bx,
        fn_ty: FnType<'tcx, Ty<'tcx>>,
        fn_ptr: Bx::Value,
        llargs: &[Bx::Value],
        destination: Option<(ReturnDest<'tcx, Bx::Value>, mir::BasicBlock)>,
        cleanup: Option<mir::BasicBlock>,
    ) {
        if let Some(cleanup) = cleanup {
            let ret_bx = if let Some((_, target)) = destination {
                fx.blocks[target]
            } else {
                fx.unreachable_block()
            };
            let invokeret = bx.invoke(fn_ptr,
                                      &llargs,
                                      ret_bx,
                                      self.llblock(fx, cleanup),
                                      self.funclet(fx));
            bx.apply_attrs_callsite(&fn_ty, invokeret);

            if let Some((ret_dest, target)) = destination {
                let mut ret_bx = fx.build_block(target);
                fx.set_debug_loc(&mut ret_bx, self.terminator.source_info);
                fx.store_return(&mut ret_bx, ret_dest, &fn_ty.ret, invokeret);
            }
        } else {
            let llret = bx.call(fn_ptr, &llargs, self.funclet(fx));
            bx.apply_attrs_callsite(&fn_ty, llret);
            if fx.mir[*self.bb].is_cleanup {
                // Cleanup is always the cold path. Don't inline
                // drop glue. Also, when there is a deeply-nested
                // struct, there are "symmetry" issues that cause
                // exponential inlining - see issue #41696.
                bx.do_not_inline(llret);
            }

            if let Some((ret_dest, target)) = destination {
                fx.store_return(bx, ret_dest, &fn_ty.ret, llret);
                self.funclet_br(fx, bx, target);
            } else {
                bx.unreachable();
            }
        }
    }
}

/// Codegen implementations for some terminator variants.
impl<'a, 'tcx, Bx: BuilderMethods<'a, 'tcx>> FunctionCx<'a, 'tcx, Bx> {
    /// Generates code for a `Resume` terminator.
    fn codegen_resume_terminator<'b>(
        &mut self,
        helper: TerminatorCodegenHelper<'b, 'tcx>,
        mut bx: Bx,
    ) {
        if let Some(funclet) = helper.funclet(self) {
            bx.cleanup_ret(funclet, None);
        } else {
            let slot = self.get_personality_slot(&mut bx);
            let lp0 = slot.project_field(&mut bx, 0);
            let lp0 = bx.load_operand(lp0).immediate();
            let lp1 = slot.project_field(&mut bx, 1);
            let lp1 = bx.load_operand(lp1).immediate();
            slot.storage_dead(&mut bx);

            if !bx.sess().target.target.options.custom_unwind_resume {
                let mut lp = bx.const_undef(self.landing_pad_type());
                lp = bx.insert_value(lp, lp0, 0);
                lp = bx.insert_value(lp, lp1, 1);
                bx.resume(lp);
            } else {
                bx.call(bx.eh_unwind_resume(), &[lp0],
                        helper.funclet(self));
                bx.unreachable();
            }
        }
    }

    fn codegen_switchint_terminator<'b>(
        &mut self,
        helper: TerminatorCodegenHelper<'b, 'tcx>,
        mut bx: Bx,
        discr: &mir::Operand<'tcx>,
        switch_ty: Ty<'tcx>,
        values: &Cow<'tcx, [u128]>,
        targets: &Vec<mir::BasicBlock>,
    ) {
        let discr = self.codegen_operand(&mut bx, &discr);
        if targets.len() == 2 {
            // If there are two targets, emit br instead of switch
            let lltrue = helper.llblock(self, targets[0]);
            let llfalse = helper.llblock(self, targets[1]);
            if switch_ty == bx.tcx().types.bool {
                // Don't generate trivial icmps when switching on bool
                if let [0] = values[..] {
                    bx.cond_br(discr.immediate(), llfalse, lltrue);
                } else {
                    assert_eq!(&values[..], &[1]);
                    bx.cond_br(discr.immediate(), lltrue, llfalse);
                }
            } else {
                let switch_llty = bx.immediate_backend_type(
                    bx.layout_of(switch_ty)
                );
                let llval = bx.const_uint_big(switch_llty, values[0]);
                let cmp = bx.icmp(IntPredicate::IntEQ, discr.immediate(), llval);
                bx.cond_br(cmp, lltrue, llfalse);
            }
        } else {
            let (otherwise, targets) = targets.split_last().unwrap();
            bx.switch(
                discr.immediate(),
                helper.llblock(self, *otherwise),
                values.iter().zip(targets).map(|(&value, target)| {
                    (value, helper.llblock(self, *target))
                })
            );
        }
    }

    fn codegen_return_terminator(&mut self, mut bx: Bx) {
        if self.fn_ty.c_variadic {
            match self.va_list_ref {
                Some(va_list) => {
                    bx.va_end(va_list.llval);
                }
                None => {
                    bug!("C-variadic function must have a `va_list_ref`");
                }
            }
        }
        if self.fn_ty.ret.layout.abi.is_uninhabited() {
            // Functions with uninhabited return values are marked `noreturn`,
            // so we should make sure that we never actually do.
            bx.abort();
            bx.unreachable();
            return;
        }
        let llval = match self.fn_ty.ret.mode {
            PassMode::Ignore(IgnoreMode::Zst) | PassMode::Indirect(..) => {
                bx.ret_void();
                return;
            }

            PassMode::Ignore(IgnoreMode::CVarArgs) => {
                bug!("C-variadic arguments should never be the return type");
            }

            PassMode::Direct(_) | PassMode::Pair(..) => {
                let op =
                    self.codegen_consume(&mut bx, &mir::Place::RETURN_PLACE);
                if let Ref(llval, _, align) = op.val {
                    bx.load(llval, align)
                } else {
                    op.immediate_or_packed_pair(&mut bx)
                }
            }

            PassMode::Cast(cast_ty) => {
                let op = match self.locals[mir::RETURN_PLACE] {
                    LocalRef::Operand(Some(op)) => op,
                    LocalRef::Operand(None) => bug!("use of return before def"),
                    LocalRef::Place(cg_place) => {
                        OperandRef {
                            val: Ref(cg_place.llval, None, cg_place.align),
                            layout: cg_place.layout
                        }
                    }
                    LocalRef::UnsizedPlace(_) => bug!("return type must be sized"),
                };
                let llslot = match op.val {
                    Immediate(_) | Pair(..) => {
                        let scratch =
                            PlaceRef::alloca(&mut bx, self.fn_ty.ret.layout, "ret");
                        op.val.store(&mut bx, scratch);
                        scratch.llval
                    }
                    Ref(llval, _, align) => {
                        assert_eq!(align, op.layout.align.abi,
                                   "return place is unaligned!");
                        llval
                    }
                };
                let addr = bx.pointercast(llslot, bx.type_ptr_to(
                    bx.cast_backend_type(&cast_ty)
                ));
                bx.load(addr, self.fn_ty.ret.layout.align.abi)
            }
        };
        bx.ret(llval);
    }


    fn codegen_drop_terminator<'b>(
        &mut self,
        helper: TerminatorCodegenHelper<'b, 'tcx>,
        mut bx: Bx,
        location: &mir::Place<'tcx>,
        target: mir::BasicBlock,
        unwind: Option<mir::BasicBlock>,
    ) {
        let ty = location.ty(self.mir, bx.tcx()).ty;
        let ty = self.monomorphize(&ty);
        let drop_fn = Instance::resolve_drop_in_place(bx.tcx(), ty);

        if let ty::InstanceDef::DropGlue(_, None) = drop_fn.def {
            // we don't actually need to drop anything.
            helper.funclet_br(self, &mut bx, target);
            return
        }

        let place = self.codegen_place(&mut bx, location);
        let (args1, args2);
        let mut args = if let Some(llextra) = place.llextra {
            args2 = [place.llval, llextra];
            &args2[..]
        } else {
            args1 = [place.llval];
            &args1[..]
        };
        let (drop_fn, fn_ty) = match ty.sty {
            ty::Dynamic(..) => {
                let sig = drop_fn.fn_sig(self.cx.tcx());
                let sig = self.cx.tcx().normalize_erasing_late_bound_regions(
                    ty::ParamEnv::reveal_all(),
                    &sig,
                );
                let fn_ty = FnType::new_vtable(&bx, sig, &[]);
                let vtable = args[1];
                args = &args[..1];
                (meth::DESTRUCTOR.get_fn(&mut bx, vtable, &fn_ty), fn_ty)
            }
            _ => {
                (bx.get_fn(drop_fn),
                 FnType::of_instance(&bx, &drop_fn))
            }
        };
        helper.do_call(self, &mut bx, fn_ty, drop_fn, args,
                       Some((ReturnDest::Nothing, target)),
                       unwind);
    }

    fn codegen_assert_terminator<'b>(
        &mut self,
        helper: TerminatorCodegenHelper<'b, 'tcx>,
        mut bx: Bx,
        terminator: &mir::Terminator<'tcx>,
        cond: &mir::Operand<'tcx>,
        expected: bool,
        msg: &mir::AssertMessage<'tcx>,
        target: mir::BasicBlock,
        cleanup: Option<mir::BasicBlock>,
    ) {
        let span = terminator.source_info.span;
        let cond = self.codegen_operand(&mut bx, cond).immediate();
        let mut const_cond = bx.const_to_opt_u128(cond, false).map(|c| c == 1);

        // This case can currently arise only from functions marked
        // with #[rustc_inherit_overflow_checks] and inlined from
        // another crate (mostly core::num generic/#[inline] fns),
        // while the current crate doesn't use overflow checks.
        // NOTE: Unlike binops, negation doesn't have its own
        // checked operation, just a comparison with the minimum
        // value, so we have to check for the assert message.
        if !bx.check_overflow() {
            if let mir::interpret::InterpError::OverflowNeg = *msg {
                const_cond = Some(expected);
            }
        }

        // Don't codegen the panic block if success if known.
        if const_cond == Some(expected) {
            helper.funclet_br(self, &mut bx, target);
            return;
        }

        // Pass the condition through llvm.expect for branch hinting.
        let cond = bx.expect(cond, expected);

        // Create the failure block and the conditional branch to it.
        let lltarget = helper.llblock(self, target);
        let panic_block = self.new_block("panic");
        if expected {
            bx.cond_br(cond, lltarget, panic_block.llbb());
        } else {
            bx.cond_br(cond, panic_block.llbb(), lltarget);
        }

        // After this point, bx is the block for the call to panic.
        bx = panic_block;
        self.set_debug_loc(&mut bx, terminator.source_info);

        // Get the location information.
        let loc = bx.sess().source_map().lookup_char_pos(span.lo());
        let filename = LocalInternedString::intern(&loc.file.name.to_string());
        let line = bx.const_u32(loc.line as u32);
        let col = bx.const_u32(loc.col.to_usize() as u32 + 1);

        // Put together the arguments to the panic entry point.
        let (lang_item, args) = match *msg {
            InterpError::BoundsCheck { ref len, ref index } => {
                let len = self.codegen_operand(&mut bx, len).immediate();
                let index = self.codegen_operand(&mut bx, index).immediate();

                let file_line_col = bx.static_panic_msg(
                    None,
                    filename,
                    line,
                    col,
                    "panic_bounds_check_loc",
                );
                (lang_items::PanicBoundsCheckFnLangItem,
                    vec![file_line_col, index, len])
            }
            _ => {
                let str = msg.description();
                let msg_str = LocalInternedString::intern(str);
                let msg_file_line_col = bx.static_panic_msg(
                    Some(msg_str),
                    filename,
                    line,
                    col,
                    "panic_loc",
                );
                (lang_items::PanicFnLangItem,
                    vec![msg_file_line_col])
            }
        };

        // Obtain the panic entry point.
        let def_id = common::langcall(bx.tcx(), Some(span), "", lang_item);
        let instance = ty::Instance::mono(bx.tcx(), def_id);
        let fn_ty = FnType::of_instance(&bx, &instance);
        let llfn = bx.get_fn(instance);

        // Codegen the actual panic invoke/call.
        helper.do_call(self, &mut bx, fn_ty, llfn, &args, None, cleanup);
    }

    fn codegen_call_terminator<'b>(
        &mut self,
        helper: TerminatorCodegenHelper<'b, 'tcx>,
        mut bx: Bx,
        terminator: &mir::Terminator<'tcx>,
        func: &mir::Operand<'tcx>,
        args: &Vec<mir::Operand<'tcx>>,
        destination: &Option<(mir::Place<'tcx>, mir::BasicBlock)>,
        cleanup: Option<mir::BasicBlock>,
    ) {
        let span = terminator.source_info.span;
        // Create the callee. This is a fn ptr or zero-sized and hence a kind of scalar.
        let callee = self.codegen_operand(&mut bx, func);

        let (instance, mut llfn) = match callee.layout.ty.sty {
            ty::FnDef(def_id, substs) => {
                (Some(ty::Instance::resolve(bx.tcx(),
                                            ty::ParamEnv::reveal_all(),
                                            def_id,
                                            substs).unwrap()),
                 None)
            }
            ty::FnPtr(_) => {
                (None, Some(callee.immediate()))
            }
            _ => bug!("{} is not callable", callee.layout.ty),
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
            Some(ty::InstanceDef::Intrinsic(def_id)) =>
                Some(bx.tcx().item_name(def_id).as_str()),
            _ => None
        };
        let intrinsic = intrinsic.as_ref().map(|s| &s[..]);

        if intrinsic == Some("transmute") {
            if let Some(destination_ref) = destination.as_ref() {
                let &(ref dest, target) = destination_ref;
                self.codegen_transmute(&mut bx, &args[0], dest);
                helper.funclet_br(self, &mut bx, target);
            } else {
                // If we are trying to transmute to an uninhabited type,
                // it is likely there is no allotted destination. In fact,
                // transmuting to an uninhabited type is UB, which means
                // we can do what we like. Here, we declare that transmuting
                // into an uninhabited type is impossible, so anything following
                // it must be unreachable.
                assert_eq!(bx.layout_of(sig.output()).abi, layout::Abi::Uninhabited);
                bx.unreachable();
            }
            return;
        }

        // The "spoofed" `VaListImpl` added to a C-variadic functions signature
        // should not be included in the `extra_args` calculation.
        let extra_args_start_idx = sig.inputs().len() - if sig.c_variadic { 1 } else { 0 };
        let extra_args = &args[extra_args_start_idx..];
        let extra_args = extra_args.iter().map(|op_arg| {
            let op_ty = op_arg.ty(self.mir, bx.tcx());
            self.monomorphize(&op_ty)
        }).collect::<Vec<_>>();

        let fn_ty = match def {
            Some(ty::InstanceDef::Virtual(..)) => {
                FnType::new_vtable(&bx, sig, &extra_args)
            }
            Some(ty::InstanceDef::DropGlue(_, None)) => {
                // Empty drop glue; a no-op.
                let &(_, target) = destination.as_ref().unwrap();
                helper.funclet_br(self, &mut bx, target);
                return;
            }
            _ => FnType::new(&bx, sig, &extra_args)
        };

        // Emit a panic or a no-op for `panic_if_uninhabited`.
        if intrinsic == Some("panic_if_uninhabited") {
            let ty = instance.unwrap().substs.type_at(0);
            let layout = bx.layout_of(ty);
            if layout.abi.is_uninhabited() {
                let loc = bx.sess().source_map().lookup_char_pos(span.lo());
                let filename = LocalInternedString::intern(&loc.file.name.to_string());
                let line = bx.const_u32(loc.line as u32);
                let col = bx.const_u32(loc.col.to_usize() as u32 + 1);

                let str = format!(
                    "Attempted to instantiate uninhabited type {}",
                    ty
                );
                let msg_str = LocalInternedString::intern(&str);
                let msg_file_line_col = bx.static_panic_msg(
                    Some(msg_str),
                    filename,
                    line,
                    col,
                    "panic_loc",
                );

                // Obtain the panic entry point.
                let def_id =
                    common::langcall(bx.tcx(), Some(span), "", lang_items::PanicFnLangItem);
                let instance = ty::Instance::mono(bx.tcx(), def_id);
                let fn_ty = FnType::of_instance(&bx, &instance);
                let llfn = bx.get_fn(instance);

                // Codegen the actual panic invoke/call.
                helper.do_call(
                    self,
                    &mut bx,
                    fn_ty,
                    llfn,
                    &[msg_file_line_col],
                    destination.as_ref().map(|(_, bb)| (ReturnDest::Nothing, *bb)),
                    cleanup,
                );
            } else {
                // a NOP
                helper.funclet_br(self, &mut bx, destination.as_ref().unwrap().1)
            }
            return;
        }

        // The arguments we'll be passing. Plus one to account for outptr, if used.
        let arg_count = fn_ty.args.len() + fn_ty.ret.is_indirect() as usize;
        let mut llargs = Vec::with_capacity(arg_count);

        // Prepare the return value destination
        let ret_dest = if let Some((ref dest, _)) = *destination {
            let is_intrinsic = intrinsic.is_some();
            self.make_return_dest(&mut bx, dest, &fn_ty.ret, &mut llargs,
                                  is_intrinsic)
        } else {
            ReturnDest::Nothing
        };

        if intrinsic.is_some() && intrinsic != Some("drop_in_place") {
            let dest = match ret_dest {
                _ if fn_ty.ret.is_indirect() => llargs[0],
                ReturnDest::Nothing =>
                    bx.const_undef(bx.type_ptr_to(bx.memory_ty(&fn_ty.ret))),
                ReturnDest::IndirectOperand(dst, _) | ReturnDest::Store(dst) =>
                    dst.llval,
                ReturnDest::DirectOperand(_) =>
                    bug!("Cannot use direct operand with an intrinsic call"),
            };

            let args: Vec<_> = args.iter().enumerate().map(|(i, arg)| {
                // The indices passed to simd_shuffle* in the
                // third argument must be constant. This is
                // checked by const-qualification, which also
                // promotes any complex rvalues to constants.
                if i == 2 && intrinsic.unwrap().starts_with("simd_shuffle") {
                    match *arg {
                        // The shuffle array argument is usually not an explicit constant,
                        // but specified directly in the code. This means it gets promoted
                        // and we can then extract the value by evaluating the promoted.
                        mir::Operand::Copy(
                            Place::Base(
                                PlaceBase::Static(
                                    box Static { kind: StaticKind::Promoted(promoted), ty }
                                )
                            )
                        ) |
                        mir::Operand::Move(
                            Place::Base(
                                PlaceBase::Static(
                                    box Static { kind: StaticKind::Promoted(promoted), ty }
                                )
                            )
                        ) => {
                            let param_env = ty::ParamEnv::reveal_all();
                            let cid = mir::interpret::GlobalId {
                                instance: self.instance,
                                promoted: Some(promoted),
                            };
                            let c = bx.tcx().const_eval(param_env.and(cid));
                            let (llval, ty) = self.simd_shuffle_indices(
                                &bx,
                                terminator.source_info.span,
                                ty,
                                c,
                            );
                            return OperandRef {
                                val: Immediate(llval),
                                layout: bx.layout_of(ty),
                            };

                        }
                        mir::Operand::Copy(_) |
                        mir::Operand::Move(_) => {
                            span_bug!(span, "shuffle indices must be constant");
                        }
                        mir::Operand::Constant(ref constant) => {
                            let c = self.eval_mir_constant(constant);
                            let (llval, ty) = self.simd_shuffle_indices(
                                &bx,
                                constant.span,
                                constant.ty,
                                c,
                            );
                            return OperandRef {
                                val: Immediate(llval),
                                layout: bx.layout_of(ty)
                            };
                        }
                    }
                }

                self.codegen_operand(&mut bx, arg)
            }).collect();


            let callee_ty = instance.as_ref().unwrap().ty(bx.tcx());
            bx.codegen_intrinsic_call(callee_ty, &fn_ty, &args, dest,
                                      terminator.source_info.span);

            if let ReturnDest::IndirectOperand(dst, _) = ret_dest {
                self.store_return(&mut bx, ret_dest, &fn_ty.ret, dst.llval);
            }

            if let Some((_, target)) = *destination {
                helper.funclet_br(self, &mut bx, target);
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

        // Useful determining if the current argument is the "spoofed" `VaListImpl`
        let last_arg_idx = if sig.inputs().is_empty() {
            None
        } else {
            Some(sig.inputs().len() - 1)
        };
        'make_args: for (i, arg) in first_args.iter().enumerate() {
            // If this is a C-variadic function the function signature contains
            // an "spoofed" `VaListImpl`. This argument is ignored, but we need to
            // populate it with a dummy operand so that the users real arguments
            // are not overwritten.
            let i = if sig.c_variadic && last_arg_idx.map(|x| i >= x).unwrap_or(false) {
                if i + 1 < fn_ty.args.len() {
                    i + 1
                } else {
                    break 'make_args
                }
            } else {
                i
            };
            let mut op = self.codegen_operand(&mut bx, arg);

            if let (0, Some(ty::InstanceDef::Virtual(_, idx))) = (i, def) {
                if let Pair(..) = op.val {
                    // In the case of Rc<Self>, we need to explicitly pass a
                    // *mut RcBox<Self> with a Scalar (not ScalarPair) ABI. This is a hack
                    // that is understood elsewhere in the compiler as a method on
                    // `dyn Trait`.
                    // To get a `*mut RcBox<Self>`, we just keep unwrapping newtypes until
                    // we get a value of a built-in pointer type
                    'descend_newtypes: while !op.layout.ty.is_unsafe_ptr()
                                    && !op.layout.ty.is_region_ptr()
                    {
                        'iter_fields: for i in 0..op.layout.fields.count() {
                            let field = op.extract_field(&mut bx, i);
                            if !field.layout.is_zst() {
                                // we found the one non-zero-sized field that is allowed
                                // now find *its* non-zero-sized field, or stop if it's a
                                // pointer
                                op = field;
                                continue 'descend_newtypes
                            }
                        }

                        span_bug!(span, "receiver has no non-zero-sized fields {:?}", op);
                    }

                    // now that we have `*dyn Trait` or `&dyn Trait`, split it up into its
                    // data pointer and vtable. Look up the method in the vtable, and pass
                    // the data pointer as the first argument
                    match op.val {
                        Pair(data_ptr, meta) => {
                            llfn = Some(meth::VirtualIndex::from_index(idx)
                                .get_fn(&mut bx, meta, &fn_ty));
                            llargs.push(data_ptr);
                            continue 'make_args
                        }
                        other => bug!("expected a Pair, got {:?}", other),
                    }
                } else if let Ref(data_ptr, Some(meta), _) = op.val {
                    // by-value dynamic dispatch
                    llfn = Some(meth::VirtualIndex::from_index(idx)
                        .get_fn(&mut bx, meta, &fn_ty));
                    llargs.push(data_ptr);
                    continue;
                } else {
                    span_bug!(span, "can't codegen a virtual call on {:?}", op);
                }
            }

            // The callee needs to own the argument memory if we pass it
            // by-ref, so make a local copy of non-immediate constants.
            match (arg, op.val) {
                (&mir::Operand::Copy(_), Ref(_, None, _)) |
                (&mir::Operand::Constant(_), Ref(_, None, _)) => {
                    let tmp = PlaceRef::alloca(&mut bx, op.layout, "const");
                    op.val.store(&mut bx, tmp);
                    op.val = Ref(tmp.llval, None, tmp.align);
                }
                _ => {}
            }

            self.codegen_argument(&mut bx, op, &mut llargs, &fn_ty.args[i]);
        }
        if let Some(tup) = untuple {
            self.codegen_arguments_untupled(&mut bx, tup, &mut llargs,
                &fn_ty.args[first_args.len()..])
        }

        let fn_ptr = match (llfn, instance) {
            (Some(llfn), _) => llfn,
            (None, Some(instance)) => bx.get_fn(instance),
            _ => span_bug!(span, "no llfn for call"),
        };

        helper.do_call(self, &mut bx, fn_ty, fn_ptr, &llargs,
                       destination.as_ref().map(|&(_, target)| (ret_dest, target)),
                       cleanup);
    }
}

impl<'a, 'tcx, Bx: BuilderMethods<'a, 'tcx>> FunctionCx<'a, 'tcx, Bx> {
    pub fn codegen_block(
        &mut self,
        bb: mir::BasicBlock,
    ) {
        let mut bx = self.build_block(bb);
        let data = &self.mir[bb];

        debug!("codegen_block({:?}={:?})", bb, data);

        for statement in &data.statements {
            bx = self.codegen_statement(bx, statement);
        }

        self.codegen_terminator(bx, bb, data.terminator());
    }

    fn codegen_terminator(
        &mut self,
        mut bx: Bx,
        bb: mir::BasicBlock,
        terminator: &mir::Terminator<'tcx>
    ) {
        debug!("codegen_terminator: {:?}", terminator);

        // Create the cleanup bundle, if needed.
        let funclet_bb = self.cleanup_kinds[bb].funclet_bb(bb);
        let helper = TerminatorCodegenHelper {
            bb: &bb, terminator, funclet_bb
        };

        self.set_debug_loc(&mut bx, terminator.source_info);
        match terminator.kind {
            mir::TerminatorKind::Resume => {
                self.codegen_resume_terminator(helper, bx)
            }

            mir::TerminatorKind::Abort => {
                bx.abort();
                bx.unreachable();
            }

            mir::TerminatorKind::Goto { target } => {
                helper.funclet_br(self, &mut bx, target);
            }

            mir::TerminatorKind::SwitchInt {
                ref discr, switch_ty, ref values, ref targets
            } => {
                self.codegen_switchint_terminator(helper, bx, discr, switch_ty,
                                                  values, targets);
            }

            mir::TerminatorKind::Return => {
                self.codegen_return_terminator(bx);
            }

            mir::TerminatorKind::Unreachable => {
                bx.unreachable();
            }

            mir::TerminatorKind::Drop { ref location, target, unwind } => {
                self.codegen_drop_terminator(helper, bx, location, target, unwind);
            }

            mir::TerminatorKind::Assert { ref cond, expected, ref msg, target, cleanup } => {
                self.codegen_assert_terminator(helper, bx, terminator, cond,
                                               expected, msg, target, cleanup);
            }

            mir::TerminatorKind::DropAndReplace { .. } => {
                bug!("undesugared DropAndReplace in codegen: {:?}", terminator);
            }

            mir::TerminatorKind::Call {
                ref func,
                ref args,
                ref destination,
                cleanup,
                from_hir_call: _
            } => {
                self.codegen_call_terminator(helper, bx, terminator, func,
                                             args, destination, cleanup);
            }
            mir::TerminatorKind::GeneratorDrop |
            mir::TerminatorKind::Yield { .. } => bug!("generator ops in codegen"),
            mir::TerminatorKind::FalseEdges { .. } |
            mir::TerminatorKind::FalseUnwind { .. } => bug!("borrowck false edges in codegen"),
        }
    }

    fn codegen_argument(
        &mut self,
        bx: &mut Bx,
        op: OperandRef<'tcx, Bx::Value>,
        llargs: &mut Vec<Bx::Value>,
        arg: &ArgType<'tcx, Ty<'tcx>>
    ) {
        // Fill padding with undef value, where applicable.
        if let Some(ty) = arg.pad {
            llargs.push(bx.const_undef(bx.reg_backend_type(&ty)))
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
                _ => bug!("codegen_argument: {:?} invalid for pair argument", op)
            }
        } else if arg.is_unsized_indirect() {
            match op.val {
                Ref(a, Some(b), _) => {
                    llargs.push(a);
                    llargs.push(b);
                    return;
                }
                _ => bug!("codegen_argument: {:?} invalid for unsized indirect argument", op)
            }
        }

        // Force by-ref if we have to load through a cast pointer.
        let (mut llval, align, by_ref) = match op.val {
            Immediate(_) | Pair(..) => {
                match arg.mode {
                    PassMode::Indirect(..) | PassMode::Cast(_) => {
                        let scratch = PlaceRef::alloca(bx, arg.layout, "arg");
                        op.val.store(bx, scratch);
                        (scratch.llval, scratch.align, true)
                    }
                    _ => {
                        (op.immediate_or_packed_pair(bx), arg.layout.align.abi, false)
                    }
                }
            }
            Ref(llval, _, align) => {
                if arg.is_indirect() && align < arg.layout.align.abi {
                    // `foo(packed.large_field)`. We can't pass the (unaligned) field directly. I
                    // think that ATM (Rust 1.16) we only pass temporaries, but we shouldn't
                    // have scary latent bugs around.

                    let scratch = PlaceRef::alloca(bx, arg.layout, "arg");
                    base::memcpy_ty(bx, scratch.llval, scratch.align, llval, align,
                                    op.layout, MemFlags::empty());
                    (scratch.llval, scratch.align, true)
                } else {
                    (llval, align, true)
                }
            }
        };

        if by_ref && !arg.is_indirect() {
            // Have to load the argument, maybe while casting it.
            if let PassMode::Cast(ty) = arg.mode {
                let addr = bx.pointercast(llval, bx.type_ptr_to(
                    bx.cast_backend_type(&ty))
                );
                llval = bx.load(addr, align.min(arg.layout.align.abi));
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
                // We store bools as `i8` so we need to truncate to `i1`.
                llval = base::to_immediate(bx, llval, arg.layout);
            }
        }

        llargs.push(llval);
    }

    fn codegen_arguments_untupled(
        &mut self,
        bx: &mut Bx,
        operand: &mir::Operand<'tcx>,
        llargs: &mut Vec<Bx::Value>,
        args: &[ArgType<'tcx, Ty<'tcx>>]
    ) {
        let tuple = self.codegen_operand(bx, operand);

        // Handle both by-ref and immediate tuples.
        if let Ref(llval, None, align) = tuple.val {
            let tuple_ptr = PlaceRef::new_sized(llval, tuple.layout, align);
            for i in 0..tuple.layout.fields.count() {
                let field_ptr = tuple_ptr.project_field(bx, i);
                let field = bx.load_operand(field_ptr);
                self.codegen_argument(bx, field, llargs, &args[i]);
            }
        } else if let Ref(_, Some(_), _) = tuple.val {
            bug!("closure arguments must be sized")
        } else {
            // If the tuple is immediate, the elements are as well.
            for i in 0..tuple.layout.fields.count() {
                let op = tuple.extract_field(bx, i);
                self.codegen_argument(bx, op, llargs, &args[i]);
            }
        }
    }

    fn get_personality_slot(
        &mut self,
        bx: &mut Bx
    ) -> PlaceRef<'tcx, Bx::Value> {
        let cx = bx.cx();
        if let Some(slot) = self.personality_slot {
            slot
        } else {
            let layout = cx.layout_of(cx.tcx().intern_tup(&[
                cx.tcx().mk_mut_ptr(cx.tcx().types.u8),
                cx.tcx().types.i32
            ]));
            let slot = PlaceRef::alloca(bx, layout, "personalityslot");
            self.personality_slot = Some(slot);
            slot
        }
    }

    /// Returns the landing-pad wrapper around the given basic block.
    ///
    /// No-op in MSVC SEH scheme.
    fn landing_pad_to(
        &mut self,
        target_bb: mir::BasicBlock
    ) -> Bx::BasicBlock {
        if let Some(block) = self.landing_pads[target_bb] {
            return block;
        }

        let block = self.blocks[target_bb];
        let landing_pad = self.landing_pad_uncached(block);
        self.landing_pads[target_bb] = Some(landing_pad);
        landing_pad
    }

    fn landing_pad_uncached(
        &mut self,
        target_bb: Bx::BasicBlock
    ) -> Bx::BasicBlock {
        if base::wants_msvc_seh(self.cx.sess()) {
            span_bug!(self.mir.span, "landing pad was not inserted?")
        }

        let mut bx = self.new_block("cleanup");

        let llpersonality = self.cx.eh_personality();
        let llretty = self.landing_pad_type();
        let lp = bx.landing_pad(llretty, llpersonality, 1);
        bx.set_cleanup(lp);

        let slot = self.get_personality_slot(&mut bx);
        slot.storage_live(&mut bx);
        Pair(bx.extract_value(lp, 0), bx.extract_value(lp, 1)).store(&mut bx, slot);

        bx.br(target_bb);
        bx.llbb()
    }

    fn landing_pad_type(&self) -> Bx::Type {
        let cx = self.cx;
        cx.type_struct(&[cx.type_i8p(), cx.type_i32()], false)
    }

    fn unreachable_block(
        &mut self
    ) -> Bx::BasicBlock {
        self.unreachable_block.unwrap_or_else(|| {
            let mut bx = self.new_block("unreachable");
            bx.unreachable();
            self.unreachable_block = Some(bx.llbb());
            bx.llbb()
        })
    }

    pub fn new_block(&self, name: &str) -> Bx {
        Bx::new_block(self.cx, self.llfn, name)
    }

    pub fn build_block(
        &self,
        bb: mir::BasicBlock
    ) -> Bx {
        let mut bx = Bx::with_cx(self.cx);
        bx.position_at_end(self.blocks[bb]);
        bx
    }

    fn make_return_dest(
        &mut self,
        bx: &mut Bx,
        dest: &mir::Place<'tcx>,
        fn_ret: &ArgType<'tcx, Ty<'tcx>>,
        llargs: &mut Vec<Bx::Value>, is_intrinsic: bool
    ) -> ReturnDest<'tcx, Bx::Value> {
        // If the return is ignored, we can just return a do-nothing `ReturnDest`.
        if fn_ret.is_ignore() {
            return ReturnDest::Nothing;
        }
        let dest = if let mir::Place::Base(mir::PlaceBase::Local(index)) = *dest {
            match self.locals[index] {
                LocalRef::Place(dest) => dest,
                LocalRef::UnsizedPlace(_) => bug!("return type must be sized"),
                LocalRef::Operand(None) => {
                    // Handle temporary places, specifically `Operand` ones, as
                    // they don't have `alloca`s.
                    return if fn_ret.is_indirect() {
                        // Odd, but possible, case, we have an operand temporary,
                        // but the calling convention has an indirect return.
                        let tmp = PlaceRef::alloca(bx, fn_ret.layout, "tmp_ret");
                        tmp.storage_live(bx);
                        llargs.push(tmp.llval);
                        ReturnDest::IndirectOperand(tmp, index)
                    } else if is_intrinsic {
                        // Currently, intrinsics always need a location to store
                        // the result, so we create a temporary `alloca` for the
                        // result.
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
            if dest.align < dest.layout.align.abi {
                // Currently, MIR code generation does not create calls
                // that store directly to fields of packed structs (in
                // fact, the calls it creates write only to temps).
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

    fn codegen_transmute(
        &mut self,
        bx: &mut Bx,
        src: &mir::Operand<'tcx>,
        dst: &mir::Place<'tcx>
    ) {
        if let mir::Place::Base(mir::PlaceBase::Local(index)) = *dst {
            match self.locals[index] {
                LocalRef::Place(place) => self.codegen_transmute_into(bx, src, place),
                LocalRef::UnsizedPlace(_) => bug!("transmute must not involve unsized locals"),
                LocalRef::Operand(None) => {
                    let dst_layout = bx.layout_of(self.monomorphized_place_ty(dst));
                    assert!(!dst_layout.ty.has_erasable_regions());
                    let place = PlaceRef::alloca(bx, dst_layout, "transmute_temp");
                    place.storage_live(bx);
                    self.codegen_transmute_into(bx, src, place);
                    let op = bx.load_operand(place);
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

    fn codegen_transmute_into(
        &mut self,
        bx: &mut Bx,
        src: &mir::Operand<'tcx>,
        dst: PlaceRef<'tcx, Bx::Value>
    ) {
        let src = self.codegen_operand(bx, src);
        let llty = bx.backend_type(src.layout);
        let cast_ptr = bx.pointercast(dst.llval, bx.type_ptr_to(llty));
        let align = src.layout.align.abi.min(dst.align);
        src.val.store(bx, PlaceRef::new_sized(cast_ptr, src.layout, align));
    }


    // Stores the return value of a function call into it's final location.
    fn store_return(
        &mut self,
        bx: &mut Bx,
        dest: ReturnDest<'tcx, Bx::Value>,
        ret_ty: &ArgType<'tcx, Ty<'tcx>>,
        llval: Bx::Value
    ) {
        use self::ReturnDest::*;

        match dest {
            Nothing => (),
            Store(dst) => bx.store_arg_ty(&ret_ty, llval, dst),
            IndirectOperand(tmp, index) => {
                let op = bx.load_operand(tmp);
                tmp.storage_dead(bx);
                self.locals[index] = LocalRef::Operand(Some(op));
            }
            DirectOperand(index) => {
                // If there is a cast, we have to store and reload.
                let op = if let PassMode::Cast(_) = ret_ty.mode {
                    let tmp = PlaceRef::alloca(bx, ret_ty.layout, "tmp_ret");
                    tmp.storage_live(bx);
                    bx.store_arg_ty(&ret_ty, llval, tmp);
                    let op = bx.load_operand(tmp);
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

enum ReturnDest<'tcx, V> {
    // Do nothing; the return value is indirect or ignored.
    Nothing,
    // Store the return value to the pointer.
    Store(PlaceRef<'tcx, V>),
    // Store an indirect return value to an operand local place.
    IndirectOperand(PlaceRef<'tcx, V>, mir::Local),
    // Store a direct return value to an operand local place.
    DirectOperand(mir::Local)
}
