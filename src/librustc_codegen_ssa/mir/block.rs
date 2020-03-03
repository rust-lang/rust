use super::operand::OperandRef;
use super::operand::OperandValue::{Immediate, Pair, Ref};
use super::place::PlaceRef;
use super::{FunctionCx, LocalRef};

use crate::base;
use crate::common::{self, IntPredicate};
use crate::meth;
use crate::traits::*;
use crate::MemFlags;

use rustc::middle::lang_items;
use rustc::mir;
use rustc::mir::AssertKind;
use rustc::ty::layout::{self, FnAbiExt, HasTyCtxt, LayoutOf};
use rustc::ty::{self, Instance, Ty, TypeFoldable};
use rustc_index::vec::Idx;
use rustc_span::{source_map::Span, symbol::Symbol};
use rustc_target::abi::call::{ArgAbi, FnAbi, PassMode};
use rustc_target::spec::abi::Abi;

use std::borrow::Cow;

/// Used by `FunctionCx::codegen_terminator` for emitting common patterns
/// e.g., creating a basic block, calling a function, etc.
struct TerminatorCodegenHelper<'tcx> {
    bb: mir::BasicBlock,
    terminator: &'tcx mir::Terminator<'tcx>,
    funclet_bb: Option<mir::BasicBlock>,
}

impl<'a, 'tcx> TerminatorCodegenHelper<'tcx> {
    /// Returns the associated funclet from `FunctionCx::funclets` for the
    /// `funclet_bb` member if it is not `None`.
    fn funclet<'b, Bx: BuilderMethods<'a, 'tcx>>(
        &self,
        fx: &'b mut FunctionCx<'a, 'tcx, Bx>,
    ) -> Option<&'b Bx::Funclet> {
        match self.funclet_bb {
            Some(funcl) => fx.funclets[funcl].as_ref(),
            None => None,
        }
    }

    fn lltarget<Bx: BuilderMethods<'a, 'tcx>>(
        &self,
        fx: &mut FunctionCx<'a, 'tcx, Bx>,
        target: mir::BasicBlock,
    ) -> (Bx::BasicBlock, bool) {
        let span = self.terminator.source_info.span;
        let lltarget = fx.blocks[target];
        let target_funclet = fx.cleanup_kinds[target].funclet_bb(target);
        match (self.funclet_bb, target_funclet) {
            (None, None) => (lltarget, false),
            (Some(f), Some(t_f)) if f == t_f || !base::wants_msvc_seh(fx.cx.tcx().sess) => {
                (lltarget, false)
            }
            // jump *into* cleanup - need a landing pad if GNU
            (None, Some(_)) => (fx.landing_pad_to(target), false),
            (Some(_), None) => span_bug!(span, "{:?} - jump out of cleanup?", self.terminator),
            (Some(_), Some(_)) => (fx.landing_pad_to(target), true),
        }
    }

    /// Create a basic block.
    fn llblock<Bx: BuilderMethods<'a, 'tcx>>(
        &self,
        fx: &mut FunctionCx<'a, 'tcx, Bx>,
        target: mir::BasicBlock,
    ) -> Bx::BasicBlock {
        let (lltarget, is_cleanupret) = self.lltarget(fx, target);
        if is_cleanupret {
            // MSVC cross-funclet jump - need a trampoline

            debug!("llblock: creating cleanup trampoline for {:?}", target);
            let name = &format!("{:?}_cleanup_trampoline_{:?}", self.bb, target);
            let mut trampoline = fx.new_block(name);
            trampoline.cleanup_ret(self.funclet(fx).unwrap(), Some(lltarget));
            trampoline.llbb()
        } else {
            lltarget
        }
    }

    fn funclet_br<Bx: BuilderMethods<'a, 'tcx>>(
        &self,
        fx: &mut FunctionCx<'a, 'tcx, Bx>,
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

    /// Call `fn_ptr` of `fn_abi` with the arguments `llargs`, the optional
    /// return destination `destination` and the cleanup function `cleanup`.
    fn do_call<Bx: BuilderMethods<'a, 'tcx>>(
        &self,
        fx: &mut FunctionCx<'a, 'tcx, Bx>,
        bx: &mut Bx,
        fn_abi: FnAbi<'tcx, Ty<'tcx>>,
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
            let invokeret =
                bx.invoke(fn_ptr, &llargs, ret_bx, self.llblock(fx, cleanup), self.funclet(fx));
            bx.apply_attrs_callsite(&fn_abi, invokeret);

            if let Some((ret_dest, target)) = destination {
                let mut ret_bx = fx.build_block(target);
                fx.set_debug_loc(&mut ret_bx, self.terminator.source_info);
                fx.store_return(&mut ret_bx, ret_dest, &fn_abi.ret, invokeret);
            }
        } else {
            let llret = bx.call(fn_ptr, &llargs, self.funclet(fx));
            bx.apply_attrs_callsite(&fn_abi, llret);
            if fx.mir[self.bb].is_cleanup {
                // Cleanup is always the cold path. Don't inline
                // drop glue. Also, when there is a deeply-nested
                // struct, there are "symmetry" issues that cause
                // exponential inlining - see issue #41696.
                bx.do_not_inline(llret);
            }

            if let Some((ret_dest, target)) = destination {
                fx.store_return(bx, ret_dest, &fn_abi.ret, llret);
                self.funclet_br(fx, bx, target);
            } else {
                bx.unreachable();
            }
        }
    }

    // Generate sideeffect intrinsic if jumping to any of the targets can form
    // a loop.
    fn maybe_sideeffect<Bx: BuilderMethods<'a, 'tcx>>(
        &self,
        mir: mir::ReadOnlyBodyAndCache<'tcx, 'tcx>,
        bx: &mut Bx,
        targets: &[mir::BasicBlock],
    ) {
        if bx.tcx().sess.opts.debugging_opts.insert_sideeffect {
            if targets.iter().any(|&target| {
                target <= self.bb
                    && target.start_location().is_predecessor_of(self.bb.start_location(), mir)
            }) {
                bx.sideeffect();
            }
        }
    }
}

/// Codegen implementations for some terminator variants.
impl<'a, 'tcx, Bx: BuilderMethods<'a, 'tcx>> FunctionCx<'a, 'tcx, Bx> {
    /// Generates code for a `Resume` terminator.
    fn codegen_resume_terminator(&mut self, helper: TerminatorCodegenHelper<'tcx>, mut bx: Bx) {
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
                bx.call(bx.eh_unwind_resume(), &[lp0], helper.funclet(self));
                bx.unreachable();
            }
        }
    }

    fn codegen_switchint_terminator(
        &mut self,
        helper: TerminatorCodegenHelper<'tcx>,
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
                helper.maybe_sideeffect(self.mir, &mut bx, targets.as_slice());
                // Don't generate trivial icmps when switching on bool
                if let [0] = values[..] {
                    bx.cond_br(discr.immediate(), llfalse, lltrue);
                } else {
                    assert_eq!(&values[..], &[1]);
                    bx.cond_br(discr.immediate(), lltrue, llfalse);
                }
            } else {
                let switch_llty = bx.immediate_backend_type(bx.layout_of(switch_ty));
                let llval = bx.const_uint_big(switch_llty, values[0]);
                let cmp = bx.icmp(IntPredicate::IntEQ, discr.immediate(), llval);
                helper.maybe_sideeffect(self.mir, &mut bx, targets.as_slice());
                bx.cond_br(cmp, lltrue, llfalse);
            }
        } else {
            helper.maybe_sideeffect(self.mir, &mut bx, targets.as_slice());
            let (otherwise, targets) = targets.split_last().unwrap();
            bx.switch(
                discr.immediate(),
                helper.llblock(self, *otherwise),
                values
                    .iter()
                    .zip(targets)
                    .map(|(&value, target)| (value, helper.llblock(self, *target))),
            );
        }
    }

    fn codegen_return_terminator(&mut self, mut bx: Bx) {
        // Call `va_end` if this is the definition of a C-variadic function.
        if self.fn_abi.c_variadic {
            // The `VaList` "spoofed" argument is just after all the real arguments.
            let va_list_arg_idx = self.fn_abi.args.len();
            match self.locals[mir::Local::new(1 + va_list_arg_idx)] {
                LocalRef::Place(va_list) => {
                    bx.va_end(va_list.llval);
                }
                _ => bug!("C-variadic function must have a `VaList` place"),
            }
        }
        if self.fn_abi.ret.layout.abi.is_uninhabited() {
            // Functions with uninhabited return values are marked `noreturn`,
            // so we should make sure that we never actually do.
            // We play it safe by using a well-defined `abort`, but we could go for immediate UB
            // if that turns out to be helpful.
            bx.abort();
            // `abort` does not terminate the block, so we still need to generate
            // an `unreachable` terminator after it.
            bx.unreachable();
            return;
        }
        let llval = match self.fn_abi.ret.mode {
            PassMode::Ignore | PassMode::Indirect(..) => {
                bx.ret_void();
                return;
            }

            PassMode::Direct(_) | PassMode::Pair(..) => {
                let op = self.codegen_consume(&mut bx, mir::Place::return_place().as_ref());
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
                    LocalRef::Place(cg_place) => OperandRef {
                        val: Ref(cg_place.llval, None, cg_place.align),
                        layout: cg_place.layout,
                    },
                    LocalRef::UnsizedPlace(_) => bug!("return type must be sized"),
                };
                let llslot = match op.val {
                    Immediate(_) | Pair(..) => {
                        let scratch = PlaceRef::alloca(&mut bx, self.fn_abi.ret.layout);
                        op.val.store(&mut bx, scratch);
                        scratch.llval
                    }
                    Ref(llval, _, align) => {
                        assert_eq!(align, op.layout.align.abi, "return place is unaligned!");
                        llval
                    }
                };
                let addr = bx.pointercast(llslot, bx.type_ptr_to(bx.cast_backend_type(&cast_ty)));
                bx.load(addr, self.fn_abi.ret.layout.align.abi)
            }
        };
        bx.ret(llval);
    }

    fn codegen_drop_terminator(
        &mut self,
        helper: TerminatorCodegenHelper<'tcx>,
        mut bx: Bx,
        location: &mir::Place<'tcx>,
        target: mir::BasicBlock,
        unwind: Option<mir::BasicBlock>,
    ) {
        let ty = location.ty(*self.mir, bx.tcx()).ty;
        let ty = self.monomorphize(&ty);
        let drop_fn = Instance::resolve_drop_in_place(bx.tcx(), ty);

        if let ty::InstanceDef::DropGlue(_, None) = drop_fn.def {
            // we don't actually need to drop anything.
            helper.maybe_sideeffect(self.mir, &mut bx, &[target]);
            helper.funclet_br(self, &mut bx, target);
            return;
        }

        let place = self.codegen_place(&mut bx, location.as_ref());
        let (args1, args2);
        let mut args = if let Some(llextra) = place.llextra {
            args2 = [place.llval, llextra];
            &args2[..]
        } else {
            args1 = [place.llval];
            &args1[..]
        };
        let (drop_fn, fn_abi) = match ty.kind {
            // FIXME(eddyb) perhaps move some of this logic into
            // `Instance::resolve_drop_in_place`?
            ty::Dynamic(..) => {
                let virtual_drop = Instance {
                    def: ty::InstanceDef::Virtual(drop_fn.def_id(), 0),
                    substs: drop_fn.substs,
                };
                let fn_abi = FnAbi::of_instance(&bx, virtual_drop, &[]);
                let vtable = args[1];
                args = &args[..1];
                (meth::DESTRUCTOR.get_fn(&mut bx, vtable, &fn_abi), fn_abi)
            }
            _ => (bx.get_fn_addr(drop_fn), FnAbi::of_instance(&bx, drop_fn, &[])),
        };
        helper.maybe_sideeffect(self.mir, &mut bx, &[target]);
        helper.do_call(
            self,
            &mut bx,
            fn_abi,
            drop_fn,
            args,
            Some((ReturnDest::Nothing, target)),
            unwind,
        );
    }

    fn codegen_assert_terminator(
        &mut self,
        helper: TerminatorCodegenHelper<'tcx>,
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
            if let AssertKind::OverflowNeg = *msg {
                const_cond = Some(expected);
            }
        }

        // Don't codegen the panic block if success if known.
        if const_cond == Some(expected) {
            helper.maybe_sideeffect(self.mir, &mut bx, &[target]);
            helper.funclet_br(self, &mut bx, target);
            return;
        }

        // Pass the condition through llvm.expect for branch hinting.
        let cond = bx.expect(cond, expected);

        // Create the failure block and the conditional branch to it.
        let lltarget = helper.llblock(self, target);
        let panic_block = self.new_block("panic");
        helper.maybe_sideeffect(self.mir, &mut bx, &[target]);
        if expected {
            bx.cond_br(cond, lltarget, panic_block.llbb());
        } else {
            bx.cond_br(cond, panic_block.llbb(), lltarget);
        }

        // After this point, bx is the block for the call to panic.
        bx = panic_block;
        self.set_debug_loc(&mut bx, terminator.source_info);

        // Get the location information.
        let location = self.get_caller_location(&mut bx, span).immediate();

        // Put together the arguments to the panic entry point.
        let (lang_item, args) = match msg {
            AssertKind::BoundsCheck { ref len, ref index } => {
                let len = self.codegen_operand(&mut bx, len).immediate();
                let index = self.codegen_operand(&mut bx, index).immediate();
                (lang_items::PanicBoundsCheckFnLangItem, vec![location, index, len])
            }
            _ => {
                let msg_str = Symbol::intern(msg.description());
                let msg = bx.const_str(msg_str);
                (lang_items::PanicFnLangItem, vec![msg.0, msg.1, location])
            }
        };

        // Obtain the panic entry point.
        let def_id = common::langcall(bx.tcx(), Some(span), "", lang_item);
        let instance = ty::Instance::mono(bx.tcx(), def_id);
        let fn_abi = FnAbi::of_instance(&bx, instance, &[]);
        let llfn = bx.get_fn_addr(instance);

        // Codegen the actual panic invoke/call.
        helper.do_call(self, &mut bx, fn_abi, llfn, &args, None, cleanup);
    }

    fn codegen_call_terminator(
        &mut self,
        helper: TerminatorCodegenHelper<'tcx>,
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

        let (instance, mut llfn) = match callee.layout.ty.kind {
            ty::FnDef(def_id, substs) => (
                Some(
                    ty::Instance::resolve(bx.tcx(), ty::ParamEnv::reveal_all(), def_id, substs)
                        .unwrap(),
                ),
                None,
            ),
            ty::FnPtr(_) => (None, Some(callee.immediate())),
            _ => bug!("{} is not callable", callee.layout.ty),
        };
        let def = instance.map(|i| i.def);

        if let Some(ty::InstanceDef::DropGlue(_, None)) = def {
            // Empty drop glue; a no-op.
            let &(_, target) = destination.as_ref().unwrap();
            helper.maybe_sideeffect(self.mir, &mut bx, &[target]);
            helper.funclet_br(self, &mut bx, target);
            return;
        }

        // FIXME(eddyb) avoid computing this if possible, when `instance` is
        // available - right now `sig` is only needed for getting the `abi`
        // and figuring out how many extra args were passed to a C-variadic `fn`.
        let sig = callee.layout.ty.fn_sig(bx.tcx());
        let abi = sig.abi();

        // Handle intrinsics old codegen wants Expr's for, ourselves.
        let intrinsic = match def {
            Some(ty::InstanceDef::Intrinsic(def_id)) => Some(bx.tcx().item_name(def_id).as_str()),
            _ => None,
        };
        let intrinsic = intrinsic.as_ref().map(|s| &s[..]);

        let extra_args = &args[sig.inputs().skip_binder().len()..];
        let extra_args = extra_args
            .iter()
            .map(|op_arg| {
                let op_ty = op_arg.ty(*self.mir, bx.tcx());
                self.monomorphize(&op_ty)
            })
            .collect::<Vec<_>>();

        let fn_abi = match instance {
            Some(instance) => FnAbi::of_instance(&bx, instance, &extra_args),
            None => FnAbi::of_fn_ptr(&bx, sig, &extra_args),
        };

        if intrinsic == Some("transmute") {
            if let Some(destination_ref) = destination.as_ref() {
                let &(ref dest, target) = destination_ref;
                self.codegen_transmute(&mut bx, &args[0], dest);
                helper.maybe_sideeffect(self.mir, &mut bx, &[target]);
                helper.funclet_br(self, &mut bx, target);
            } else {
                // If we are trying to transmute to an uninhabited type,
                // it is likely there is no allotted destination. In fact,
                // transmuting to an uninhabited type is UB, which means
                // we can do what we like. Here, we declare that transmuting
                // into an uninhabited type is impossible, so anything following
                // it must be unreachable.
                assert_eq!(fn_abi.ret.layout.abi, layout::Abi::Uninhabited);
                bx.unreachable();
            }
            return;
        }

        // For normal codegen, this Miri-specific intrinsic should never occur.
        if intrinsic == Some("miri_start_panic") {
            bug!("`miri_start_panic` should never end up in compiled code");
        }

        // Emit a panic or a no-op for `panic_if_uninhabited`.
        if intrinsic == Some("panic_if_uninhabited") {
            let ty = instance.unwrap().substs.type_at(0);
            let layout = bx.layout_of(ty);
            if layout.abi.is_uninhabited() {
                let msg_str = format!("Attempted to instantiate uninhabited type {}", ty);
                let msg = bx.const_str(Symbol::intern(&msg_str));
                let location = self.get_caller_location(&mut bx, span).immediate();

                // Obtain the panic entry point.
                let def_id =
                    common::langcall(bx.tcx(), Some(span), "", lang_items::PanicFnLangItem);
                let instance = ty::Instance::mono(bx.tcx(), def_id);
                let fn_abi = FnAbi::of_instance(&bx, instance, &[]);
                let llfn = bx.get_fn_addr(instance);

                if let Some((_, target)) = destination.as_ref() {
                    helper.maybe_sideeffect(self.mir, &mut bx, &[*target]);
                }
                // Codegen the actual panic invoke/call.
                helper.do_call(
                    self,
                    &mut bx,
                    fn_abi,
                    llfn,
                    &[msg.0, msg.1, location],
                    destination.as_ref().map(|(_, bb)| (ReturnDest::Nothing, *bb)),
                    cleanup,
                );
            } else {
                // a NOP
                let target = destination.as_ref().unwrap().1;
                helper.maybe_sideeffect(self.mir, &mut bx, &[target]);
                helper.funclet_br(self, &mut bx, target)
            }
            return;
        }

        // The arguments we'll be passing. Plus one to account for outptr, if used.
        let arg_count = fn_abi.args.len() + fn_abi.ret.is_indirect() as usize;
        let mut llargs = Vec::with_capacity(arg_count);

        // Prepare the return value destination
        let ret_dest = if let Some((ref dest, _)) = *destination {
            let is_intrinsic = intrinsic.is_some();
            self.make_return_dest(&mut bx, dest, &fn_abi.ret, &mut llargs, is_intrinsic)
        } else {
            ReturnDest::Nothing
        };

        if intrinsic == Some("caller_location") {
            if let Some((_, target)) = destination.as_ref() {
                let location = self.get_caller_location(&mut bx, span);

                if let ReturnDest::IndirectOperand(tmp, _) = ret_dest {
                    location.val.store(&mut bx, tmp);
                }
                self.store_return(&mut bx, ret_dest, &fn_abi.ret, location.immediate());

                helper.maybe_sideeffect(self.mir, &mut bx, &[*target]);
                helper.funclet_br(self, &mut bx, *target);
            }
            return;
        }

        if intrinsic.is_some() && intrinsic != Some("drop_in_place") {
            let dest = match ret_dest {
                _ if fn_abi.ret.is_indirect() => llargs[0],
                ReturnDest::Nothing => {
                    bx.const_undef(bx.type_ptr_to(bx.arg_memory_ty(&fn_abi.ret)))
                }
                ReturnDest::IndirectOperand(dst, _) | ReturnDest::Store(dst) => dst.llval,
                ReturnDest::DirectOperand(_) => {
                    bug!("Cannot use direct operand with an intrinsic call")
                }
            };

            let args: Vec<_> = args
                .iter()
                .enumerate()
                .map(|(i, arg)| {
                    // The indices passed to simd_shuffle* in the
                    // third argument must be constant. This is
                    // checked by const-qualification, which also
                    // promotes any complex rvalues to constants.
                    if i == 2 && intrinsic.unwrap().starts_with("simd_shuffle") {
                        if let mir::Operand::Constant(constant) = arg {
                            let c = self.eval_mir_constant(constant);
                            let (llval, ty) = self.simd_shuffle_indices(
                                &bx,
                                constant.span,
                                constant.literal.ty,
                                c,
                            );
                            return OperandRef { val: Immediate(llval), layout: bx.layout_of(ty) };
                        } else {
                            span_bug!(span, "shuffle indices must be constant");
                        }
                    }

                    self.codegen_operand(&mut bx, arg)
                })
                .collect();

            bx.codegen_intrinsic_call(
                *instance.as_ref().unwrap(),
                &fn_abi,
                &args,
                dest,
                terminator.source_info.span,
            );

            if let ReturnDest::IndirectOperand(dst, _) = ret_dest {
                self.store_return(&mut bx, ret_dest, &fn_abi.ret, dst.llval);
            }

            if let Some((_, target)) = *destination {
                helper.maybe_sideeffect(self.mir, &mut bx, &[target]);
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

        'make_args: for (i, arg) in first_args.iter().enumerate() {
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
                        for i in 0..op.layout.fields.count() {
                            let field = op.extract_field(&mut bx, i);
                            if !field.layout.is_zst() {
                                // we found the one non-zero-sized field that is allowed
                                // now find *its* non-zero-sized field, or stop if it's a
                                // pointer
                                op = field;
                                continue 'descend_newtypes;
                            }
                        }

                        span_bug!(span, "receiver has no non-zero-sized fields {:?}", op);
                    }

                    // now that we have `*dyn Trait` or `&dyn Trait`, split it up into its
                    // data pointer and vtable. Look up the method in the vtable, and pass
                    // the data pointer as the first argument
                    match op.val {
                        Pair(data_ptr, meta) => {
                            llfn = Some(
                                meth::VirtualIndex::from_index(idx).get_fn(&mut bx, meta, &fn_abi),
                            );
                            llargs.push(data_ptr);
                            continue 'make_args;
                        }
                        other => bug!("expected a Pair, got {:?}", other),
                    }
                } else if let Ref(data_ptr, Some(meta), _) = op.val {
                    // by-value dynamic dispatch
                    llfn = Some(meth::VirtualIndex::from_index(idx).get_fn(&mut bx, meta, &fn_abi));
                    llargs.push(data_ptr);
                    continue;
                } else {
                    span_bug!(span, "can't codegen a virtual call on {:?}", op);
                }
            }

            // The callee needs to own the argument memory if we pass it
            // by-ref, so make a local copy of non-immediate constants.
            match (arg, op.val) {
                (&mir::Operand::Copy(_), Ref(_, None, _))
                | (&mir::Operand::Constant(_), Ref(_, None, _)) => {
                    let tmp = PlaceRef::alloca(&mut bx, op.layout);
                    op.val.store(&mut bx, tmp);
                    op.val = Ref(tmp.llval, None, tmp.align);
                }
                _ => {}
            }

            self.codegen_argument(&mut bx, op, &mut llargs, &fn_abi.args[i]);
        }
        if let Some(tup) = untuple {
            self.codegen_arguments_untupled(
                &mut bx,
                tup,
                &mut llargs,
                &fn_abi.args[first_args.len()..],
            )
        }

        let needs_location =
            instance.map_or(false, |i| i.def.requires_caller_location(self.cx.tcx()));
        if needs_location {
            assert_eq!(
                fn_abi.args.len(),
                args.len() + 1,
                "#[track_caller] fn's must have 1 more argument in their ABI than in their MIR",
            );
            let location = self.get_caller_location(&mut bx, span);
            let last_arg = fn_abi.args.last().unwrap();
            self.codegen_argument(&mut bx, location, &mut llargs, last_arg);
        }

        let fn_ptr = match (llfn, instance) {
            (Some(llfn), _) => llfn,
            (None, Some(instance)) => bx.get_fn_addr(instance),
            _ => span_bug!(span, "no llfn for call"),
        };

        if let Some((_, target)) = destination.as_ref() {
            helper.maybe_sideeffect(self.mir, &mut bx, &[*target]);
        }
        helper.do_call(
            self,
            &mut bx,
            fn_abi,
            fn_ptr,
            &llargs,
            destination.as_ref().map(|&(_, target)| (ret_dest, target)),
            cleanup,
        );
    }
}

impl<'a, 'tcx, Bx: BuilderMethods<'a, 'tcx>> FunctionCx<'a, 'tcx, Bx> {
    pub fn codegen_block(&mut self, bb: mir::BasicBlock) {
        let mut bx = self.build_block(bb);
        let mir = self.mir;
        let data = &mir[bb];

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
        terminator: &'tcx mir::Terminator<'tcx>,
    ) {
        debug!("codegen_terminator: {:?}", terminator);

        // Create the cleanup bundle, if needed.
        let funclet_bb = self.cleanup_kinds[bb].funclet_bb(bb);
        let helper = TerminatorCodegenHelper { bb, terminator, funclet_bb };

        self.set_debug_loc(&mut bx, terminator.source_info);
        match terminator.kind {
            mir::TerminatorKind::Resume => self.codegen_resume_terminator(helper, bx),

            mir::TerminatorKind::Abort => {
                bx.abort();
                // `abort` does not terminate the block, so we still need to generate
                // an `unreachable` terminator after it.
                bx.unreachable();
            }

            mir::TerminatorKind::Goto { target } => {
                helper.maybe_sideeffect(self.mir, &mut bx, &[target]);
                helper.funclet_br(self, &mut bx, target);
            }

            mir::TerminatorKind::SwitchInt { ref discr, switch_ty, ref values, ref targets } => {
                self.codegen_switchint_terminator(helper, bx, discr, switch_ty, values, targets);
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
                self.codegen_assert_terminator(
                    helper, bx, terminator, cond, expected, msg, target, cleanup,
                );
            }

            mir::TerminatorKind::DropAndReplace { .. } => {
                bug!("undesugared DropAndReplace in codegen: {:?}", terminator);
            }

            mir::TerminatorKind::Call {
                ref func,
                ref args,
                ref destination,
                cleanup,
                from_hir_call: _,
            } => {
                self.codegen_call_terminator(
                    helper,
                    bx,
                    terminator,
                    func,
                    args,
                    destination,
                    cleanup,
                );
            }
            mir::TerminatorKind::GeneratorDrop | mir::TerminatorKind::Yield { .. } => {
                bug!("generator ops in codegen")
            }
            mir::TerminatorKind::FalseEdges { .. } | mir::TerminatorKind::FalseUnwind { .. } => {
                bug!("borrowck false edges in codegen")
            }
        }
    }

    fn codegen_argument(
        &mut self,
        bx: &mut Bx,
        op: OperandRef<'tcx, Bx::Value>,
        llargs: &mut Vec<Bx::Value>,
        arg: &ArgAbi<'tcx, Ty<'tcx>>,
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
                _ => bug!("codegen_argument: {:?} invalid for pair argument", op),
            }
        } else if arg.is_unsized_indirect() {
            match op.val {
                Ref(a, Some(b), _) => {
                    llargs.push(a);
                    llargs.push(b);
                    return;
                }
                _ => bug!("codegen_argument: {:?} invalid for unsized indirect argument", op),
            }
        }

        // Force by-ref if we have to load through a cast pointer.
        let (mut llval, align, by_ref) = match op.val {
            Immediate(_) | Pair(..) => match arg.mode {
                PassMode::Indirect(..) | PassMode::Cast(_) => {
                    let scratch = PlaceRef::alloca(bx, arg.layout);
                    op.val.store(bx, scratch);
                    (scratch.llval, scratch.align, true)
                }
                _ => (op.immediate_or_packed_pair(bx), arg.layout.align.abi, false),
            },
            Ref(llval, _, align) => {
                if arg.is_indirect() && align < arg.layout.align.abi {
                    // `foo(packed.large_field)`. We can't pass the (unaligned) field directly. I
                    // think that ATM (Rust 1.16) we only pass temporaries, but we shouldn't
                    // have scary latent bugs around.

                    let scratch = PlaceRef::alloca(bx, arg.layout);
                    base::memcpy_ty(
                        bx,
                        scratch.llval,
                        scratch.align,
                        llval,
                        align,
                        op.layout,
                        MemFlags::empty(),
                    );
                    (scratch.llval, scratch.align, true)
                } else {
                    (llval, align, true)
                }
            }
        };

        if by_ref && !arg.is_indirect() {
            // Have to load the argument, maybe while casting it.
            if let PassMode::Cast(ty) = arg.mode {
                let addr = bx.pointercast(llval, bx.type_ptr_to(bx.cast_backend_type(&ty)));
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
        args: &[ArgAbi<'tcx, Ty<'tcx>>],
    ) {
        let tuple = self.codegen_operand(bx, operand);

        // Handle both by-ref and immediate tuples.
        if let Ref(llval, None, align) = tuple.val {
            let tuple_ptr = PlaceRef::new_sized_aligned(llval, tuple.layout, align);
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

    fn get_caller_location(&mut self, bx: &mut Bx, span: Span) -> OperandRef<'tcx, Bx::Value> {
        self.caller_location.unwrap_or_else(|| {
            let topmost = span.ctxt().outer_expn().expansion_cause().unwrap_or(span);
            let caller = bx.tcx().sess.source_map().lookup_char_pos(topmost.lo());
            let const_loc = bx.tcx().const_caller_location((
                Symbol::intern(&caller.file.name.to_string()),
                caller.line as u32,
                caller.col_display as u32 + 1,
            ));
            OperandRef::from_const(bx, const_loc, bx.tcx().caller_location_ty())
        })
    }

    fn get_personality_slot(&mut self, bx: &mut Bx) -> PlaceRef<'tcx, Bx::Value> {
        let cx = bx.cx();
        if let Some(slot) = self.personality_slot {
            slot
        } else {
            let layout = cx.layout_of(
                cx.tcx().intern_tup(&[cx.tcx().mk_mut_ptr(cx.tcx().types.u8), cx.tcx().types.i32]),
            );
            let slot = PlaceRef::alloca(bx, layout);
            self.personality_slot = Some(slot);
            slot
        }
    }

    /// Returns the landing-pad wrapper around the given basic block.
    ///
    /// No-op in MSVC SEH scheme.
    fn landing_pad_to(&mut self, target_bb: mir::BasicBlock) -> Bx::BasicBlock {
        if let Some(block) = self.landing_pads[target_bb] {
            return block;
        }

        let block = self.blocks[target_bb];
        let landing_pad = self.landing_pad_uncached(block);
        self.landing_pads[target_bb] = Some(landing_pad);
        landing_pad
    }

    fn landing_pad_uncached(&mut self, target_bb: Bx::BasicBlock) -> Bx::BasicBlock {
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

    fn unreachable_block(&mut self) -> Bx::BasicBlock {
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

    pub fn build_block(&self, bb: mir::BasicBlock) -> Bx {
        let mut bx = Bx::with_cx(self.cx);
        bx.position_at_end(self.blocks[bb]);
        bx
    }

    fn make_return_dest(
        &mut self,
        bx: &mut Bx,
        dest: &mir::Place<'tcx>,
        fn_ret: &ArgAbi<'tcx, Ty<'tcx>>,
        llargs: &mut Vec<Bx::Value>,
        is_intrinsic: bool,
    ) -> ReturnDest<'tcx, Bx::Value> {
        // If the return is ignored, we can just return a do-nothing `ReturnDest`.
        if fn_ret.is_ignore() {
            return ReturnDest::Nothing;
        }
        let dest = if let Some(index) = dest.as_local() {
            match self.locals[index] {
                LocalRef::Place(dest) => dest,
                LocalRef::UnsizedPlace(_) => bug!("return type must be sized"),
                LocalRef::Operand(None) => {
                    // Handle temporary places, specifically `Operand` ones, as
                    // they don't have `alloca`s.
                    return if fn_ret.is_indirect() {
                        // Odd, but possible, case, we have an operand temporary,
                        // but the calling convention has an indirect return.
                        let tmp = PlaceRef::alloca(bx, fn_ret.layout);
                        tmp.storage_live(bx);
                        llargs.push(tmp.llval);
                        ReturnDest::IndirectOperand(tmp, index)
                    } else if is_intrinsic {
                        // Currently, intrinsics always need a location to store
                        // the result, so we create a temporary `alloca` for the
                        // result.
                        let tmp = PlaceRef::alloca(bx, fn_ret.layout);
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
            self.codegen_place(
                bx,
                mir::PlaceRef { local: dest.local, projection: &dest.projection },
            )
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

    fn codegen_transmute(&mut self, bx: &mut Bx, src: &mir::Operand<'tcx>, dst: &mir::Place<'tcx>) {
        if let Some(index) = dst.as_local() {
            match self.locals[index] {
                LocalRef::Place(place) => self.codegen_transmute_into(bx, src, place),
                LocalRef::UnsizedPlace(_) => bug!("transmute must not involve unsized locals"),
                LocalRef::Operand(None) => {
                    let dst_layout = bx.layout_of(self.monomorphized_place_ty(dst.as_ref()));
                    assert!(!dst_layout.ty.has_erasable_regions());
                    let place = PlaceRef::alloca(bx, dst_layout);
                    place.storage_live(bx);
                    self.codegen_transmute_into(bx, src, place);
                    let op = bx.load_operand(place);
                    place.storage_dead(bx);
                    self.locals[index] = LocalRef::Operand(Some(op));
                    self.debug_introduce_local(bx, index);
                }
                LocalRef::Operand(Some(op)) => {
                    assert!(op.layout.is_zst(), "assigning to initialized SSAtemp");
                }
            }
        } else {
            let dst = self.codegen_place(bx, dst.as_ref());
            self.codegen_transmute_into(bx, src, dst);
        }
    }

    fn codegen_transmute_into(
        &mut self,
        bx: &mut Bx,
        src: &mir::Operand<'tcx>,
        dst: PlaceRef<'tcx, Bx::Value>,
    ) {
        let src = self.codegen_operand(bx, src);
        let llty = bx.backend_type(src.layout);
        let cast_ptr = bx.pointercast(dst.llval, bx.type_ptr_to(llty));
        let align = src.layout.align.abi.min(dst.align);
        src.val.store(bx, PlaceRef::new_sized_aligned(cast_ptr, src.layout, align));
    }

    // Stores the return value of a function call into it's final location.
    fn store_return(
        &mut self,
        bx: &mut Bx,
        dest: ReturnDest<'tcx, Bx::Value>,
        ret_abi: &ArgAbi<'tcx, Ty<'tcx>>,
        llval: Bx::Value,
    ) {
        use self::ReturnDest::*;

        match dest {
            Nothing => (),
            Store(dst) => bx.store_arg(&ret_abi, llval, dst),
            IndirectOperand(tmp, index) => {
                let op = bx.load_operand(tmp);
                tmp.storage_dead(bx);
                self.locals[index] = LocalRef::Operand(Some(op));
                self.debug_introduce_local(bx, index);
            }
            DirectOperand(index) => {
                // If there is a cast, we have to store and reload.
                let op = if let PassMode::Cast(_) = ret_abi.mode {
                    let tmp = PlaceRef::alloca(bx, ret_abi.layout);
                    tmp.storage_live(bx);
                    bx.store_arg(&ret_abi, llval, tmp);
                    let op = bx.load_operand(tmp);
                    tmp.storage_dead(bx);
                    op
                } else {
                    OperandRef::from_immediate_or_packed_pair(bx, llval, ret_abi.layout)
                };
                self.locals[index] = LocalRef::Operand(Some(op));
                self.debug_introduce_local(bx, index);
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
    DirectOperand(mir::Local),
}
