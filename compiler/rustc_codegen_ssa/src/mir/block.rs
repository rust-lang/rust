use super::operand::OperandRef;
use super::operand::OperandValue::{Immediate, Pair, Ref};
use super::place::PlaceRef;
use super::{FunctionCx, LocalRef};

use crate::base;
use crate::common::{self, IntPredicate};
use crate::meth;
use crate::traits::*;
use crate::MemFlags;

use rustc_ast as ast;
use rustc_hir::lang_items::LangItem;
use rustc_index::vec::Idx;
use rustc_middle::mir::AssertKind;
use rustc_middle::mir::{self, SwitchTargets};
use rustc_middle::ty::layout::{HasTyCtxt, LayoutOf};
use rustc_middle::ty::print::{with_no_trimmed_paths, with_no_visible_paths};
use rustc_middle::ty::{self, Instance, Ty, TypeFoldable};
use rustc_span::source_map::Span;
use rustc_span::{sym, Symbol};
use rustc_symbol_mangling::typeid_for_fnabi;
use rustc_target::abi::call::{ArgAbi, FnAbi, PassMode};
use rustc_target::abi::{self, HasDataLayout, WrappingRange};
use rustc_target::spec::abi::Abi;

/// Used by `FunctionCx::codegen_terminator` for emitting common patterns
/// e.g., creating a basic block, calling a function, etc.
struct TerminatorCodegenHelper<'tcx> {
    bb: mir::BasicBlock,
    terminator: &'tcx mir::Terminator<'tcx>,
    funclet_bb: Option<mir::BasicBlock>,
}

impl<'a, 'tcx> TerminatorCodegenHelper<'tcx> {
    /// Returns the appropriate `Funclet` for the current funclet, if on MSVC,
    /// either already previously cached, or newly created, by `landing_pad_for`.
    fn funclet<'b, Bx: BuilderMethods<'a, 'tcx>>(
        &self,
        fx: &'b mut FunctionCx<'a, 'tcx, Bx>,
    ) -> Option<&'b Bx::Funclet> {
        let funclet_bb = self.funclet_bb?;
        if base::wants_msvc_seh(fx.cx.tcx().sess) {
            // If `landing_pad_for` hasn't been called yet to create the `Funclet`,
            // it has to be now. This may not seem necessary, as RPO should lead
            // to all the unwind edges being visited (and so to `landing_pad_for`
            // getting called for them), before building any of the blocks inside
            // the funclet itself - however, if MIR contains edges that end up not
            // being needed in the LLVM IR after monomorphization, the funclet may
            // be unreachable, and we don't have yet a way to skip building it in
            // such an eventuality (which may be a better solution than this).
            if fx.funclets[funclet_bb].is_none() {
                fx.landing_pad_for(funclet_bb);
            }

            Some(
                fx.funclets[funclet_bb]
                    .as_ref()
                    .expect("landing_pad_for didn't also create funclets entry"),
            )
        } else {
            None
        }
    }

    fn lltarget<Bx: BuilderMethods<'a, 'tcx>>(
        &self,
        fx: &mut FunctionCx<'a, 'tcx, Bx>,
        target: mir::BasicBlock,
    ) -> (Bx::BasicBlock, bool) {
        let span = self.terminator.source_info.span;
        let lltarget = fx.llbb(target);
        let target_funclet = fx.cleanup_kinds[target].funclet_bb(target);
        match (self.funclet_bb, target_funclet) {
            (None, None) => (lltarget, false),
            (Some(f), Some(t_f)) if f == t_f || !base::wants_msvc_seh(fx.cx.tcx().sess) => {
                (lltarget, false)
            }
            // jump *into* cleanup - need a landing pad if GNU, cleanup pad if MSVC
            (None, Some(_)) => (fx.landing_pad_for(target), false),
            (Some(_), None) => span_bug!(span, "{:?} - jump out of cleanup?", self.terminator),
            (Some(_), Some(_)) => (fx.landing_pad_for(target), true),
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
        fn_abi: &'tcx FnAbi<'tcx, Ty<'tcx>>,
        fn_ptr: Bx::Value,
        llargs: &[Bx::Value],
        destination: Option<(ReturnDest<'tcx, Bx::Value>, mir::BasicBlock)>,
        cleanup: Option<mir::BasicBlock>,
    ) {
        // If there is a cleanup block and the function we're calling can unwind, then
        // do an invoke, otherwise do a call.
        let fn_ty = bx.fn_decl_backend_type(&fn_abi);
        if let Some(cleanup) = cleanup.filter(|_| fn_abi.can_unwind) {
            let ret_llbb = if let Some((_, target)) = destination {
                fx.llbb(target)
            } else {
                fx.unreachable_block()
            };
            let invokeret = bx.invoke(
                fn_ty,
                fn_ptr,
                &llargs,
                ret_llbb,
                self.llblock(fx, cleanup),
                self.funclet(fx),
            );
            bx.apply_attrs_callsite(&fn_abi, invokeret);

            if let Some((ret_dest, target)) = destination {
                let mut ret_bx = fx.build_block(target);
                fx.set_debug_loc(&mut ret_bx, self.terminator.source_info);
                fx.store_return(&mut ret_bx, ret_dest, &fn_abi.ret, invokeret);
            }
        } else {
            let llret = bx.call(fn_ty, fn_ptr, &llargs, self.funclet(fx));
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

            let mut lp = bx.const_undef(self.landing_pad_type());
            lp = bx.insert_value(lp, lp0, 0);
            lp = bx.insert_value(lp, lp1, 1);
            bx.resume(lp);
        }
    }

    fn codegen_switchint_terminator(
        &mut self,
        helper: TerminatorCodegenHelper<'tcx>,
        mut bx: Bx,
        discr: &mir::Operand<'tcx>,
        switch_ty: Ty<'tcx>,
        targets: &SwitchTargets,
    ) {
        let discr = self.codegen_operand(&mut bx, &discr);
        // `switch_ty` is redundant, sanity-check that.
        assert_eq!(discr.layout.ty, switch_ty);
        let mut target_iter = targets.iter();
        if target_iter.len() == 1 {
            // If there are two targets (one conditional, one fallback), emit br instead of switch
            let (test_value, target) = target_iter.next().unwrap();
            let lltrue = helper.llblock(self, target);
            let llfalse = helper.llblock(self, targets.otherwise());
            if switch_ty == bx.tcx().types.bool {
                // Don't generate trivial icmps when switching on bool
                match test_value {
                    0 => bx.cond_br(discr.immediate(), llfalse, lltrue),
                    1 => bx.cond_br(discr.immediate(), lltrue, llfalse),
                    _ => bug!(),
                }
            } else {
                let switch_llty = bx.immediate_backend_type(bx.layout_of(switch_ty));
                let llval = bx.const_uint_big(switch_llty, test_value);
                let cmp = bx.icmp(IntPredicate::IntEQ, discr.immediate(), llval);
                bx.cond_br(cmp, lltrue, llfalse);
            }
        } else {
            bx.switch(
                discr.immediate(),
                helper.llblock(self, targets.otherwise()),
                target_iter.map(|(value, target)| (value, helper.llblock(self, target))),
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
            PassMode::Ignore | PassMode::Indirect { .. } => {
                bx.ret_void();
                return;
            }

            PassMode::Direct(_) | PassMode::Pair(..) => {
                let op = self.codegen_consume(&mut bx, mir::Place::return_place().as_ref());
                if let Ref(llval, _, align) = op.val {
                    bx.load(bx.backend_type(op.layout), llval, align)
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
                let ty = bx.cast_backend_type(&cast_ty);
                let addr = bx.pointercast(llslot, bx.type_ptr_to(ty));
                bx.load(ty, addr, self.fn_abi.ret.layout.align.abi)
            }
        };
        bx.ret(llval);
    }

    fn codegen_drop_terminator(
        &mut self,
        helper: TerminatorCodegenHelper<'tcx>,
        mut bx: Bx,
        location: mir::Place<'tcx>,
        target: mir::BasicBlock,
        unwind: Option<mir::BasicBlock>,
    ) {
        let ty = location.ty(self.mir, bx.tcx()).ty;
        let ty = self.monomorphize(ty);
        let drop_fn = Instance::resolve_drop_in_place(bx.tcx(), ty);

        if let ty::InstanceDef::DropGlue(_, None) = drop_fn.def {
            // we don't actually need to drop anything.
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
        let (drop_fn, fn_abi) = match ty.kind() {
            // FIXME(eddyb) perhaps move some of this logic into
            // `Instance::resolve_drop_in_place`?
            ty::Dynamic(..) => {
                let virtual_drop = Instance {
                    def: ty::InstanceDef::Virtual(drop_fn.def_id(), 0),
                    substs: drop_fn.substs,
                };
                let fn_abi = bx.fn_abi_of_instance(virtual_drop, ty::List::empty());
                let vtable = args[1];
                args = &args[..1];
                (
                    meth::VirtualIndex::from_index(ty::COMMON_VTABLE_ENTRIES_DROPINPLACE)
                        .get_fn(&mut bx, vtable, &fn_abi),
                    fn_abi,
                )
            }
            _ => (bx.get_fn_addr(drop_fn), bx.fn_abi_of_instance(drop_fn, ty::List::empty())),
        };
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
            if let AssertKind::OverflowNeg(_) = *msg {
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
        let panic_block = bx.build_sibling_block("panic");
        if expected {
            bx.cond_br(cond, lltarget, panic_block.llbb());
        } else {
            bx.cond_br(cond, panic_block.llbb(), lltarget);
        }

        // After this point, bx is the block for the call to panic.
        bx = panic_block;
        self.set_debug_loc(&mut bx, terminator.source_info);

        // Get the location information.
        let location = self.get_caller_location(&mut bx, terminator.source_info).immediate();

        // Put together the arguments to the panic entry point.
        let (lang_item, args) = match msg {
            AssertKind::BoundsCheck { ref len, ref index } => {
                let len = self.codegen_operand(&mut bx, len).immediate();
                let index = self.codegen_operand(&mut bx, index).immediate();
                // It's `fn panic_bounds_check(index: usize, len: usize)`,
                // and `#[track_caller]` adds an implicit third argument.
                (LangItem::PanicBoundsCheck, vec![index, len, location])
            }
            _ => {
                let msg_str = Symbol::intern(msg.description());
                let msg = bx.const_str(msg_str);
                // It's `pub fn panic(expr: &str)`, with the wide reference being passed
                // as two arguments, and `#[track_caller]` adds an implicit third argument.
                (LangItem::Panic, vec![msg.0, msg.1, location])
            }
        };

        // Obtain the panic entry point.
        let def_id = common::langcall(bx.tcx(), Some(span), "", lang_item);
        let instance = ty::Instance::mono(bx.tcx(), def_id);
        let fn_abi = bx.fn_abi_of_instance(instance, ty::List::empty());
        let llfn = bx.get_fn_addr(instance);

        // Codegen the actual panic invoke/call.
        helper.do_call(self, &mut bx, fn_abi, llfn, &args, None, cleanup);
    }

    /// Returns `true` if this is indeed a panic intrinsic and codegen is done.
    fn codegen_panic_intrinsic(
        &mut self,
        helper: &TerminatorCodegenHelper<'tcx>,
        bx: &mut Bx,
        intrinsic: Option<Symbol>,
        instance: Option<Instance<'tcx>>,
        source_info: mir::SourceInfo,
        destination: &Option<(mir::Place<'tcx>, mir::BasicBlock)>,
        cleanup: Option<mir::BasicBlock>,
    ) -> bool {
        // Emit a panic or a no-op for `assert_*` intrinsics.
        // These are intrinsics that compile to panics so that we can get a message
        // which mentions the offending type, even from a const context.
        #[derive(Debug, PartialEq)]
        enum AssertIntrinsic {
            Inhabited,
            ZeroValid,
            UninitValid,
        }
        let panic_intrinsic = intrinsic.and_then(|i| match i {
            sym::assert_inhabited => Some(AssertIntrinsic::Inhabited),
            sym::assert_zero_valid => Some(AssertIntrinsic::ZeroValid),
            sym::assert_uninit_valid => Some(AssertIntrinsic::UninitValid),
            _ => None,
        });
        if let Some(intrinsic) = panic_intrinsic {
            use AssertIntrinsic::*;
            let ty = instance.unwrap().substs.type_at(0);
            let layout = bx.layout_of(ty);
            let do_panic = match intrinsic {
                Inhabited => layout.abi.is_uninhabited(),
                ZeroValid => !layout.might_permit_raw_init(bx, /*zero:*/ true),
                UninitValid => !layout.might_permit_raw_init(bx, /*zero:*/ false),
            };
            if do_panic {
                let msg_str = with_no_visible_paths(|| {
                    with_no_trimmed_paths(|| {
                        if layout.abi.is_uninhabited() {
                            // Use this error even for the other intrinsics as it is more precise.
                            format!("attempted to instantiate uninhabited type `{}`", ty)
                        } else if intrinsic == ZeroValid {
                            format!("attempted to zero-initialize type `{}`, which is invalid", ty)
                        } else {
                            format!(
                                "attempted to leave type `{}` uninitialized, which is invalid",
                                ty
                            )
                        }
                    })
                });
                let msg = bx.const_str(Symbol::intern(&msg_str));
                let location = self.get_caller_location(bx, source_info).immediate();

                // Obtain the panic entry point.
                // FIXME: dedup this with `codegen_assert_terminator` above.
                let def_id =
                    common::langcall(bx.tcx(), Some(source_info.span), "", LangItem::Panic);
                let instance = ty::Instance::mono(bx.tcx(), def_id);
                let fn_abi = bx.fn_abi_of_instance(instance, ty::List::empty());
                let llfn = bx.get_fn_addr(instance);

                // Codegen the actual panic invoke/call.
                helper.do_call(
                    self,
                    bx,
                    fn_abi,
                    llfn,
                    &[msg.0, msg.1, location],
                    destination.as_ref().map(|(_, bb)| (ReturnDest::Nothing, *bb)),
                    cleanup,
                );
            } else {
                // a NOP
                let target = destination.as_ref().unwrap().1;
                helper.funclet_br(self, bx, target)
            }
            true
        } else {
            false
        }
    }

    fn codegen_call_terminator(
        &mut self,
        helper: TerminatorCodegenHelper<'tcx>,
        mut bx: Bx,
        terminator: &mir::Terminator<'tcx>,
        func: &mir::Operand<'tcx>,
        args: &[mir::Operand<'tcx>],
        destination: &Option<(mir::Place<'tcx>, mir::BasicBlock)>,
        cleanup: Option<mir::BasicBlock>,
        fn_span: Span,
    ) {
        let source_info = terminator.source_info;
        let span = source_info.span;

        // Create the callee. This is a fn ptr or zero-sized and hence a kind of scalar.
        let callee = self.codegen_operand(&mut bx, func);

        let (instance, mut llfn) = match *callee.layout.ty.kind() {
            ty::FnDef(def_id, substs) => (
                Some(
                    ty::Instance::resolve(bx.tcx(), ty::ParamEnv::reveal_all(), def_id, substs)
                        .unwrap()
                        .unwrap()
                        .polymorphize(bx.tcx()),
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
            Some(ty::InstanceDef::Intrinsic(def_id)) => Some(bx.tcx().item_name(def_id)),
            _ => None,
        };

        let extra_args = &args[sig.inputs().skip_binder().len()..];
        let extra_args = bx.tcx().mk_type_list(extra_args.iter().map(|op_arg| {
            let op_ty = op_arg.ty(self.mir, bx.tcx());
            self.monomorphize(op_ty)
        }));

        let fn_abi = match instance {
            Some(instance) => bx.fn_abi_of_instance(instance, extra_args),
            None => bx.fn_abi_of_fn_ptr(sig, extra_args),
        };

        if intrinsic == Some(sym::transmute) {
            if let Some(destination_ref) = destination.as_ref() {
                let &(dest, target) = destination_ref;
                self.codegen_transmute(&mut bx, &args[0], dest);
                helper.funclet_br(self, &mut bx, target);
            } else {
                // If we are trying to transmute to an uninhabited type,
                // it is likely there is no allotted destination. In fact,
                // transmuting to an uninhabited type is UB, which means
                // we can do what we like. Here, we declare that transmuting
                // into an uninhabited type is impossible, so anything following
                // it must be unreachable.
                assert_eq!(fn_abi.ret.layout.abi, abi::Abi::Uninhabited);
                bx.unreachable();
            }
            return;
        }

        if self.codegen_panic_intrinsic(
            &helper,
            &mut bx,
            intrinsic,
            instance,
            source_info,
            destination,
            cleanup,
        ) {
            return;
        }

        // The arguments we'll be passing. Plus one to account for outptr, if used.
        let arg_count = fn_abi.args.len() + fn_abi.ret.is_indirect() as usize;
        let mut llargs = Vec::with_capacity(arg_count);

        // Prepare the return value destination
        let ret_dest = if let Some((dest, _)) = *destination {
            let is_intrinsic = intrinsic.is_some();
            self.make_return_dest(&mut bx, dest, &fn_abi.ret, &mut llargs, is_intrinsic)
        } else {
            ReturnDest::Nothing
        };

        if intrinsic == Some(sym::caller_location) {
            if let Some((_, target)) = destination.as_ref() {
                let location = self
                    .get_caller_location(&mut bx, mir::SourceInfo { span: fn_span, ..source_info });

                if let ReturnDest::IndirectOperand(tmp, _) = ret_dest {
                    location.val.store(&mut bx, tmp);
                }
                self.store_return(&mut bx, ret_dest, &fn_abi.ret, location.immediate());
                helper.funclet_br(self, &mut bx, *target);
            }
            return;
        }

        match intrinsic {
            None | Some(sym::drop_in_place) => {}
            Some(sym::copy_nonoverlapping) => unreachable!(),
            Some(intrinsic) => {
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
                        if i == 2 && intrinsic.as_str().starts_with("simd_shuffle") {
                            if let mir::Operand::Constant(constant) = arg {
                                let c = self.eval_mir_constant(constant);
                                let (llval, ty) = self.simd_shuffle_indices(
                                    &bx,
                                    constant.span,
                                    self.monomorphize(constant.ty()),
                                    c,
                                );
                                return OperandRef {
                                    val: Immediate(llval),
                                    layout: bx.layout_of(ty),
                                };
                            } else {
                                span_bug!(span, "shuffle indices must be constant");
                            }
                        }

                        self.codegen_operand(&mut bx, arg)
                    })
                    .collect();

                Self::codegen_intrinsic_call(
                    &mut bx,
                    *instance.as_ref().unwrap(),
                    &fn_abi,
                    &args,
                    dest,
                    span,
                );

                if let ReturnDest::IndirectOperand(dst, _) = ret_dest {
                    self.store_return(&mut bx, ret_dest, &fn_abi.ret, dst.llval);
                }

                if let Some((_, target)) = *destination {
                    helper.funclet_br(self, &mut bx, target);
                } else {
                    bx.unreachable();
                }

                return;
            }
        }

        // Split the rust-call tupled arguments off.
        let (first_args, untuple) = if abi == Abi::RustCall && !args.is_empty() {
            let (tup, args) = args.split_last().unwrap();
            (args, Some(tup))
        } else {
            (args, None)
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
        let num_untupled = untuple.map(|tup| {
            self.codegen_arguments_untupled(
                &mut bx,
                tup,
                &mut llargs,
                &fn_abi.args[first_args.len()..],
            )
        });

        let needs_location =
            instance.map_or(false, |i| i.def.requires_caller_location(self.cx.tcx()));
        if needs_location {
            let mir_args = if let Some(num_untupled) = num_untupled {
                first_args.len() + num_untupled
            } else {
                args.len()
            };
            assert_eq!(
                fn_abi.args.len(),
                mir_args + 1,
                "#[track_caller] fn's must have 1 more argument in their ABI than in their MIR: {:?} {:?} {:?}",
                instance,
                fn_span,
                fn_abi,
            );
            let location =
                self.get_caller_location(&mut bx, mir::SourceInfo { span: fn_span, ..source_info });
            debug!(
                "codegen_call_terminator({:?}): location={:?} (fn_span {:?})",
                terminator, location, fn_span
            );

            let last_arg = fn_abi.args.last().unwrap();
            self.codegen_argument(&mut bx, location, &mut llargs, last_arg);
        }

        let (is_indirect_call, fn_ptr) = match (llfn, instance) {
            (Some(llfn), _) => (true, llfn),
            (None, Some(instance)) => (false, bx.get_fn_addr(instance)),
            _ => span_bug!(span, "no llfn for call"),
        };

        // For backends that support CFI using type membership (i.e., testing whether a given
        // pointer is associated with a type identifier).
        if bx.tcx().sess.is_sanitizer_cfi_enabled() && is_indirect_call {
            // Emit type metadata and checks.
            // FIXME(rcvalle): Add support for generalized identifiers.
            // FIXME(rcvalle): Create distinct unnamed MDNodes for internal identifiers.
            let typeid = typeid_for_fnabi(bx.tcx(), fn_abi);
            let typeid_metadata = bx.typeid_metadata(typeid.clone());

            // Test whether the function pointer is associated with the type identifier.
            let cond = bx.type_test(fn_ptr, typeid_metadata);
            let mut bx_pass = bx.build_sibling_block("type_test.pass");
            let mut bx_fail = bx.build_sibling_block("type_test.fail");
            bx.cond_br(cond, bx_pass.llbb(), bx_fail.llbb());

            helper.do_call(
                self,
                &mut bx_pass,
                fn_abi,
                fn_ptr,
                &llargs,
                destination.as_ref().map(|&(_, target)| (ret_dest, target)),
                cleanup,
            );

            bx_fail.abort();
            bx_fail.unreachable();

            return;
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

    fn codegen_asm_terminator(
        &mut self,
        helper: TerminatorCodegenHelper<'tcx>,
        mut bx: Bx,
        terminator: &mir::Terminator<'tcx>,
        template: &[ast::InlineAsmTemplatePiece],
        operands: &[mir::InlineAsmOperand<'tcx>],
        options: ast::InlineAsmOptions,
        line_spans: &[Span],
        destination: Option<mir::BasicBlock>,
        instance: Instance<'_>,
    ) {
        let span = terminator.source_info.span;

        let operands: Vec<_> = operands
            .iter()
            .map(|op| match *op {
                mir::InlineAsmOperand::In { reg, ref value } => {
                    let value = self.codegen_operand(&mut bx, value);
                    InlineAsmOperandRef::In { reg, value }
                }
                mir::InlineAsmOperand::Out { reg, late, ref place } => {
                    let place = place.map(|place| self.codegen_place(&mut bx, place.as_ref()));
                    InlineAsmOperandRef::Out { reg, late, place }
                }
                mir::InlineAsmOperand::InOut { reg, late, ref in_value, ref out_place } => {
                    let in_value = self.codegen_operand(&mut bx, in_value);
                    let out_place =
                        out_place.map(|out_place| self.codegen_place(&mut bx, out_place.as_ref()));
                    InlineAsmOperandRef::InOut { reg, late, in_value, out_place }
                }
                mir::InlineAsmOperand::Const { ref value } => {
                    let const_value = self
                        .eval_mir_constant(value)
                        .unwrap_or_else(|_| span_bug!(span, "asm const cannot be resolved"));
                    let string = common::asm_const_to_str(
                        bx.tcx(),
                        span,
                        const_value,
                        bx.layout_of(value.ty()),
                    );
                    InlineAsmOperandRef::Const { string }
                }
                mir::InlineAsmOperand::SymFn { ref value } => {
                    let literal = self.monomorphize(value.literal);
                    if let ty::FnDef(def_id, substs) = *literal.ty().kind() {
                        let instance = ty::Instance::resolve_for_fn_ptr(
                            bx.tcx(),
                            ty::ParamEnv::reveal_all(),
                            def_id,
                            substs,
                        )
                        .unwrap();
                        InlineAsmOperandRef::SymFn { instance }
                    } else {
                        span_bug!(span, "invalid type for asm sym (fn)");
                    }
                }
                mir::InlineAsmOperand::SymStatic { def_id } => {
                    InlineAsmOperandRef::SymStatic { def_id }
                }
            })
            .collect();

        bx.codegen_inline_asm(template, &operands, options, line_spans, instance);

        if let Some(target) = destination {
            helper.funclet_br(self, &mut bx, target);
        } else {
            bx.unreachable();
        }
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
                if bb == target {
                    // This is an unconditional branch back to this same basic block. That means we
                    // have something like a `loop {}` statement. LLVM versions before 12.0
                    // miscompile this because they assume forward progress. For older versions
                    // try to handle just this specific case which comes up commonly in practice
                    // (e.g., in embedded code).
                    //
                    // NB: the `sideeffect` currently checks for the LLVM version used internally.
                    bx.sideeffect();
                }

                helper.funclet_br(self, &mut bx, target);
            }

            mir::TerminatorKind::SwitchInt { ref discr, switch_ty, ref targets } => {
                self.codegen_switchint_terminator(helper, bx, discr, switch_ty, targets);
            }

            mir::TerminatorKind::Return => {
                self.codegen_return_terminator(bx);
            }

            mir::TerminatorKind::Unreachable => {
                bx.unreachable();
            }

            mir::TerminatorKind::Drop { place, target, unwind } => {
                self.codegen_drop_terminator(helper, bx, place, target, unwind);
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
                fn_span,
            } => {
                self.codegen_call_terminator(
                    helper,
                    bx,
                    terminator,
                    func,
                    args,
                    destination,
                    cleanup,
                    fn_span,
                );
            }
            mir::TerminatorKind::GeneratorDrop | mir::TerminatorKind::Yield { .. } => {
                bug!("generator ops in codegen")
            }
            mir::TerminatorKind::FalseEdge { .. } | mir::TerminatorKind::FalseUnwind { .. } => {
                bug!("borrowck false edges in codegen")
            }

            mir::TerminatorKind::InlineAsm {
                template,
                ref operands,
                options,
                line_spans,
                destination,
            } => {
                self.codegen_asm_terminator(
                    helper,
                    bx,
                    terminator,
                    template,
                    operands,
                    options,
                    line_spans,
                    destination,
                    self.instance,
                );
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
                PassMode::Indirect { .. } | PassMode::Cast(_) => {
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
                let llty = bx.cast_backend_type(&ty);
                let addr = bx.pointercast(llval, bx.type_ptr_to(llty));
                llval = bx.load(llty, addr, align.min(arg.layout.align.abi));
            } else {
                // We can't use `PlaceRef::load` here because the argument
                // may have a type we don't treat as immediate, but the ABI
                // used for this call is passing it by-value. In that case,
                // the load would just produce `OperandValue::Ref` instead
                // of the `OperandValue::Immediate` we need for the call.
                llval = bx.load(bx.backend_type(arg.layout), llval, align);
                if let abi::Abi::Scalar(scalar) = arg.layout.abi {
                    if scalar.is_bool() {
                        bx.range_metadata(llval, WrappingRange { start: 0, end: 1 });
                    }
                }
                // We store bools as `i8` so we need to truncate to `i1`.
                llval = bx.to_immediate(llval, arg.layout);
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
    ) -> usize {
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
        tuple.layout.fields.count()
    }

    fn get_caller_location(
        &mut self,
        bx: &mut Bx,
        mut source_info: mir::SourceInfo,
    ) -> OperandRef<'tcx, Bx::Value> {
        let tcx = bx.tcx();

        let mut span_to_caller_location = |span: Span| {
            let topmost = span.ctxt().outer_expn().expansion_cause().unwrap_or(span);
            let caller = tcx.sess.source_map().lookup_char_pos(topmost.lo());
            let const_loc = tcx.const_caller_location((
                Symbol::intern(&caller.file.name.prefer_remapped().to_string_lossy()),
                caller.line as u32,
                caller.col_display as u32 + 1,
            ));
            OperandRef::from_const(bx, const_loc, bx.tcx().caller_location_ty())
        };

        // Walk up the `SourceScope`s, in case some of them are from MIR inlining.
        // If so, the starting `source_info.span` is in the innermost inlined
        // function, and will be replaced with outer callsite spans as long
        // as the inlined functions were `#[track_caller]`.
        loop {
            let scope_data = &self.mir.source_scopes[source_info.scope];

            if let Some((callee, callsite_span)) = scope_data.inlined {
                // Stop inside the most nested non-`#[track_caller]` function,
                // before ever reaching its caller (which is irrelevant).
                if !callee.def.requires_caller_location(tcx) {
                    return span_to_caller_location(source_info.span);
                }
                source_info.span = callsite_span;
            }

            // Skip past all of the parents with `inlined: None`.
            match scope_data.inlined_parent_scope {
                Some(parent) => source_info.scope = parent,
                None => break,
            }
        }

        // No inlined `SourceScope`s, or all of them were `#[track_caller]`.
        self.caller_location.unwrap_or_else(|| span_to_caller_location(source_info.span))
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

    /// Returns the landing/cleanup pad wrapper around the given basic block.
    // FIXME(eddyb) rename this to `eh_pad_for`.
    fn landing_pad_for(&mut self, bb: mir::BasicBlock) -> Bx::BasicBlock {
        if let Some(landing_pad) = self.landing_pads[bb] {
            return landing_pad;
        }

        let landing_pad = self.landing_pad_for_uncached(bb);
        self.landing_pads[bb] = Some(landing_pad);
        landing_pad
    }

    // FIXME(eddyb) rename this to `eh_pad_for_uncached`.
    fn landing_pad_for_uncached(&mut self, bb: mir::BasicBlock) -> Bx::BasicBlock {
        let llbb = self.llbb(bb);
        if base::wants_msvc_seh(self.cx.sess()) {
            let funclet;
            let ret_llbb;
            match self.mir[bb].terminator.as_ref().map(|t| &t.kind) {
                // This is a basic block that we're aborting the program for,
                // notably in an `extern` function. These basic blocks are inserted
                // so that we assert that `extern` functions do indeed not panic,
                // and if they do we abort the process.
                //
                // On MSVC these are tricky though (where we're doing funclets). If
                // we were to do a cleanuppad (like below) the normal functions like
                // `longjmp` would trigger the abort logic, terminating the
                // program. Instead we insert the equivalent of `catch(...)` for C++
                // which magically doesn't trigger when `longjmp` files over this
                // frame.
                //
                // Lots more discussion can be found on #48251 but this codegen is
                // modeled after clang's for:
                //
                //      try {
                //          foo();
                //      } catch (...) {
                //          bar();
                //      }
                Some(&mir::TerminatorKind::Abort) => {
                    let mut cs_bx = self.new_block(&format!("cs_funclet{:?}", bb));
                    let mut cp_bx = self.new_block(&format!("cp_funclet{:?}", bb));
                    ret_llbb = cs_bx.llbb();

                    let cs = cs_bx.catch_switch(None, None, 1);
                    cs_bx.add_handler(cs, cp_bx.llbb());

                    // The "null" here is actually a RTTI type descriptor for the
                    // C++ personality function, but `catch (...)` has no type so
                    // it's null. The 64 here is actually a bitfield which
                    // represents that this is a catch-all block.
                    let null = cp_bx.const_null(
                        cp_bx.type_i8p_ext(cp_bx.cx().data_layout().instruction_address_space),
                    );
                    let sixty_four = cp_bx.const_i32(64);
                    funclet = cp_bx.catch_pad(cs, &[null, sixty_four, null]);
                    cp_bx.br(llbb);
                }
                _ => {
                    let mut cleanup_bx = self.new_block(&format!("funclet_{:?}", bb));
                    ret_llbb = cleanup_bx.llbb();
                    funclet = cleanup_bx.cleanup_pad(None, &[]);
                    cleanup_bx.br(llbb);
                }
            }
            self.funclets[bb] = Some(funclet);
            ret_llbb
        } else {
            let mut bx = self.new_block("cleanup");

            let llpersonality = self.cx.eh_personality();
            let llretty = self.landing_pad_type();
            let lp = bx.landing_pad(llretty, llpersonality, 1);
            bx.set_cleanup(lp);

            let slot = self.get_personality_slot(&mut bx);
            slot.storage_live(&mut bx);
            Pair(bx.extract_value(lp, 0), bx.extract_value(lp, 1)).store(&mut bx, slot);

            bx.br(llbb);
            bx.llbb()
        }
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

    // FIXME(eddyb) replace with `build_sibling_block`/`append_sibling_block`
    // (which requires having a `Bx` already, and not all callers do).
    fn new_block(&self, name: &str) -> Bx {
        let llbb = Bx::append_block(self.cx, self.llfn, name);
        Bx::build(self.cx, llbb)
    }

    /// Get the backend `BasicBlock` for a MIR `BasicBlock`, either already
    /// cached in `self.cached_llbbs`, or created on demand (and cached).
    // FIXME(eddyb) rename `llbb` and other `ll`-prefixed things to use a
    // more backend-agnostic prefix such as `cg` (i.e. this would be `cgbb`).
    pub fn llbb(&mut self, bb: mir::BasicBlock) -> Bx::BasicBlock {
        self.cached_llbbs[bb].unwrap_or_else(|| {
            // FIXME(eddyb) only name the block if `fewer_names` is `false`.
            let llbb = Bx::append_block(self.cx, self.llfn, &format!("{:?}", bb));
            self.cached_llbbs[bb] = Some(llbb);
            llbb
        })
    }

    pub fn build_block(&mut self, bb: mir::BasicBlock) -> Bx {
        let llbb = self.llbb(bb);
        Bx::build(self.cx, llbb)
    }

    fn make_return_dest(
        &mut self,
        bx: &mut Bx,
        dest: mir::Place<'tcx>,
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

    fn codegen_transmute(&mut self, bx: &mut Bx, src: &mir::Operand<'tcx>, dst: mir::Place<'tcx>) {
        if let Some(index) = dst.as_local() {
            match self.locals[index] {
                LocalRef::Place(place) => self.codegen_transmute_into(bx, src, place),
                LocalRef::UnsizedPlace(_) => bug!("transmute must not involve unsized locals"),
                LocalRef::Operand(None) => {
                    let dst_layout = bx.layout_of(self.monomorphized_place_ty(dst.as_ref()));
                    assert!(!dst_layout.ty.has_erasable_regions(self.cx.tcx()));
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

        // Special-case transmutes between scalars as simple bitcasts.
        match (src.layout.abi, dst.layout.abi) {
            (abi::Abi::Scalar(src_scalar), abi::Abi::Scalar(dst_scalar)) => {
                // HACK(eddyb) LLVM doesn't like `bitcast`s between pointers and non-pointers.
                if (src_scalar.value == abi::Pointer) == (dst_scalar.value == abi::Pointer) {
                    assert_eq!(src.layout.size, dst.layout.size);

                    // NOTE(eddyb) the `from_immediate` and `to_immediate_scalar`
                    // conversions allow handling `bool`s the same as `u8`s.
                    let src = bx.from_immediate(src.immediate());
                    let src_as_dst = bx.bitcast(src, bx.backend_type(dst.layout));
                    Immediate(bx.to_immediate_scalar(src_as_dst, dst_scalar)).store(bx, dst);
                    return;
                }
            }
            _ => {}
        }

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
