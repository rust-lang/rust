use super::operand::OperandRef;
use super::operand::OperandValue::{Immediate, Pair, Ref, ZeroSized};
use super::place::PlaceRef;
use super::{CachedLlbb, FunctionCx, LocalRef};

use crate::base;
use crate::common::{self, IntPredicate};
use crate::meth;
use crate::traits::*;
use crate::MemFlags;

use rustc_ast as ast;
use rustc_ast::{InlineAsmOptions, InlineAsmTemplatePiece};
use rustc_hir::lang_items::LangItem;
use rustc_middle::mir::{self, AssertKind, SwitchTargets};
use rustc_middle::ty::layout::{HasTyCtxt, LayoutOf, ValidityRequirement};
use rustc_middle::ty::print::{with_no_trimmed_paths, with_no_visible_paths};
use rustc_middle::ty::{self, Instance, Ty};
use rustc_session::config::OptLevel;
use rustc_span::source_map::Span;
use rustc_span::{sym, Symbol};
use rustc_target::abi::call::{ArgAbi, FnAbi, PassMode, Reg};
use rustc_target::abi::{self, HasDataLayout, WrappingRange};
use rustc_target::spec::abi::Abi;

// Indicates if we are in the middle of merging a BB's successor into it. This
// can happen when BB jumps directly to its successor and the successor has no
// other predecessors.
#[derive(Debug, PartialEq)]
enum MergingSucc {
    False,
    True,
}

/// Used by `FunctionCx::codegen_terminator` for emitting common patterns
/// e.g., creating a basic block, calling a function, etc.
struct TerminatorCodegenHelper<'tcx> {
    bb: mir::BasicBlock,
    terminator: &'tcx mir::Terminator<'tcx>,
}

impl<'a, 'tcx> TerminatorCodegenHelper<'tcx> {
    /// Returns the appropriate `Funclet` for the current funclet, if on MSVC,
    /// either already previously cached, or newly created, by `landing_pad_for`.
    fn funclet<'b, Bx: BuilderMethods<'a, 'tcx>>(
        &self,
        fx: &'b mut FunctionCx<'a, 'tcx, Bx>,
    ) -> Option<&'b Bx::Funclet> {
        let cleanup_kinds = (&fx.cleanup_kinds).as_ref()?;
        let funclet_bb = cleanup_kinds[self.bb].funclet_bb(self.bb)?;
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
    }

    /// Get a basic block (creating it if necessary), possibly with cleanup
    /// stuff in it or next to it.
    fn llbb_with_cleanup<Bx: BuilderMethods<'a, 'tcx>>(
        &self,
        fx: &mut FunctionCx<'a, 'tcx, Bx>,
        target: mir::BasicBlock,
    ) -> Bx::BasicBlock {
        let (needs_landing_pad, is_cleanupret) = self.llbb_characteristics(fx, target);
        let mut lltarget = fx.llbb(target);
        if needs_landing_pad {
            lltarget = fx.landing_pad_for(target);
        }
        if is_cleanupret {
            // MSVC cross-funclet jump - need a trampoline
            debug_assert!(base::wants_msvc_seh(fx.cx.tcx().sess));
            debug!("llbb_with_cleanup: creating cleanup trampoline for {:?}", target);
            let name = &format!("{:?}_cleanup_trampoline_{:?}", self.bb, target);
            let trampoline_llbb = Bx::append_block(fx.cx, fx.llfn, name);
            let mut trampoline_bx = Bx::build(fx.cx, trampoline_llbb);
            trampoline_bx.cleanup_ret(self.funclet(fx).unwrap(), Some(lltarget));
            trampoline_llbb
        } else {
            lltarget
        }
    }

    fn llbb_characteristics<Bx: BuilderMethods<'a, 'tcx>>(
        &self,
        fx: &mut FunctionCx<'a, 'tcx, Bx>,
        target: mir::BasicBlock,
    ) -> (bool, bool) {
        if let Some(ref cleanup_kinds) = fx.cleanup_kinds {
            let funclet_bb = cleanup_kinds[self.bb].funclet_bb(self.bb);
            let target_funclet = cleanup_kinds[target].funclet_bb(target);
            let (needs_landing_pad, is_cleanupret) = match (funclet_bb, target_funclet) {
                (None, None) => (false, false),
                (None, Some(_)) => (true, false),
                (Some(f), Some(t_f)) => (f != t_f, f != t_f),
                (Some(_), None) => {
                    let span = self.terminator.source_info.span;
                    span_bug!(span, "{:?} - jump out of cleanup?", self.terminator);
                }
            };
            (needs_landing_pad, is_cleanupret)
        } else {
            let needs_landing_pad = !fx.mir[self.bb].is_cleanup && fx.mir[target].is_cleanup;
            let is_cleanupret = false;
            (needs_landing_pad, is_cleanupret)
        }
    }

    fn funclet_br<Bx: BuilderMethods<'a, 'tcx>>(
        &self,
        fx: &mut FunctionCx<'a, 'tcx, Bx>,
        bx: &mut Bx,
        target: mir::BasicBlock,
        mergeable_succ: bool,
    ) -> MergingSucc {
        let (needs_landing_pad, is_cleanupret) = self.llbb_characteristics(fx, target);
        if mergeable_succ && !needs_landing_pad && !is_cleanupret {
            // We can merge the successor into this bb, so no need for a `br`.
            MergingSucc::True
        } else {
            let mut lltarget = fx.llbb(target);
            if needs_landing_pad {
                lltarget = fx.landing_pad_for(target);
            }
            if is_cleanupret {
                // micro-optimization: generate a `ret` rather than a jump
                // to a trampoline.
                bx.cleanup_ret(self.funclet(fx).unwrap(), Some(lltarget));
            } else {
                bx.br(lltarget);
            }
            MergingSucc::False
        }
    }

    /// Call `fn_ptr` of `fn_abi` with the arguments `llargs`, the optional
    /// return destination `destination` and the unwind action `unwind`.
    fn do_call<Bx: BuilderMethods<'a, 'tcx>>(
        &self,
        fx: &mut FunctionCx<'a, 'tcx, Bx>,
        bx: &mut Bx,
        fn_abi: &'tcx FnAbi<'tcx, Ty<'tcx>>,
        fn_ptr: Bx::Value,
        llargs: &[Bx::Value],
        destination: Option<(ReturnDest<'tcx, Bx::Value>, mir::BasicBlock)>,
        mut unwind: mir::UnwindAction,
        copied_constant_arguments: &[PlaceRef<'tcx, <Bx as BackendTypes>::Value>],
        mergeable_succ: bool,
    ) -> MergingSucc {
        // If there is a cleanup block and the function we're calling can unwind, then
        // do an invoke, otherwise do a call.
        let fn_ty = bx.fn_decl_backend_type(&fn_abi);

        let fn_attrs = if bx.tcx().def_kind(fx.instance.def_id()).has_codegen_attrs() {
            Some(bx.tcx().codegen_fn_attrs(fx.instance.def_id()))
        } else {
            None
        };

        if !fn_abi.can_unwind {
            unwind = mir::UnwindAction::Unreachable;
        }

        let unwind_block = match unwind {
            mir::UnwindAction::Cleanup(cleanup) => Some(self.llbb_with_cleanup(fx, cleanup)),
            mir::UnwindAction::Continue => None,
            mir::UnwindAction::Unreachable => None,
            mir::UnwindAction::Terminate => {
                if fx.mir[self.bb].is_cleanup && base::wants_msvc_seh(fx.cx.tcx().sess) {
                    // SEH will abort automatically if an exception tries to
                    // propagate out from cleanup.
                    None
                } else {
                    Some(fx.terminate_block())
                }
            }
        };

        if let Some(unwind_block) = unwind_block {
            let ret_llbb = if let Some((_, target)) = destination {
                fx.llbb(target)
            } else {
                fx.unreachable_block()
            };
            let invokeret = bx.invoke(
                fn_ty,
                fn_attrs,
                Some(&fn_abi),
                fn_ptr,
                &llargs,
                ret_llbb,
                unwind_block,
                self.funclet(fx),
            );
            if fx.mir[self.bb].is_cleanup {
                bx.do_not_inline(invokeret);
            }

            if let Some((ret_dest, target)) = destination {
                bx.switch_to_block(fx.llbb(target));
                fx.set_debug_loc(bx, self.terminator.source_info);
                for tmp in copied_constant_arguments {
                    bx.lifetime_end(tmp.llval, tmp.layout.size);
                }
                fx.store_return(bx, ret_dest, &fn_abi.ret, invokeret);
            }
            MergingSucc::False
        } else {
            let llret = bx.call(fn_ty, fn_attrs, Some(&fn_abi), fn_ptr, &llargs, self.funclet(fx));
            if fx.mir[self.bb].is_cleanup {
                // Cleanup is always the cold path. Don't inline
                // drop glue. Also, when there is a deeply-nested
                // struct, there are "symmetry" issues that cause
                // exponential inlining - see issue #41696.
                bx.do_not_inline(llret);
            }

            if let Some((ret_dest, target)) = destination {
                for tmp in copied_constant_arguments {
                    bx.lifetime_end(tmp.llval, tmp.layout.size);
                }
                fx.store_return(bx, ret_dest, &fn_abi.ret, llret);
                self.funclet_br(fx, bx, target, mergeable_succ)
            } else {
                bx.unreachable();
                MergingSucc::False
            }
        }
    }

    /// Generates inline assembly with optional `destination` and `unwind`.
    fn do_inlineasm<Bx: BuilderMethods<'a, 'tcx>>(
        &self,
        fx: &mut FunctionCx<'a, 'tcx, Bx>,
        bx: &mut Bx,
        template: &[InlineAsmTemplatePiece],
        operands: &[InlineAsmOperandRef<'tcx, Bx>],
        options: InlineAsmOptions,
        line_spans: &[Span],
        destination: Option<mir::BasicBlock>,
        unwind: mir::UnwindAction,
        instance: Instance<'_>,
        mergeable_succ: bool,
    ) -> MergingSucc {
        let unwind_target = match unwind {
            mir::UnwindAction::Cleanup(cleanup) => Some(self.llbb_with_cleanup(fx, cleanup)),
            mir::UnwindAction::Terminate => Some(fx.terminate_block()),
            mir::UnwindAction::Continue => None,
            mir::UnwindAction::Unreachable => None,
        };

        if let Some(cleanup) = unwind_target {
            let ret_llbb = if let Some(target) = destination {
                fx.llbb(target)
            } else {
                fx.unreachable_block()
            };

            bx.codegen_inline_asm(
                template,
                &operands,
                options,
                line_spans,
                instance,
                Some((ret_llbb, cleanup, self.funclet(fx))),
            );
            MergingSucc::False
        } else {
            bx.codegen_inline_asm(template, &operands, options, line_spans, instance, None);

            if let Some(target) = destination {
                self.funclet_br(fx, bx, target, mergeable_succ)
            } else {
                bx.unreachable();
                MergingSucc::False
            }
        }
    }
}

/// Codegen implementations for some terminator variants.
impl<'a, 'tcx, Bx: BuilderMethods<'a, 'tcx>> FunctionCx<'a, 'tcx, Bx> {
    /// Generates code for a `Resume` terminator.
    fn codegen_resume_terminator(&mut self, helper: TerminatorCodegenHelper<'tcx>, bx: &mut Bx) {
        if let Some(funclet) = helper.funclet(self) {
            bx.cleanup_ret(funclet, None);
        } else {
            let slot = self.get_personality_slot(bx);
            let exn0 = slot.project_field(bx, 0);
            let exn0 = bx.load_operand(exn0).immediate();
            let exn1 = slot.project_field(bx, 1);
            let exn1 = bx.load_operand(exn1).immediate();
            slot.storage_dead(bx);

            bx.resume(exn0, exn1);
        }
    }

    fn codegen_switchint_terminator(
        &mut self,
        helper: TerminatorCodegenHelper<'tcx>,
        bx: &mut Bx,
        discr: &mir::Operand<'tcx>,
        targets: &SwitchTargets,
    ) {
        let discr = self.codegen_operand(bx, &discr);
        let switch_ty = discr.layout.ty;
        let mut target_iter = targets.iter();
        if target_iter.len() == 1 {
            // If there are two targets (one conditional, one fallback), emit `br` instead of
            // `switch`.
            let (test_value, target) = target_iter.next().unwrap();
            let lltrue = helper.llbb_with_cleanup(self, target);
            let llfalse = helper.llbb_with_cleanup(self, targets.otherwise());
            if switch_ty == bx.tcx().types.bool {
                // Don't generate trivial icmps when switching on bool.
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
        } else if self.cx.sess().opts.optimize == OptLevel::No
            && target_iter.len() == 2
            && self.mir[targets.otherwise()].is_empty_unreachable()
        {
            // In unoptimized builds, if there are two normal targets and the `otherwise` target is
            // an unreachable BB, emit `br` instead of `switch`. This leaves behind the unreachable
            // BB, which will usually (but not always) be dead code.
            //
            // Why only in unoptimized builds?
            // - In unoptimized builds LLVM uses FastISel which does not support switches, so it
            //   must fall back to the to the slower SelectionDAG isel. Therefore, using `br` gives
            //   significant compile time speedups for unoptimized builds.
            // - In optimized builds the above doesn't hold, and using `br` sometimes results in
            //   worse generated code because LLVM can no longer tell that the value being switched
            //   on can only have two values, e.g. 0 and 1.
            //
            let (test_value1, target1) = target_iter.next().unwrap();
            let (_test_value2, target2) = target_iter.next().unwrap();
            let ll1 = helper.llbb_with_cleanup(self, target1);
            let ll2 = helper.llbb_with_cleanup(self, target2);
            let switch_llty = bx.immediate_backend_type(bx.layout_of(switch_ty));
            let llval = bx.const_uint_big(switch_llty, test_value1);
            let cmp = bx.icmp(IntPredicate::IntEQ, discr.immediate(), llval);
            bx.cond_br(cmp, ll1, ll2);
        } else {
            bx.switch(
                discr.immediate(),
                helper.llbb_with_cleanup(self, targets.otherwise()),
                target_iter.map(|(value, target)| (value, helper.llbb_with_cleanup(self, target))),
            );
        }
    }

    fn codegen_return_terminator(&mut self, bx: &mut Bx) {
        // Call `va_end` if this is the definition of a C-variadic function.
        if self.fn_abi.c_variadic {
            // The `VaList` "spoofed" argument is just after all the real arguments.
            let va_list_arg_idx = self.fn_abi.args.len();
            match self.locals[mir::Local::from_usize(1 + va_list_arg_idx)] {
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
        let llval = match &self.fn_abi.ret.mode {
            PassMode::Ignore | PassMode::Indirect { .. } => {
                bx.ret_void();
                return;
            }

            PassMode::Direct(_) | PassMode::Pair(..) => {
                let op = self.codegen_consume(bx, mir::Place::return_place().as_ref());
                if let Ref(llval, _, align) = op.val {
                    bx.load(bx.backend_type(op.layout), llval, align)
                } else {
                    op.immediate_or_packed_pair(bx)
                }
            }

            PassMode::Cast(cast_ty, _) => {
                let op = match self.locals[mir::RETURN_PLACE] {
                    LocalRef::Operand(op) => op,
                    LocalRef::PendingOperand => bug!("use of return before def"),
                    LocalRef::Place(cg_place) => OperandRef {
                        val: Ref(cg_place.llval, None, cg_place.align),
                        layout: cg_place.layout,
                    },
                    LocalRef::UnsizedPlace(_) => bug!("return type must be sized"),
                };
                let llslot = match op.val {
                    Immediate(_) | Pair(..) => {
                        let scratch = PlaceRef::alloca(bx, self.fn_abi.ret.layout);
                        op.val.store(bx, scratch);
                        scratch.llval
                    }
                    Ref(llval, _, align) => {
                        assert_eq!(align, op.layout.align.abi, "return place is unaligned!");
                        llval
                    }
                    ZeroSized => bug!("ZST return value shouldn't be in PassMode::Cast"),
                };
                let ty = bx.cast_backend_type(cast_ty);
                let addr = bx.pointercast(llslot, bx.type_ptr_to(ty));
                bx.load(ty, addr, self.fn_abi.ret.layout.align.abi)
            }
        };
        bx.ret(llval);
    }

    #[tracing::instrument(level = "trace", skip(self, helper, bx))]
    fn codegen_drop_terminator(
        &mut self,
        helper: TerminatorCodegenHelper<'tcx>,
        bx: &mut Bx,
        location: mir::Place<'tcx>,
        target: mir::BasicBlock,
        unwind: mir::UnwindAction,
        mergeable_succ: bool,
    ) -> MergingSucc {
        let ty = location.ty(self.mir, bx.tcx()).ty;
        let ty = self.monomorphize(ty);
        let drop_fn = Instance::resolve_drop_in_place(bx.tcx(), ty);

        if let ty::InstanceDef::DropGlue(_, None) = drop_fn.def {
            // we don't actually need to drop anything.
            return helper.funclet_br(self, bx, target, mergeable_succ);
        }

        let place = self.codegen_place(bx, location.as_ref());
        let (args1, args2);
        let mut args = if let Some(llextra) = place.llextra {
            args2 = [place.llval, llextra];
            &args2[..]
        } else {
            args1 = [place.llval];
            &args1[..]
        };
        let (drop_fn, fn_abi) =
            match ty.kind() {
                // FIXME(eddyb) perhaps move some of this logic into
                // `Instance::resolve_drop_in_place`?
                ty::Dynamic(_, _, ty::Dyn) => {
                    // IN THIS ARM, WE HAVE:
                    // ty = *mut (dyn Trait)
                    // which is: exists<T> ( *mut T,    Vtable<T: Trait> )
                    //                       args[0]    args[1]
                    //
                    // args = ( Data, Vtable )
                    //                  |
                    //                  v
                    //                /-------\
                    //                | ...   |
                    //                \-------/
                    //
                    let virtual_drop = Instance {
                        def: ty::InstanceDef::Virtual(drop_fn.def_id(), 0),
                        substs: drop_fn.substs,
                    };
                    debug!("ty = {:?}", ty);
                    debug!("drop_fn = {:?}", drop_fn);
                    debug!("args = {:?}", args);
                    let fn_abi = bx.fn_abi_of_instance(virtual_drop, ty::List::empty());
                    let vtable = args[1];
                    // Truncate vtable off of args list
                    args = &args[..1];
                    (
                        meth::VirtualIndex::from_index(ty::COMMON_VTABLE_ENTRIES_DROPINPLACE)
                            .get_fn(bx, vtable, ty, &fn_abi),
                        fn_abi,
                    )
                }
                ty::Dynamic(_, _, ty::DynStar) => {
                    // IN THIS ARM, WE HAVE:
                    // ty = *mut (dyn* Trait)
                    // which is: *mut exists<T: sizeof(T) == sizeof(usize)> (T, Vtable<T: Trait>)
                    //
                    // args = [ * ]
                    //          |
                    //          v
                    //      ( Data, Vtable )
                    //                |
                    //                v
                    //              /-------\
                    //              | ...   |
                    //              \-------/
                    //
                    //
                    // WE CAN CONVERT THIS INTO THE ABOVE LOGIC BY DOING
                    //
                    // data = &(*args[0]).0    // gives a pointer to Data above (really the same pointer)
                    // vtable = (*args[0]).1   // loads the vtable out
                    // (data, vtable)          // an equivalent Rust `*mut dyn Trait`
                    //
                    // SO THEN WE CAN USE THE ABOVE CODE.
                    let virtual_drop = Instance {
                        def: ty::InstanceDef::Virtual(drop_fn.def_id(), 0),
                        substs: drop_fn.substs,
                    };
                    debug!("ty = {:?}", ty);
                    debug!("drop_fn = {:?}", drop_fn);
                    debug!("args = {:?}", args);
                    let fn_abi = bx.fn_abi_of_instance(virtual_drop, ty::List::empty());
                    let meta_ptr = place.project_field(bx, 1);
                    let meta = bx.load_operand(meta_ptr);
                    // Truncate vtable off of args list
                    args = &args[..1];
                    debug!("args' = {:?}", args);
                    (
                        meth::VirtualIndex::from_index(ty::COMMON_VTABLE_ENTRIES_DROPINPLACE)
                            .get_fn(bx, meta.immediate(), ty, &fn_abi),
                        fn_abi,
                    )
                }
                _ => (bx.get_fn_addr(drop_fn), bx.fn_abi_of_instance(drop_fn, ty::List::empty())),
            };
        helper.do_call(
            self,
            bx,
            fn_abi,
            drop_fn,
            args,
            Some((ReturnDest::Nothing, target)),
            unwind,
            &[],
            mergeable_succ,
        )
    }

    fn codegen_assert_terminator(
        &mut self,
        helper: TerminatorCodegenHelper<'tcx>,
        bx: &mut Bx,
        terminator: &mir::Terminator<'tcx>,
        cond: &mir::Operand<'tcx>,
        expected: bool,
        msg: &mir::AssertMessage<'tcx>,
        target: mir::BasicBlock,
        unwind: mir::UnwindAction,
        mergeable_succ: bool,
    ) -> MergingSucc {
        let span = terminator.source_info.span;
        let cond = self.codegen_operand(bx, cond).immediate();
        let mut const_cond = bx.const_to_opt_u128(cond, false).map(|c| c == 1);

        // This case can currently arise only from functions marked
        // with #[rustc_inherit_overflow_checks] and inlined from
        // another crate (mostly core::num generic/#[inline] fns),
        // while the current crate doesn't use overflow checks.
        if !bx.cx().check_overflow() && msg.is_optional_overflow_check() {
            const_cond = Some(expected);
        }

        // Don't codegen the panic block if success if known.
        if const_cond == Some(expected) {
            return helper.funclet_br(self, bx, target, mergeable_succ);
        }

        // Pass the condition through llvm.expect for branch hinting.
        let cond = bx.expect(cond, expected);

        // Create the failure block and the conditional branch to it.
        let lltarget = helper.llbb_with_cleanup(self, target);
        let panic_block = bx.append_sibling_block("panic");
        if expected {
            bx.cond_br(cond, lltarget, panic_block);
        } else {
            bx.cond_br(cond, panic_block, lltarget);
        }

        // After this point, bx is the block for the call to panic.
        bx.switch_to_block(panic_block);
        self.set_debug_loc(bx, terminator.source_info);

        // Get the location information.
        let location = self.get_caller_location(bx, terminator.source_info).immediate();

        // Put together the arguments to the panic entry point.
        let (lang_item, args) = match msg {
            AssertKind::BoundsCheck { ref len, ref index } => {
                let len = self.codegen_operand(bx, len).immediate();
                let index = self.codegen_operand(bx, index).immediate();
                // It's `fn panic_bounds_check(index: usize, len: usize)`,
                // and `#[track_caller]` adds an implicit third argument.
                (LangItem::PanicBoundsCheck, vec![index, len, location])
            }
            AssertKind::MisalignedPointerDereference { ref required, ref found } => {
                let required = self.codegen_operand(bx, required).immediate();
                let found = self.codegen_operand(bx, found).immediate();
                // It's `fn panic_bounds_check(index: usize, len: usize)`,
                // and `#[track_caller]` adds an implicit third argument.
                (LangItem::PanicMisalignedPointerDereference, vec![required, found, location])
            }
            _ => {
                let msg = bx.const_str(msg.description());
                // It's `pub fn panic(expr: &str)`, with the wide reference being passed
                // as two arguments, and `#[track_caller]` adds an implicit third argument.
                (LangItem::Panic, vec![msg.0, msg.1, location])
            }
        };

        let (fn_abi, llfn) = common::build_langcall(bx, Some(span), lang_item);

        // Codegen the actual panic invoke/call.
        let merging_succ = helper.do_call(self, bx, fn_abi, llfn, &args, None, unwind, &[], false);
        assert_eq!(merging_succ, MergingSucc::False);
        MergingSucc::False
    }

    fn codegen_terminate_terminator(
        &mut self,
        helper: TerminatorCodegenHelper<'tcx>,
        bx: &mut Bx,
        terminator: &mir::Terminator<'tcx>,
    ) {
        let span = terminator.source_info.span;
        self.set_debug_loc(bx, terminator.source_info);

        // Obtain the panic entry point.
        let (fn_abi, llfn) = common::build_langcall(bx, Some(span), LangItem::PanicCannotUnwind);

        // Codegen the actual panic invoke/call.
        let merging_succ = helper.do_call(
            self,
            bx,
            fn_abi,
            llfn,
            &[],
            None,
            mir::UnwindAction::Unreachable,
            &[],
            false,
        );
        assert_eq!(merging_succ, MergingSucc::False);
    }

    /// Returns `Some` if this is indeed a panic intrinsic and codegen is done.
    fn codegen_panic_intrinsic(
        &mut self,
        helper: &TerminatorCodegenHelper<'tcx>,
        bx: &mut Bx,
        intrinsic: Option<Symbol>,
        instance: Option<Instance<'tcx>>,
        source_info: mir::SourceInfo,
        target: Option<mir::BasicBlock>,
        unwind: mir::UnwindAction,
        mergeable_succ: bool,
    ) -> Option<MergingSucc> {
        // Emit a panic or a no-op for `assert_*` intrinsics.
        // These are intrinsics that compile to panics so that we can get a message
        // which mentions the offending type, even from a const context.
        let panic_intrinsic = intrinsic.and_then(|s| ValidityRequirement::from_intrinsic(s));
        if let Some(requirement) = panic_intrinsic {
            let ty = instance.unwrap().substs.type_at(0);

            let do_panic = !bx
                .tcx()
                .check_validity_requirement((requirement, bx.param_env().and(ty)))
                .expect("expect to have layout during codegen");

            let layout = bx.layout_of(ty);

            Some(if do_panic {
                let msg_str = with_no_visible_paths!({
                    with_no_trimmed_paths!({
                        if layout.abi.is_uninhabited() {
                            // Use this error even for the other intrinsics as it is more precise.
                            format!("attempted to instantiate uninhabited type `{}`", ty)
                        } else if requirement == ValidityRequirement::Zero {
                            format!("attempted to zero-initialize type `{}`, which is invalid", ty)
                        } else {
                            format!(
                                "attempted to leave type `{}` uninitialized, which is invalid",
                                ty
                            )
                        }
                    })
                });
                let msg = bx.const_str(&msg_str);

                // Obtain the panic entry point.
                let (fn_abi, llfn) =
                    common::build_langcall(bx, Some(source_info.span), LangItem::PanicNounwind);

                // Codegen the actual panic invoke/call.
                helper.do_call(
                    self,
                    bx,
                    fn_abi,
                    llfn,
                    &[msg.0, msg.1],
                    target.as_ref().map(|bb| (ReturnDest::Nothing, *bb)),
                    unwind,
                    &[],
                    mergeable_succ,
                )
            } else {
                // a NOP
                let target = target.unwrap();
                helper.funclet_br(self, bx, target, mergeable_succ)
            })
        } else {
            None
        }
    }

    fn codegen_call_terminator(
        &mut self,
        helper: TerminatorCodegenHelper<'tcx>,
        bx: &mut Bx,
        terminator: &mir::Terminator<'tcx>,
        func: &mir::Operand<'tcx>,
        args: &[mir::Operand<'tcx>],
        destination: mir::Place<'tcx>,
        target: Option<mir::BasicBlock>,
        unwind: mir::UnwindAction,
        fn_span: Span,
        mergeable_succ: bool,
    ) -> MergingSucc {
        let source_info = terminator.source_info;
        let span = source_info.span;

        // Create the callee. This is a fn ptr or zero-sized and hence a kind of scalar.
        let callee = self.codegen_operand(bx, func);

        let (instance, mut llfn) = match *callee.layout.ty.kind() {
            ty::FnDef(def_id, substs) => (
                Some(
                    ty::Instance::expect_resolve(
                        bx.tcx(),
                        ty::ParamEnv::reveal_all(),
                        def_id,
                        substs,
                    )
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
            let target = target.unwrap();
            return helper.funclet_br(self, bx, target, mergeable_succ);
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
        let extra_args = bx.tcx().mk_type_list_from_iter(extra_args.iter().map(|op_arg| {
            let op_ty = op_arg.ty(self.mir, bx.tcx());
            self.monomorphize(op_ty)
        }));

        let fn_abi = match instance {
            Some(instance) => bx.fn_abi_of_instance(instance, extra_args),
            None => bx.fn_abi_of_fn_ptr(sig, extra_args),
        };

        if let Some(merging_succ) = self.codegen_panic_intrinsic(
            &helper,
            bx,
            intrinsic,
            instance,
            source_info,
            target,
            unwind,
            mergeable_succ,
        ) {
            return merging_succ;
        }

        // The arguments we'll be passing. Plus one to account for outptr, if used.
        let arg_count = fn_abi.args.len() + fn_abi.ret.is_indirect() as usize;
        let mut llargs = Vec::with_capacity(arg_count);

        // Prepare the return value destination
        let ret_dest = if target.is_some() {
            let is_intrinsic = intrinsic.is_some();
            self.make_return_dest(bx, destination, &fn_abi.ret, &mut llargs, is_intrinsic)
        } else {
            ReturnDest::Nothing
        };

        if intrinsic == Some(sym::caller_location) {
            return if let Some(target) = target {
                let location =
                    self.get_caller_location(bx, mir::SourceInfo { span: fn_span, ..source_info });

                if let ReturnDest::IndirectOperand(tmp, _) = ret_dest {
                    location.val.store(bx, tmp);
                }
                self.store_return(bx, ret_dest, &fn_abi.ret, location.immediate());
                helper.funclet_br(self, bx, target, mergeable_succ)
            } else {
                MergingSucc::False
            };
        }

        match intrinsic {
            None | Some(sym::drop_in_place) => {}
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

                        self.codegen_operand(bx, arg)
                    })
                    .collect();

                Self::codegen_intrinsic_call(
                    bx,
                    *instance.as_ref().unwrap(),
                    &fn_abi,
                    &args,
                    dest,
                    span,
                );

                if let ReturnDest::IndirectOperand(dst, _) = ret_dest {
                    self.store_return(bx, ret_dest, &fn_abi.ret, dst.llval);
                }

                return if let Some(target) = target {
                    helper.funclet_br(self, bx, target, mergeable_succ)
                } else {
                    bx.unreachable();
                    MergingSucc::False
                };
            }
        }

        // Split the rust-call tupled arguments off.
        let (first_args, untuple) = if abi == Abi::RustCall && !args.is_empty() {
            let (tup, args) = args.split_last().unwrap();
            (args, Some(tup))
        } else {
            (args, None)
        };

        let mut copied_constant_arguments = vec![];
        'make_args: for (i, arg) in first_args.iter().enumerate() {
            let mut op = self.codegen_operand(bx, arg);

            if let (0, Some(ty::InstanceDef::Virtual(_, idx))) = (i, def) {
                match op.val {
                    Pair(data_ptr, meta) => {
                        // In the case of Rc<Self>, we need to explicitly pass a
                        // *mut RcBox<Self> with a Scalar (not ScalarPair) ABI. This is a hack
                        // that is understood elsewhere in the compiler as a method on
                        // `dyn Trait`.
                        // To get a `*mut RcBox<Self>`, we just keep unwrapping newtypes until
                        // we get a value of a built-in pointer type.
                        //
                        // This is also relevant for `Pin<&mut Self>`, where we need to peel the `Pin`.
                        'descend_newtypes: while !op.layout.ty.is_unsafe_ptr()
                            && !op.layout.ty.is_ref()
                        {
                            for i in 0..op.layout.fields.count() {
                                let field = op.extract_field(bx, i);
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
                        llfn = Some(meth::VirtualIndex::from_index(idx).get_fn(
                            bx,
                            meta,
                            op.layout.ty,
                            &fn_abi,
                        ));
                        llargs.push(data_ptr);
                        continue 'make_args;
                    }
                    Ref(data_ptr, Some(meta), _) => {
                        // by-value dynamic dispatch
                        llfn = Some(meth::VirtualIndex::from_index(idx).get_fn(
                            bx,
                            meta,
                            op.layout.ty,
                            &fn_abi,
                        ));
                        llargs.push(data_ptr);
                        continue;
                    }
                    Immediate(_) => {
                        // See comment above explaining why we peel these newtypes
                        'descend_newtypes: while !op.layout.ty.is_unsafe_ptr()
                            && !op.layout.ty.is_ref()
                        {
                            for i in 0..op.layout.fields.count() {
                                let field = op.extract_field(bx, i);
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

                        // Make sure that we've actually unwrapped the rcvr down
                        // to a pointer or ref to `dyn* Trait`.
                        if !op.layout.ty.builtin_deref(true).unwrap().ty.is_dyn_star() {
                            span_bug!(span, "can't codegen a virtual call on {:#?}", op);
                        }
                        let place = op.deref(bx.cx());
                        let data_ptr = place.project_field(bx, 0);
                        let meta_ptr = place.project_field(bx, 1);
                        let meta = bx.load_operand(meta_ptr);
                        llfn = Some(meth::VirtualIndex::from_index(idx).get_fn(
                            bx,
                            meta.immediate(),
                            op.layout.ty,
                            &fn_abi,
                        ));
                        llargs.push(data_ptr.llval);
                        continue;
                    }
                    _ => {
                        span_bug!(span, "can't codegen a virtual call on {:#?}", op);
                    }
                }
            }

            // The callee needs to own the argument memory if we pass it
            // by-ref, so make a local copy of non-immediate constants.
            match (arg, op.val) {
                (&mir::Operand::Copy(_), Ref(_, None, _))
                | (&mir::Operand::Constant(_), Ref(_, None, _)) => {
                    let tmp = PlaceRef::alloca(bx, op.layout);
                    bx.lifetime_start(tmp.llval, tmp.layout.size);
                    op.val.store(bx, tmp);
                    op.val = Ref(tmp.llval, None, tmp.align);
                    copied_constant_arguments.push(tmp);
                }
                _ => {}
            }

            self.codegen_argument(bx, op, &mut llargs, &fn_abi.args[i]);
        }
        let num_untupled = untuple.map(|tup| {
            self.codegen_arguments_untupled(bx, tup, &mut llargs, &fn_abi.args[first_args.len()..])
        });

        let needs_location =
            instance.is_some_and(|i| i.def.requires_caller_location(self.cx.tcx()));
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
                self.get_caller_location(bx, mir::SourceInfo { span: fn_span, ..source_info });
            debug!(
                "codegen_call_terminator({:?}): location={:?} (fn_span {:?})",
                terminator, location, fn_span
            );

            let last_arg = fn_abi.args.last().unwrap();
            self.codegen_argument(bx, location, &mut llargs, last_arg);
        }

        let fn_ptr = match (instance, llfn) {
            (Some(instance), None) => bx.get_fn_addr(instance),
            (_, Some(llfn)) => llfn,
            _ => span_bug!(span, "no instance or llfn for call"),
        };

        helper.do_call(
            self,
            bx,
            fn_abi,
            fn_ptr,
            &llargs,
            target.as_ref().map(|&target| (ret_dest, target)),
            unwind,
            &copied_constant_arguments,
            mergeable_succ,
        )
    }

    fn codegen_asm_terminator(
        &mut self,
        helper: TerminatorCodegenHelper<'tcx>,
        bx: &mut Bx,
        terminator: &mir::Terminator<'tcx>,
        template: &[ast::InlineAsmTemplatePiece],
        operands: &[mir::InlineAsmOperand<'tcx>],
        options: ast::InlineAsmOptions,
        line_spans: &[Span],
        destination: Option<mir::BasicBlock>,
        unwind: mir::UnwindAction,
        instance: Instance<'_>,
        mergeable_succ: bool,
    ) -> MergingSucc {
        let span = terminator.source_info.span;

        let operands: Vec<_> = operands
            .iter()
            .map(|op| match *op {
                mir::InlineAsmOperand::In { reg, ref value } => {
                    let value = self.codegen_operand(bx, value);
                    InlineAsmOperandRef::In { reg, value }
                }
                mir::InlineAsmOperand::Out { reg, late, ref place } => {
                    let place = place.map(|place| self.codegen_place(bx, place.as_ref()));
                    InlineAsmOperandRef::Out { reg, late, place }
                }
                mir::InlineAsmOperand::InOut { reg, late, ref in_value, ref out_place } => {
                    let in_value = self.codegen_operand(bx, in_value);
                    let out_place =
                        out_place.map(|out_place| self.codegen_place(bx, out_place.as_ref()));
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

        helper.do_inlineasm(
            self,
            bx,
            template,
            &operands,
            options,
            line_spans,
            destination,
            unwind,
            instance,
            mergeable_succ,
        )
    }
}

impl<'a, 'tcx, Bx: BuilderMethods<'a, 'tcx>> FunctionCx<'a, 'tcx, Bx> {
    pub fn codegen_block(&mut self, mut bb: mir::BasicBlock) {
        let llbb = match self.try_llbb(bb) {
            Some(llbb) => llbb,
            None => return,
        };
        let bx = &mut Bx::build(self.cx, llbb);
        let mir = self.mir;

        // MIR basic blocks stop at any function call. This may not be the case
        // for the backend's basic blocks, in which case we might be able to
        // combine multiple MIR basic blocks into a single backend basic block.
        loop {
            let data = &mir[bb];

            debug!("codegen_block({:?}={:?})", bb, data);

            for statement in &data.statements {
                self.codegen_statement(bx, statement);
            }

            let merging_succ = self.codegen_terminator(bx, bb, data.terminator());
            if let MergingSucc::False = merging_succ {
                break;
            }

            // We are merging the successor into the produced backend basic
            // block. Record that the successor should be skipped when it is
            // reached.
            //
            // Note: we must not have already generated code for the successor.
            // This is implicitly ensured by the reverse postorder traversal,
            // and the assertion explicitly guarantees that.
            let mut successors = data.terminator().successors();
            let succ = successors.next().unwrap();
            assert!(matches!(self.cached_llbbs[succ], CachedLlbb::None));
            self.cached_llbbs[succ] = CachedLlbb::Skip;
            bb = succ;
        }
    }

    fn codegen_terminator(
        &mut self,
        bx: &mut Bx,
        bb: mir::BasicBlock,
        terminator: &'tcx mir::Terminator<'tcx>,
    ) -> MergingSucc {
        debug!("codegen_terminator: {:?}", terminator);

        let helper = TerminatorCodegenHelper { bb, terminator };

        let mergeable_succ = || {
            // Note: any call to `switch_to_block` will invalidate a `true` value
            // of `mergeable_succ`.
            let mut successors = terminator.successors();
            if let Some(succ) = successors.next()
                && successors.next().is_none()
                && let &[succ_pred] = self.mir.basic_blocks.predecessors()[succ].as_slice()
            {
                // bb has a single successor, and bb is its only predecessor. This
                // makes it a candidate for merging.
                assert_eq!(succ_pred, bb);
                true
            } else {
                false
            }
        };

        self.set_debug_loc(bx, terminator.source_info);
        match terminator.kind {
            mir::TerminatorKind::Resume => {
                self.codegen_resume_terminator(helper, bx);
                MergingSucc::False
            }

            mir::TerminatorKind::Terminate => {
                self.codegen_terminate_terminator(helper, bx, terminator);
                MergingSucc::False
            }

            mir::TerminatorKind::Goto { target } => {
                helper.funclet_br(self, bx, target, mergeable_succ())
            }

            mir::TerminatorKind::SwitchInt { ref discr, ref targets } => {
                self.codegen_switchint_terminator(helper, bx, discr, targets);
                MergingSucc::False
            }

            mir::TerminatorKind::Return => {
                self.codegen_return_terminator(bx);
                MergingSucc::False
            }

            mir::TerminatorKind::Unreachable => {
                bx.unreachable();
                MergingSucc::False
            }

            mir::TerminatorKind::Drop { place, target, unwind, replace: _ } => {
                self.codegen_drop_terminator(helper, bx, place, target, unwind, mergeable_succ())
            }

            mir::TerminatorKind::Assert { ref cond, expected, ref msg, target, unwind } => self
                .codegen_assert_terminator(
                    helper,
                    bx,
                    terminator,
                    cond,
                    expected,
                    msg,
                    target,
                    unwind,
                    mergeable_succ(),
                ),

            mir::TerminatorKind::Call {
                ref func,
                ref args,
                destination,
                target,
                unwind,
                from_hir_call: _,
                fn_span,
            } => self.codegen_call_terminator(
                helper,
                bx,
                terminator,
                func,
                args,
                destination,
                target,
                unwind,
                fn_span,
                mergeable_succ(),
            ),
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
                unwind,
            } => self.codegen_asm_terminator(
                helper,
                bx,
                terminator,
                template,
                operands,
                options,
                line_spans,
                destination,
                unwind,
                self.instance,
                mergeable_succ(),
            ),
        }
    }

    fn codegen_argument(
        &mut self,
        bx: &mut Bx,
        op: OperandRef<'tcx, Bx::Value>,
        llargs: &mut Vec<Bx::Value>,
        arg: &ArgAbi<'tcx, Ty<'tcx>>,
    ) {
        match arg.mode {
            PassMode::Ignore => return,
            PassMode::Cast(_, true) => {
                // Fill padding with undef value, where applicable.
                llargs.push(bx.const_undef(bx.reg_backend_type(&Reg::i32())));
            }
            PassMode::Pair(..) => match op.val {
                Pair(a, b) => {
                    llargs.push(a);
                    llargs.push(b);
                    return;
                }
                _ => bug!("codegen_argument: {:?} invalid for pair argument", op),
            },
            PassMode::Indirect { attrs: _, extra_attrs: Some(_), on_stack: _ } => match op.val {
                Ref(a, Some(b), _) => {
                    llargs.push(a);
                    llargs.push(b);
                    return;
                }
                _ => bug!("codegen_argument: {:?} invalid for unsized indirect argument", op),
            },
            _ => {}
        }

        // Force by-ref if we have to load through a cast pointer.
        let (mut llval, align, by_ref) = match op.val {
            Immediate(_) | Pair(..) => match arg.mode {
                PassMode::Indirect { .. } | PassMode::Cast(..) => {
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
            ZeroSized => match arg.mode {
                PassMode::Indirect { .. } => {
                    // Though `extern "Rust"` doesn't pass ZSTs, some ABIs pass
                    // a pointer for `repr(C)` structs even when empty, so get
                    // one from an `alloca` (which can be left uninitialized).
                    let scratch = PlaceRef::alloca(bx, arg.layout);
                    (scratch.llval, scratch.align, true)
                }
                _ => bug!("ZST {op:?} wasn't ignored, but was passed with abi {arg:?}"),
            },
        };

        if by_ref && !arg.is_indirect() {
            // Have to load the argument, maybe while casting it.
            if let PassMode::Cast(ty, _) = &arg.mode {
                let llty = bx.cast_backend_type(ty);
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
                cx.tcx().mk_tup(&[cx.tcx().mk_mut_ptr(cx.tcx().types.u8), cx.tcx().types.i32]),
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
            let cleanup_bb = Bx::append_block(self.cx, self.llfn, &format!("funclet_{:?}", bb));
            let mut cleanup_bx = Bx::build(self.cx, cleanup_bb);
            let funclet = cleanup_bx.cleanup_pad(None, &[]);
            cleanup_bx.br(llbb);
            self.funclets[bb] = Some(funclet);
            cleanup_bb
        } else {
            let cleanup_llbb = Bx::append_block(self.cx, self.llfn, "cleanup");
            let mut cleanup_bx = Bx::build(self.cx, cleanup_llbb);

            let llpersonality = self.cx.eh_personality();
            let (exn0, exn1) = cleanup_bx.cleanup_landing_pad(llpersonality);

            let slot = self.get_personality_slot(&mut cleanup_bx);
            slot.storage_live(&mut cleanup_bx);
            Pair(exn0, exn1).store(&mut cleanup_bx, slot);

            cleanup_bx.br(llbb);
            cleanup_llbb
        }
    }

    fn unreachable_block(&mut self) -> Bx::BasicBlock {
        self.unreachable_block.unwrap_or_else(|| {
            let llbb = Bx::append_block(self.cx, self.llfn, "unreachable");
            let mut bx = Bx::build(self.cx, llbb);
            bx.unreachable();
            self.unreachable_block = Some(llbb);
            llbb
        })
    }

    fn terminate_block(&mut self) -> Bx::BasicBlock {
        self.terminate_block.unwrap_or_else(|| {
            let funclet;
            let llbb;
            let mut bx;
            if base::wants_msvc_seh(self.cx.sess()) {
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
                llbb = Bx::append_block(self.cx, self.llfn, "cs_terminate");
                let cp_llbb = Bx::append_block(self.cx, self.llfn, "cp_terminate");

                let mut cs_bx = Bx::build(self.cx, llbb);
                let cs = cs_bx.catch_switch(None, None, &[cp_llbb]);

                // The "null" here is actually a RTTI type descriptor for the
                // C++ personality function, but `catch (...)` has no type so
                // it's null. The 64 here is actually a bitfield which
                // represents that this is a catch-all block.
                bx = Bx::build(self.cx, cp_llbb);
                let null =
                    bx.const_null(bx.type_i8p_ext(bx.cx().data_layout().instruction_address_space));
                let sixty_four = bx.const_i32(64);
                funclet = Some(bx.catch_pad(cs, &[null, sixty_four, null]));
            } else {
                llbb = Bx::append_block(self.cx, self.llfn, "terminate");
                bx = Bx::build(self.cx, llbb);

                let llpersonality = self.cx.eh_personality();
                bx.filter_landing_pad(llpersonality);

                funclet = None;
            }

            self.set_debug_loc(&mut bx, mir::SourceInfo::outermost(self.mir.span));

            let (fn_abi, fn_ptr) = common::build_langcall(&bx, None, LangItem::PanicCannotUnwind);
            let fn_ty = bx.fn_decl_backend_type(&fn_abi);

            let llret = bx.call(fn_ty, None, Some(&fn_abi), fn_ptr, &[], funclet.as_ref());
            bx.do_not_inline(llret);

            bx.unreachable();

            self.terminate_block = Some(llbb);
            llbb
        })
    }

    /// Get the backend `BasicBlock` for a MIR `BasicBlock`, either already
    /// cached in `self.cached_llbbs`, or created on demand (and cached).
    // FIXME(eddyb) rename `llbb` and other `ll`-prefixed things to use a
    // more backend-agnostic prefix such as `cg` (i.e. this would be `cgbb`).
    pub fn llbb(&mut self, bb: mir::BasicBlock) -> Bx::BasicBlock {
        self.try_llbb(bb).unwrap()
    }

    /// Like `llbb`, but may fail if the basic block should be skipped.
    pub fn try_llbb(&mut self, bb: mir::BasicBlock) -> Option<Bx::BasicBlock> {
        match self.cached_llbbs[bb] {
            CachedLlbb::None => {
                // FIXME(eddyb) only name the block if `fewer_names` is `false`.
                let llbb = Bx::append_block(self.cx, self.llfn, &format!("{:?}", bb));
                self.cached_llbbs[bb] = CachedLlbb::Some(llbb);
                Some(llbb)
            }
            CachedLlbb::Some(llbb) => Some(llbb),
            CachedLlbb::Skip => None,
        }
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
                LocalRef::PendingOperand => {
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
                LocalRef::Operand(_) => {
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
                self.locals[index] = LocalRef::Operand(op);
                self.debug_introduce_local(bx, index);
            }
            DirectOperand(index) => {
                // If there is a cast, we have to store and reload.
                let op = if let PassMode::Cast(..) = ret_abi.mode {
                    let tmp = PlaceRef::alloca(bx, ret_abi.layout);
                    tmp.storage_live(bx);
                    bx.store_arg(&ret_abi, llval, tmp);
                    let op = bx.load_operand(tmp);
                    tmp.storage_dead(bx);
                    op
                } else {
                    OperandRef::from_immediate_or_packed_pair(bx, llval, ret_abi.layout)
                };
                self.locals[index] = LocalRef::Operand(op);
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
