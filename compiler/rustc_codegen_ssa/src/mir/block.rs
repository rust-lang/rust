use std::cmp;

use rustc_abi::{Align, BackendRepr, ExternAbi, HasDataLayout, Reg, Size, WrappingRange};
use rustc_ast as ast;
use rustc_ast::{InlineAsmOptions, InlineAsmTemplatePiece};
use rustc_data_structures::packed::Pu128;
use rustc_hir::lang_items::LangItem;
use rustc_middle::mir::{self, AssertKind, InlineAsmMacro, SwitchTargets, UnwindTerminateReason};
use rustc_middle::ty::layout::{HasTyCtxt, LayoutOf, ValidityRequirement};
use rustc_middle::ty::print::{with_no_trimmed_paths, with_no_visible_paths};
use rustc_middle::ty::{self, Instance, Ty};
use rustc_middle::{bug, span_bug};
use rustc_session::config::OptLevel;
use rustc_span::Span;
use rustc_span::source_map::Spanned;
use rustc_target::callconv::{ArgAbi, CastTarget, FnAbi, PassMode};
use tracing::{debug, info};

use super::operand::OperandRef;
use super::operand::OperandValue::{Immediate, Pair, Ref, ZeroSized};
use super::place::{PlaceRef, PlaceValue};
use super::{CachedLlbb, FunctionCx, LocalRef};
use crate::base::{self, is_call_from_compiler_builtins_to_upstream_monomorphization};
use crate::common::{self, IntPredicate};
use crate::errors::CompilerBuiltinsCannotCall;
use crate::traits::*;
use crate::{MemFlags, meth};

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
        let cleanup_kinds = fx.cleanup_kinds.as_ref()?;
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
            // Cross-funclet jump - need a trampoline
            assert!(base::wants_new_eh_instructions(fx.cx.tcx().sess));
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
        lifetime_ends_after_call: &[(Bx::Value, Size)],
        instance: Option<Instance<'tcx>>,
        mergeable_succ: bool,
    ) -> MergingSucc {
        let tcx = bx.tcx();
        if let Some(instance) = instance
            && is_call_from_compiler_builtins_to_upstream_monomorphization(tcx, instance)
        {
            if destination.is_some() {
                let caller_def = fx.instance.def_id();
                let e = CompilerBuiltinsCannotCall {
                    span: tcx.def_span(caller_def),
                    caller: with_no_trimmed_paths!(tcx.def_path_str(caller_def)),
                    callee: with_no_trimmed_paths!(tcx.def_path_str(instance.def_id())),
                };
                tcx.dcx().emit_err(e);
            } else {
                info!(
                    "compiler_builtins call to diverging function {:?} replaced with abort",
                    instance.def_id()
                );
                bx.abort();
                bx.unreachable();
                return MergingSucc::False;
            }
        }

        // If there is a cleanup block and the function we're calling can unwind, then
        // do an invoke, otherwise do a call.
        let fn_ty = bx.fn_decl_backend_type(fn_abi);

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
            mir::UnwindAction::Terminate(reason) => {
                if fx.mir[self.bb].is_cleanup && base::wants_new_eh_instructions(fx.cx.tcx().sess) {
                    // MSVC SEH will abort automatically if an exception tries to
                    // propagate out from cleanup.

                    // FIXME(@mirkootter): For wasm, we currently do not support terminate during
                    // cleanup, because this requires a few more changes: The current code
                    // caches the `terminate_block` for each function; funclet based code - however -
                    // requires a different terminate_block for each funclet
                    // Until this is implemented, we just do not unwind inside cleanup blocks

                    None
                } else {
                    Some(fx.terminate_block(reason))
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
                Some(fn_abi),
                fn_ptr,
                llargs,
                ret_llbb,
                unwind_block,
                self.funclet(fx),
                instance,
            );
            if fx.mir[self.bb].is_cleanup {
                bx.apply_attrs_to_cleanup_callsite(invokeret);
            }

            if let Some((ret_dest, target)) = destination {
                bx.switch_to_block(fx.llbb(target));
                fx.set_debug_loc(bx, self.terminator.source_info);
                for &(tmp, size) in lifetime_ends_after_call {
                    bx.lifetime_end(tmp, size);
                }
                fx.store_return(bx, ret_dest, &fn_abi.ret, invokeret);
            }
            MergingSucc::False
        } else {
            let llret =
                bx.call(fn_ty, fn_attrs, Some(fn_abi), fn_ptr, llargs, self.funclet(fx), instance);
            if fx.mir[self.bb].is_cleanup {
                bx.apply_attrs_to_cleanup_callsite(llret);
            }

            if let Some((ret_dest, target)) = destination {
                for &(tmp, size) in lifetime_ends_after_call {
                    bx.lifetime_end(tmp, size);
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
            mir::UnwindAction::Terminate(reason) => Some(fx.terminate_block(reason)),
            mir::UnwindAction::Continue => None,
            mir::UnwindAction::Unreachable => None,
        };

        if operands.iter().any(|x| matches!(x, InlineAsmOperandRef::Label { .. })) {
            assert!(unwind_target.is_none());
            let ret_llbb = if let Some(target) = destination {
                fx.llbb(target)
            } else {
                fx.unreachable_block()
            };

            bx.codegen_inline_asm(
                template,
                operands,
                options,
                line_spans,
                instance,
                Some(ret_llbb),
                None,
            );
            MergingSucc::False
        } else if let Some(cleanup) = unwind_target {
            let ret_llbb = if let Some(target) = destination {
                fx.llbb(target)
            } else {
                fx.unreachable_block()
            };

            bx.codegen_inline_asm(
                template,
                operands,
                options,
                line_spans,
                instance,
                Some(ret_llbb),
                Some((cleanup, self.funclet(fx))),
            );
            MergingSucc::False
        } else {
            bx.codegen_inline_asm(template, operands, options, line_spans, instance, None, None);

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
        let discr = self.codegen_operand(bx, discr);
        let discr_value = discr.immediate();
        let switch_ty = discr.layout.ty;
        // If our discriminant is a constant we can branch directly
        if let Some(const_discr) = bx.const_to_opt_u128(discr_value, false) {
            let target = targets.target_for_value(const_discr);
            bx.br(helper.llbb_with_cleanup(self, target));
            return;
        };

        let mut target_iter = targets.iter();
        if target_iter.len() == 1 {
            // If there are two targets (one conditional, one fallback), emit `br` instead of
            // `switch`.
            let (test_value, target) = target_iter.next().unwrap();
            let otherwise = targets.otherwise();
            let lltarget = helper.llbb_with_cleanup(self, target);
            let llotherwise = helper.llbb_with_cleanup(self, otherwise);
            let target_cold = self.cold_blocks[target];
            let otherwise_cold = self.cold_blocks[otherwise];
            // If `target_cold == otherwise_cold`, the branches have the same weight
            // so there is no expectation. If they differ, the `target` branch is expected
            // when the `otherwise` branch is cold.
            let expect = if target_cold == otherwise_cold { None } else { Some(otherwise_cold) };
            if switch_ty == bx.tcx().types.bool {
                // Don't generate trivial icmps when switching on bool.
                match test_value {
                    0 => {
                        let expect = expect.map(|e| !e);
                        bx.cond_br_with_expect(discr_value, llotherwise, lltarget, expect);
                    }
                    1 => {
                        bx.cond_br_with_expect(discr_value, lltarget, llotherwise, expect);
                    }
                    _ => bug!(),
                }
            } else {
                let switch_llty = bx.immediate_backend_type(bx.layout_of(switch_ty));
                let llval = bx.const_uint_big(switch_llty, test_value);
                let cmp = bx.icmp(IntPredicate::IntEQ, discr_value, llval);
                bx.cond_br_with_expect(cmp, lltarget, llotherwise, expect);
            }
        } else if target_iter.len() == 2
            && self.mir[targets.otherwise()].is_empty_unreachable()
            && targets.all_values().contains(&Pu128(0))
            && targets.all_values().contains(&Pu128(1))
        {
            // This is the really common case for `bool`, `Option`, etc.
            // By using `trunc nuw` we communicate that other values are
            // impossible without needing `switch` or `assume`s.
            let true_bb = targets.target_for_value(1);
            let false_bb = targets.target_for_value(0);
            let true_ll = helper.llbb_with_cleanup(self, true_bb);
            let false_ll = helper.llbb_with_cleanup(self, false_bb);

            let expected_cond_value = if self.cx.sess().opts.optimize == OptLevel::No {
                None
            } else {
                match (self.cold_blocks[true_bb], self.cold_blocks[false_bb]) {
                    // Same coldness, no expectation
                    (true, true) | (false, false) => None,
                    // Different coldness, expect the non-cold one
                    (true, false) => Some(false),
                    (false, true) => Some(true),
                }
            };

            let bool_ty = bx.tcx().types.bool;
            let cond = if switch_ty == bool_ty {
                discr_value
            } else {
                let bool_llty = bx.immediate_backend_type(bx.layout_of(bool_ty));
                bx.unchecked_utrunc(discr_value, bool_llty)
            };
            bx.cond_br_with_expect(cond, true_ll, false_ll, expected_cond_value);
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
            //   must fall back to the slower SelectionDAG isel. Therefore, using `br` gives
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
            let cmp = bx.icmp(IntPredicate::IntEQ, discr_value, llval);
            bx.cond_br(cmp, ll1, ll2);
        } else {
            let otherwise = targets.otherwise();
            let otherwise_cold = self.cold_blocks[otherwise];
            let otherwise_unreachable = self.mir[otherwise].is_empty_unreachable();
            let cold_count = targets.iter().filter(|(_, target)| self.cold_blocks[*target]).count();
            let none_cold = cold_count == 0;
            let all_cold = cold_count == targets.iter().len();
            if (none_cold && (!otherwise_cold || otherwise_unreachable))
                || (all_cold && (otherwise_cold || otherwise_unreachable))
            {
                // All targets have the same weight,
                // or `otherwise` is unreachable and it's the only target with a different weight.
                bx.switch(
                    discr_value,
                    helper.llbb_with_cleanup(self, targets.otherwise()),
                    target_iter
                        .map(|(value, target)| (value, helper.llbb_with_cleanup(self, target))),
                );
            } else {
                // Targets have different weights
                bx.switch_with_weights(
                    discr_value,
                    helper.llbb_with_cleanup(self, targets.otherwise()),
                    otherwise_cold,
                    target_iter.map(|(value, target)| {
                        (value, helper.llbb_with_cleanup(self, target), self.cold_blocks[target])
                    }),
                );
            }
        }
    }

    fn codegen_return_terminator(&mut self, bx: &mut Bx) {
        // Call `va_end` if this is the definition of a C-variadic function.
        if self.fn_abi.c_variadic {
            // The `VaList` "spoofed" argument is just after all the real arguments.
            let va_list_arg_idx = self.fn_abi.args.len();
            match self.locals[mir::Local::from_usize(1 + va_list_arg_idx)] {
                LocalRef::Place(va_list) => {
                    bx.va_end(va_list.val.llval);
                }
                _ => bug!("C-variadic function must have a `VaList` place"),
            }
        }
        if self.fn_abi.ret.layout.is_uninhabited() {
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
                if let Ref(place_val) = op.val {
                    bx.load_from_place(bx.backend_type(op.layout), place_val)
                } else {
                    op.immediate_or_packed_pair(bx)
                }
            }

            PassMode::Cast { cast: cast_ty, pad_i32: _ } => {
                let op = match self.locals[mir::RETURN_PLACE] {
                    LocalRef::Operand(op) => op,
                    LocalRef::PendingOperand => bug!("use of return before def"),
                    LocalRef::Place(cg_place) => {
                        OperandRef { val: Ref(cg_place.val), layout: cg_place.layout }
                    }
                    LocalRef::UnsizedPlace(_) => bug!("return type must be sized"),
                };
                let llslot = match op.val {
                    Immediate(_) | Pair(..) => {
                        let scratch = PlaceRef::alloca(bx, self.fn_abi.ret.layout);
                        op.val.store(bx, scratch);
                        scratch.val.llval
                    }
                    Ref(place_val) => {
                        assert_eq!(
                            place_val.align, op.layout.align.abi,
                            "return place is unaligned!"
                        );
                        place_val.llval
                    }
                    ZeroSized => bug!("ZST return value shouldn't be in PassMode::Cast"),
                };
                load_cast(bx, cast_ty, llslot, self.fn_abi.ret.layout.align.abi)
            }
        };
        bx.ret(llval);
    }

    #[tracing::instrument(level = "trace", skip(self, helper, bx))]
    fn codegen_drop_terminator(
        &mut self,
        helper: TerminatorCodegenHelper<'tcx>,
        bx: &mut Bx,
        source_info: &mir::SourceInfo,
        location: mir::Place<'tcx>,
        target: mir::BasicBlock,
        unwind: mir::UnwindAction,
        mergeable_succ: bool,
    ) -> MergingSucc {
        let ty = location.ty(self.mir, bx.tcx()).ty;
        let ty = self.monomorphize(ty);
        let drop_fn = Instance::resolve_drop_in_place(bx.tcx(), ty);

        if let ty::InstanceKind::DropGlue(_, None) = drop_fn.def {
            // we don't actually need to drop anything.
            return helper.funclet_br(self, bx, target, mergeable_succ);
        }

        let place = self.codegen_place(bx, location.as_ref());
        let (args1, args2);
        let mut args = if let Some(llextra) = place.val.llextra {
            args2 = [place.val.llval, llextra];
            &args2[..]
        } else {
            args1 = [place.val.llval];
            &args1[..]
        };
        let (maybe_null, drop_fn, fn_abi, drop_instance) = match ty.kind() {
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
                    def: ty::InstanceKind::Virtual(drop_fn.def_id(), 0), // idx 0: the drop function
                    args: drop_fn.args,
                };
                debug!("ty = {:?}", ty);
                debug!("drop_fn = {:?}", drop_fn);
                debug!("args = {:?}", args);
                let fn_abi = bx.fn_abi_of_instance(virtual_drop, ty::List::empty());
                let vtable = args[1];
                // Truncate vtable off of args list
                args = &args[..1];
                (
                    true,
                    meth::VirtualIndex::from_index(ty::COMMON_VTABLE_ENTRIES_DROPINPLACE)
                        .get_optional_fn(bx, vtable, ty, fn_abi),
                    fn_abi,
                    virtual_drop,
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
                    def: ty::InstanceKind::Virtual(drop_fn.def_id(), 0), // idx 0: the drop function
                    args: drop_fn.args,
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
                    true,
                    meth::VirtualIndex::from_index(ty::COMMON_VTABLE_ENTRIES_DROPINPLACE)
                        .get_optional_fn(bx, meta.immediate(), ty, fn_abi),
                    fn_abi,
                    virtual_drop,
                )
            }
            _ => (
                false,
                bx.get_fn_addr(drop_fn),
                bx.fn_abi_of_instance(drop_fn, ty::List::empty()),
                drop_fn,
            ),
        };

        // We generate a null check for the drop_fn. This saves a bunch of relocations being
        // generated for no-op drops.
        if maybe_null {
            let is_not_null = bx.append_sibling_block("is_not_null");
            let llty = bx.fn_ptr_backend_type(fn_abi);
            let null = bx.const_null(llty);
            let non_null =
                bx.icmp(base::bin_op_to_icmp_predicate(mir::BinOp::Ne, false), drop_fn, null);
            bx.cond_br(non_null, is_not_null, helper.llbb_with_cleanup(self, target));
            bx.switch_to_block(is_not_null);
            self.set_debug_loc(bx, *source_info);
        }

        helper.do_call(
            self,
            bx,
            fn_abi,
            drop_fn,
            args,
            Some((ReturnDest::Nothing, target)),
            unwind,
            &[],
            Some(drop_instance),
            !maybe_null && mergeable_succ,
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
        if !bx.sess().overflow_checks() && msg.is_optional_overflow_check() {
            const_cond = Some(expected);
        }

        // Don't codegen the panic block if success if known.
        if const_cond == Some(expected) {
            return helper.funclet_br(self, bx, target, mergeable_succ);
        }

        // Because we're branching to a panic block (either a `#[cold]` one
        // or an inlined abort), there's no need to `expect` it.

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
            AssertKind::BoundsCheck { len, index } => {
                let len = self.codegen_operand(bx, len).immediate();
                let index = self.codegen_operand(bx, index).immediate();
                // It's `fn panic_bounds_check(index: usize, len: usize)`,
                // and `#[track_caller]` adds an implicit third argument.
                (LangItem::PanicBoundsCheck, vec![index, len, location])
            }
            AssertKind::MisalignedPointerDereference { required, found } => {
                let required = self.codegen_operand(bx, required).immediate();
                let found = self.codegen_operand(bx, found).immediate();
                // It's `fn panic_misaligned_pointer_dereference(required: usize, found: usize)`,
                // and `#[track_caller]` adds an implicit third argument.
                (LangItem::PanicMisalignedPointerDereference, vec![required, found, location])
            }
            AssertKind::NullPointerDereference => {
                // It's `fn panic_null_pointer_dereference()`,
                // `#[track_caller]` adds an implicit argument.
                (LangItem::PanicNullPointerDereference, vec![location])
            }
            _ => {
                // It's `pub fn panic_...()` and `#[track_caller]` adds an implicit argument.
                (msg.panic_function(), vec![location])
            }
        };

        let (fn_abi, llfn, instance) = common::build_langcall(bx, span, lang_item);

        // Codegen the actual panic invoke/call.
        let merging_succ =
            helper.do_call(self, bx, fn_abi, llfn, &args, None, unwind, &[], Some(instance), false);
        assert_eq!(merging_succ, MergingSucc::False);
        MergingSucc::False
    }

    fn codegen_terminate_terminator(
        &mut self,
        helper: TerminatorCodegenHelper<'tcx>,
        bx: &mut Bx,
        terminator: &mir::Terminator<'tcx>,
        reason: UnwindTerminateReason,
    ) {
        let span = terminator.source_info.span;
        self.set_debug_loc(bx, terminator.source_info);

        // Obtain the panic entry point.
        let (fn_abi, llfn, instance) = common::build_langcall(bx, span, reason.lang_item());

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
            Some(instance),
            false,
        );
        assert_eq!(merging_succ, MergingSucc::False);
    }

    /// Returns `Some` if this is indeed a panic intrinsic and codegen is done.
    fn codegen_panic_intrinsic(
        &mut self,
        helper: &TerminatorCodegenHelper<'tcx>,
        bx: &mut Bx,
        intrinsic: ty::IntrinsicDef,
        instance: Instance<'tcx>,
        source_info: mir::SourceInfo,
        target: Option<mir::BasicBlock>,
        unwind: mir::UnwindAction,
        mergeable_succ: bool,
    ) -> Option<MergingSucc> {
        // Emit a panic or a no-op for `assert_*` intrinsics.
        // These are intrinsics that compile to panics so that we can get a message
        // which mentions the offending type, even from a const context.
        let Some(requirement) = ValidityRequirement::from_intrinsic(intrinsic.name) else {
            return None;
        };

        let ty = instance.args.type_at(0);

        let is_valid = bx
            .tcx()
            .check_validity_requirement((requirement, bx.typing_env().as_query_input(ty)))
            .expect("expect to have layout during codegen");

        if is_valid {
            // a NOP
            let target = target.unwrap();
            return Some(helper.funclet_br(self, bx, target, mergeable_succ));
        }

        let layout = bx.layout_of(ty);

        let msg_str = with_no_visible_paths!({
            with_no_trimmed_paths!({
                if layout.is_uninhabited() {
                    // Use this error even for the other intrinsics as it is more precise.
                    format!("attempted to instantiate uninhabited type `{ty}`")
                } else if requirement == ValidityRequirement::Zero {
                    format!("attempted to zero-initialize type `{ty}`, which is invalid")
                } else {
                    format!("attempted to leave type `{ty}` uninitialized, which is invalid")
                }
            })
        });
        let msg = bx.const_str(&msg_str);

        // Obtain the panic entry point.
        let (fn_abi, llfn, instance) =
            common::build_langcall(bx, source_info.span, LangItem::PanicNounwind);

        // Codegen the actual panic invoke/call.
        Some(helper.do_call(
            self,
            bx,
            fn_abi,
            llfn,
            &[msg.0, msg.1],
            target.as_ref().map(|bb| (ReturnDest::Nothing, *bb)),
            unwind,
            &[],
            Some(instance),
            mergeable_succ,
        ))
    }

    fn codegen_call_terminator(
        &mut self,
        helper: TerminatorCodegenHelper<'tcx>,
        bx: &mut Bx,
        terminator: &mir::Terminator<'tcx>,
        func: &mir::Operand<'tcx>,
        args: &[Spanned<mir::Operand<'tcx>>],
        destination: mir::Place<'tcx>,
        target: Option<mir::BasicBlock>,
        unwind: mir::UnwindAction,
        fn_span: Span,
        mergeable_succ: bool,
    ) -> MergingSucc {
        let source_info = mir::SourceInfo { span: fn_span, ..terminator.source_info };

        // Create the callee. This is a fn ptr or zero-sized and hence a kind of scalar.
        let callee = self.codegen_operand(bx, func);

        let (instance, mut llfn) = match *callee.layout.ty.kind() {
            ty::FnDef(def_id, generic_args) => {
                let instance = ty::Instance::expect_resolve(
                    bx.tcx(),
                    bx.typing_env(),
                    def_id,
                    generic_args,
                    fn_span,
                );

                let instance = match instance.def {
                    // We don't need AsyncDropGlueCtorShim here because it is not `noop func`,
                    // it is `func returning noop future`
                    ty::InstanceKind::DropGlue(_, None) => {
                        // Empty drop glue; a no-op.
                        let target = target.unwrap();
                        return helper.funclet_br(self, bx, target, mergeable_succ);
                    }
                    ty::InstanceKind::Intrinsic(def_id) => {
                        let intrinsic = bx.tcx().intrinsic(def_id).unwrap();
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

                        let result_layout =
                            self.cx.layout_of(self.monomorphized_place_ty(destination.as_ref()));

                        let (result, store_in_local) = if result_layout.is_zst() {
                            (
                                PlaceRef::new_sized(bx.const_undef(bx.type_ptr()), result_layout),
                                None,
                            )
                        } else if let Some(local) = destination.as_local() {
                            match self.locals[local] {
                                LocalRef::Place(dest) => (dest, None),
                                LocalRef::UnsizedPlace(_) => bug!("return type must be sized"),
                                LocalRef::PendingOperand => {
                                    // Currently, intrinsics always need a location to store
                                    // the result, so we create a temporary `alloca` for the
                                    // result.
                                    let tmp = PlaceRef::alloca(bx, result_layout);
                                    tmp.storage_live(bx);
                                    (tmp, Some(local))
                                }
                                LocalRef::Operand(_) => {
                                    bug!("place local already assigned to");
                                }
                            }
                        } else {
                            (self.codegen_place(bx, destination.as_ref()), None)
                        };

                        if result.val.align < result.layout.align.abi {
                            // Currently, MIR code generation does not create calls
                            // that store directly to fields of packed structs (in
                            // fact, the calls it creates write only to temps).
                            //
                            // If someone changes that, please update this code path
                            // to create a temporary.
                            span_bug!(self.mir.span, "can't directly store to unaligned value");
                        }

                        let args: Vec<_> =
                            args.iter().map(|arg| self.codegen_operand(bx, &arg.node)).collect();

                        match self.codegen_intrinsic_call(bx, instance, &args, result, source_info)
                        {
                            Ok(()) => {
                                if let Some(local) = store_in_local {
                                    let op = bx.load_operand(result);
                                    result.storage_dead(bx);
                                    self.overwrite_local(local, LocalRef::Operand(op));
                                    self.debug_introduce_local(bx, local);
                                }

                                return if let Some(target) = target {
                                    helper.funclet_br(self, bx, target, mergeable_succ)
                                } else {
                                    bx.unreachable();
                                    MergingSucc::False
                                };
                            }
                            Err(instance) => {
                                if intrinsic.must_be_overridden {
                                    span_bug!(
                                        fn_span,
                                        "intrinsic {} must be overridden by codegen backend, but isn't",
                                        intrinsic.name,
                                    );
                                }
                                instance
                            }
                        }
                    }
                    _ => instance,
                };

                (Some(instance), None)
            }
            ty::FnPtr(..) => (None, Some(callee.immediate())),
            _ => bug!("{} is not callable", callee.layout.ty),
        };

        // FIXME(eddyb) avoid computing this if possible, when `instance` is
        // available - right now `sig` is only needed for getting the `abi`
        // and figuring out how many extra args were passed to a C-variadic `fn`.
        let sig = callee.layout.ty.fn_sig(bx.tcx());

        let extra_args = &args[sig.inputs().skip_binder().len()..];
        let extra_args = bx.tcx().mk_type_list_from_iter(extra_args.iter().map(|op_arg| {
            let op_ty = op_arg.node.ty(self.mir, bx.tcx());
            self.monomorphize(op_ty)
        }));

        let fn_abi = match instance {
            Some(instance) => bx.fn_abi_of_instance(instance, extra_args),
            None => bx.fn_abi_of_fn_ptr(sig, extra_args),
        };

        // The arguments we'll be passing. Plus one to account for outptr, if used.
        let arg_count = fn_abi.args.len() + fn_abi.ret.is_indirect() as usize;

        let mut llargs = Vec::with_capacity(arg_count);

        // We still need to call `make_return_dest` even if there's no `target`, since
        // `fn_abi.ret` could be `PassMode::Indirect`, even if it is uninhabited,
        // and `make_return_dest` adds the return-place indirect pointer to `llargs`.
        let return_dest = self.make_return_dest(bx, destination, &fn_abi.ret, &mut llargs);
        let destination = target.map(|target| (return_dest, target));

        // Split the rust-call tupled arguments off.
        let (first_args, untuple) = if sig.abi() == ExternAbi::RustCall
            && let Some((tup, args)) = args.split_last()
        {
            (args, Some(tup))
        } else {
            (args, None)
        };

        // When generating arguments we sometimes introduce temporary allocations with lifetime
        // that extend for the duration of a call. Keep track of those allocations and their sizes
        // to generate `lifetime_end` when the call returns.
        let mut lifetime_ends_after_call: Vec<(Bx::Value, Size)> = Vec::new();
        'make_args: for (i, arg) in first_args.iter().enumerate() {
            let mut op = self.codegen_operand(bx, &arg.node);

            if let (0, Some(ty::InstanceKind::Virtual(_, idx))) = (i, instance.map(|i| i.def)) {
                match op.val {
                    Pair(data_ptr, meta) => {
                        // In the case of Rc<Self>, we need to explicitly pass a
                        // *mut RcInner<Self> with a Scalar (not ScalarPair) ABI. This is a hack
                        // that is understood elsewhere in the compiler as a method on
                        // `dyn Trait`.
                        // To get a `*mut RcInner<Self>`, we just keep unwrapping newtypes until
                        // we get a value of a built-in pointer type.
                        //
                        // This is also relevant for `Pin<&mut Self>`, where we need to peel the
                        // `Pin`.
                        while !op.layout.ty.is_raw_ptr() && !op.layout.ty.is_ref() {
                            let (idx, _) = op.layout.non_1zst_field(bx).expect(
                                "not exactly one non-1-ZST field in a `DispatchFromDyn` type",
                            );
                            op = op.extract_field(self, bx, idx.as_usize());
                        }

                        // Now that we have `*dyn Trait` or `&dyn Trait`, split it up into its
                        // data pointer and vtable. Look up the method in the vtable, and pass
                        // the data pointer as the first argument.
                        llfn = Some(meth::VirtualIndex::from_index(idx).get_fn(
                            bx,
                            meta,
                            op.layout.ty,
                            fn_abi,
                        ));
                        llargs.push(data_ptr);
                        continue 'make_args;
                    }
                    Ref(PlaceValue { llval: data_ptr, llextra: Some(meta), .. }) => {
                        // by-value dynamic dispatch
                        llfn = Some(meth::VirtualIndex::from_index(idx).get_fn(
                            bx,
                            meta,
                            op.layout.ty,
                            fn_abi,
                        ));
                        llargs.push(data_ptr);
                        continue;
                    }
                    Immediate(_) => {
                        // See comment above explaining why we peel these newtypes
                        while !op.layout.ty.is_raw_ptr() && !op.layout.ty.is_ref() {
                            let (idx, _) = op.layout.non_1zst_field(bx).expect(
                                "not exactly one non-1-ZST field in a `DispatchFromDyn` type",
                            );
                            op = op.extract_field(self, bx, idx.as_usize());
                        }

                        // Make sure that we've actually unwrapped the rcvr down
                        // to a pointer or ref to `dyn* Trait`.
                        if !op.layout.ty.builtin_deref(true).unwrap().is_dyn_star() {
                            span_bug!(fn_span, "can't codegen a virtual call on {:#?}", op);
                        }
                        let place = op.deref(bx.cx());
                        let data_place = place.project_field(bx, 0);
                        let meta_place = place.project_field(bx, 1);
                        let meta = bx.load_operand(meta_place);
                        llfn = Some(meth::VirtualIndex::from_index(idx).get_fn(
                            bx,
                            meta.immediate(),
                            op.layout.ty,
                            fn_abi,
                        ));
                        llargs.push(data_place.val.llval);
                        continue;
                    }
                    _ => {
                        span_bug!(fn_span, "can't codegen a virtual call on {:#?}", op);
                    }
                }
            }

            // The callee needs to own the argument memory if we pass it
            // by-ref, so make a local copy of non-immediate constants.
            match (&arg.node, op.val) {
                (&mir::Operand::Copy(_), Ref(PlaceValue { llextra: None, .. }))
                | (&mir::Operand::Constant(_), Ref(PlaceValue { llextra: None, .. })) => {
                    let tmp = PlaceRef::alloca(bx, op.layout);
                    bx.lifetime_start(tmp.val.llval, tmp.layout.size);
                    op.val.store(bx, tmp);
                    op.val = Ref(tmp.val);
                    lifetime_ends_after_call.push((tmp.val.llval, tmp.layout.size));
                }
                _ => {}
            }

            self.codegen_argument(
                bx,
                op,
                &mut llargs,
                &fn_abi.args[i],
                &mut lifetime_ends_after_call,
            );
        }
        let num_untupled = untuple.map(|tup| {
            self.codegen_arguments_untupled(
                bx,
                &tup.node,
                &mut llargs,
                &fn_abi.args[first_args.len()..],
                &mut lifetime_ends_after_call,
            )
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
                "#[track_caller] fn's must have 1 more argument in their ABI than in their MIR: {instance:?} {fn_span:?} {fn_abi:?}",
            );
            let location = self.get_caller_location(bx, source_info);
            debug!(
                "codegen_call_terminator({:?}): location={:?} (fn_span {:?})",
                terminator, location, fn_span
            );

            let last_arg = fn_abi.args.last().unwrap();
            self.codegen_argument(
                bx,
                location,
                &mut llargs,
                last_arg,
                &mut lifetime_ends_after_call,
            );
        }

        let fn_ptr = match (instance, llfn) {
            (Some(instance), None) => bx.get_fn_addr(instance),
            (_, Some(llfn)) => llfn,
            _ => span_bug!(fn_span, "no instance or llfn for call"),
        };
        self.set_debug_loc(bx, source_info);
        helper.do_call(
            self,
            bx,
            fn_abi,
            fn_ptr,
            &llargs,
            destination,
            unwind,
            &lifetime_ends_after_call,
            instance,
            mergeable_succ,
        )
    }

    fn codegen_asm_terminator(
        &mut self,
        helper: TerminatorCodegenHelper<'tcx>,
        bx: &mut Bx,
        asm_macro: InlineAsmMacro,
        terminator: &mir::Terminator<'tcx>,
        template: &[ast::InlineAsmTemplatePiece],
        operands: &[mir::InlineAsmOperand<'tcx>],
        options: ast::InlineAsmOptions,
        line_spans: &[Span],
        targets: &[mir::BasicBlock],
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
                    let const_value = self.eval_mir_constant(value);
                    let string = common::asm_const_to_str(
                        bx.tcx(),
                        span,
                        const_value,
                        bx.layout_of(value.ty()),
                    );
                    InlineAsmOperandRef::Const { string }
                }
                mir::InlineAsmOperand::SymFn { ref value } => {
                    let const_ = self.monomorphize(value.const_);
                    if let ty::FnDef(def_id, args) = *const_.ty().kind() {
                        let instance = ty::Instance::resolve_for_fn_ptr(
                            bx.tcx(),
                            bx.typing_env(),
                            def_id,
                            args,
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
                mir::InlineAsmOperand::Label { target_index } => {
                    InlineAsmOperandRef::Label { label: self.llbb(targets[target_index]) }
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
            if asm_macro.diverges(options) { None } else { targets.get(0).copied() },
            unwind,
            instance,
            mergeable_succ,
        )
    }

    pub(crate) fn codegen_block(&mut self, mut bb: mir::BasicBlock) {
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

    pub(crate) fn codegen_block_as_unreachable(&mut self, bb: mir::BasicBlock) {
        let llbb = match self.try_llbb(bb) {
            Some(llbb) => llbb,
            None => return,
        };
        let bx = &mut Bx::build(self.cx, llbb);
        debug!("codegen_block_as_unreachable({:?})", bb);
        bx.unreachable();
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
            mir::TerminatorKind::UnwindResume => {
                self.codegen_resume_terminator(helper, bx);
                MergingSucc::False
            }

            mir::TerminatorKind::UnwindTerminate(reason) => {
                self.codegen_terminate_terminator(helper, bx, terminator, reason);
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

            mir::TerminatorKind::Drop { place, target, unwind, replace: _, drop, async_fut } => {
                assert!(
                    async_fut.is_none() && drop.is_none(),
                    "Async Drop must be expanded or reset to sync before codegen"
                );
                self.codegen_drop_terminator(
                    helper,
                    bx,
                    &terminator.source_info,
                    place,
                    target,
                    unwind,
                    mergeable_succ(),
                )
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
                call_source: _,
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
            mir::TerminatorKind::TailCall { .. } => {
                // FIXME(explicit_tail_calls): implement tail calls in ssa backend
                span_bug!(
                    terminator.source_info.span,
                    "`TailCall` terminator is not yet supported by `rustc_codegen_ssa`"
                )
            }
            mir::TerminatorKind::CoroutineDrop | mir::TerminatorKind::Yield { .. } => {
                bug!("coroutine ops in codegen")
            }
            mir::TerminatorKind::FalseEdge { .. } | mir::TerminatorKind::FalseUnwind { .. } => {
                bug!("borrowck false edges in codegen")
            }

            mir::TerminatorKind::InlineAsm {
                asm_macro,
                template,
                ref operands,
                options,
                line_spans,
                ref targets,
                unwind,
            } => self.codegen_asm_terminator(
                helper,
                bx,
                asm_macro,
                terminator,
                template,
                operands,
                options,
                line_spans,
                targets,
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
        lifetime_ends_after_call: &mut Vec<(Bx::Value, Size)>,
    ) {
        match arg.mode {
            PassMode::Ignore => return,
            PassMode::Cast { pad_i32: true, .. } => {
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
            PassMode::Indirect { attrs: _, meta_attrs: Some(_), on_stack: _ } => match op.val {
                Ref(PlaceValue { llval: a, llextra: Some(b), .. }) => {
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
                PassMode::Indirect { attrs, .. } => {
                    // Indirect argument may have higher alignment requirements than the type's
                    // alignment. This can happen, e.g. when passing types with <4 byte alignment
                    // on the stack on x86.
                    let required_align = match attrs.pointee_align {
                        Some(pointee_align) => cmp::max(pointee_align, arg.layout.align.abi),
                        None => arg.layout.align.abi,
                    };
                    let scratch = PlaceValue::alloca(bx, arg.layout.size, required_align);
                    bx.lifetime_start(scratch.llval, arg.layout.size);
                    op.val.store(bx, scratch.with_type(arg.layout));
                    lifetime_ends_after_call.push((scratch.llval, arg.layout.size));
                    (scratch.llval, scratch.align, true)
                }
                PassMode::Cast { .. } => {
                    let scratch = PlaceRef::alloca(bx, arg.layout);
                    op.val.store(bx, scratch);
                    (scratch.val.llval, scratch.val.align, true)
                }
                _ => (op.immediate_or_packed_pair(bx), arg.layout.align.abi, false),
            },
            Ref(op_place_val) => match arg.mode {
                PassMode::Indirect { attrs, .. } => {
                    let required_align = match attrs.pointee_align {
                        Some(pointee_align) => cmp::max(pointee_align, arg.layout.align.abi),
                        None => arg.layout.align.abi,
                    };
                    if op_place_val.align < required_align {
                        // For `foo(packed.large_field)`, and types with <4 byte alignment on x86,
                        // alignment requirements may be higher than the type's alignment, so copy
                        // to a higher-aligned alloca.
                        let scratch = PlaceValue::alloca(bx, arg.layout.size, required_align);
                        bx.lifetime_start(scratch.llval, arg.layout.size);
                        bx.typed_place_copy(scratch, op_place_val, op.layout);
                        lifetime_ends_after_call.push((scratch.llval, arg.layout.size));
                        (scratch.llval, scratch.align, true)
                    } else {
                        (op_place_val.llval, op_place_val.align, true)
                    }
                }
                _ => (op_place_val.llval, op_place_val.align, true),
            },
            ZeroSized => match arg.mode {
                PassMode::Indirect { on_stack, .. } => {
                    if on_stack {
                        // It doesn't seem like any target can have `byval` ZSTs, so this assert
                        // is here to replace a would-be untested codepath.
                        bug!("ZST {op:?} passed on stack with abi {arg:?}");
                    }
                    // Though `extern "Rust"` doesn't pass ZSTs, some ABIs pass
                    // a pointer for `repr(C)` structs even when empty, so get
                    // one from an `alloca` (which can be left uninitialized).
                    let scratch = PlaceRef::alloca(bx, arg.layout);
                    (scratch.val.llval, scratch.val.align, true)
                }
                _ => bug!("ZST {op:?} wasn't ignored, but was passed with abi {arg:?}"),
            },
        };

        if by_ref && !arg.is_indirect() {
            // Have to load the argument, maybe while casting it.
            if let PassMode::Cast { cast, pad_i32: _ } = &arg.mode {
                // The ABI mandates that the value is passed as a different struct representation.
                // Spill and reload it from the stack to convert from the Rust representation to
                // the ABI representation.
                let scratch_size = cast.size(bx);
                let scratch_align = cast.align(bx);
                // Note that the ABI type may be either larger or smaller than the Rust type,
                // due to the presence or absence of trailing padding. For example:
                // - On some ABIs, the Rust layout { f64, f32, <f32 padding> } may omit padding
                //   when passed by value, making it smaller.
                // - On some ABIs, the Rust layout { u16, u16, u16 } may be padded up to 8 bytes
                //   when passed by value, making it larger.
                let copy_bytes = cmp::min(cast.unaligned_size(bx).bytes(), arg.layout.size.bytes());
                // Allocate some scratch space...
                let llscratch = bx.alloca(scratch_size, scratch_align);
                bx.lifetime_start(llscratch, scratch_size);
                // ...memcpy the value...
                bx.memcpy(
                    llscratch,
                    scratch_align,
                    llval,
                    align,
                    bx.const_usize(copy_bytes),
                    MemFlags::empty(),
                );
                // ...and then load it with the ABI type.
                llval = load_cast(bx, cast, llscratch, scratch_align);
                bx.lifetime_end(llscratch, scratch_size);
            } else {
                // We can't use `PlaceRef::load` here because the argument
                // may have a type we don't treat as immediate, but the ABI
                // used for this call is passing it by-value. In that case,
                // the load would just produce `OperandValue::Ref` instead
                // of the `OperandValue::Immediate` we need for the call.
                llval = bx.load(bx.backend_type(arg.layout), llval, align);
                if let BackendRepr::Scalar(scalar) = arg.layout.backend_repr {
                    if scalar.is_bool() {
                        bx.range_metadata(llval, WrappingRange { start: 0, end: 1 });
                    }
                    // We store bools as `i8` so we need to truncate to `i1`.
                    llval = bx.to_immediate_scalar(llval, scalar);
                }
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
        lifetime_ends_after_call: &mut Vec<(Bx::Value, Size)>,
    ) -> usize {
        let tuple = self.codegen_operand(bx, operand);

        // Handle both by-ref and immediate tuples.
        if let Ref(place_val) = tuple.val {
            if place_val.llextra.is_some() {
                bug!("closure arguments must be sized");
            }
            let tuple_ptr = place_val.with_type(tuple.layout);
            for i in 0..tuple.layout.fields.count() {
                let field_ptr = tuple_ptr.project_field(bx, i);
                let field = bx.load_operand(field_ptr);
                self.codegen_argument(bx, field, llargs, &args[i], lifetime_ends_after_call);
            }
        } else {
            // If the tuple is immediate, the elements are as well.
            for i in 0..tuple.layout.fields.count() {
                let op = tuple.extract_field(self, bx, i);
                self.codegen_argument(bx, op, llargs, &args[i], lifetime_ends_after_call);
            }
        }
        tuple.layout.fields.count()
    }

    pub(super) fn get_caller_location(
        &mut self,
        bx: &mut Bx,
        source_info: mir::SourceInfo,
    ) -> OperandRef<'tcx, Bx::Value> {
        self.mir.caller_location_span(source_info, self.caller_location, bx.tcx(), |span: Span| {
            let const_loc = bx.tcx().span_as_caller_location(span);
            OperandRef::from_const(bx, const_loc, bx.tcx().caller_location_ty())
        })
    }

    fn get_personality_slot(&mut self, bx: &mut Bx) -> PlaceRef<'tcx, Bx::Value> {
        let cx = bx.cx();
        if let Some(slot) = self.personality_slot {
            slot
        } else {
            let layout = cx.layout_of(Ty::new_tup(
                cx.tcx(),
                &[Ty::new_mut_ptr(cx.tcx(), cx.tcx().types.u8), cx.tcx().types.i32],
            ));
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
        if base::wants_new_eh_instructions(self.cx.sess()) {
            let cleanup_bb = Bx::append_block(self.cx, self.llfn, &format!("funclet_{bb:?}"));
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

    fn terminate_block(&mut self, reason: UnwindTerminateReason) -> Bx::BasicBlock {
        if let Some((cached_bb, cached_reason)) = self.terminate_block
            && reason == cached_reason
        {
            return cached_bb;
        }

        let funclet;
        let llbb;
        let mut bx;
        if base::wants_new_eh_instructions(self.cx.sess()) {
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
            //
            // which creates an IR snippet like
            //
            //      cs_terminate:
            //         %cs = catchswitch within none [%cp_terminate] unwind to caller
            //      cp_terminate:
            //         %cp = catchpad within %cs [null, i32 64, null]
            //         ...

            llbb = Bx::append_block(self.cx, self.llfn, "cs_terminate");
            let cp_llbb = Bx::append_block(self.cx, self.llfn, "cp_terminate");

            let mut cs_bx = Bx::build(self.cx, llbb);
            let cs = cs_bx.catch_switch(None, None, &[cp_llbb]);

            bx = Bx::build(self.cx, cp_llbb);
            let null =
                bx.const_null(bx.type_ptr_ext(bx.cx().data_layout().instruction_address_space));

            // The `null` in first argument here is actually a RTTI type
            // descriptor for the C++ personality function, but `catch (...)`
            // has no type so it's null.
            let args = if base::wants_msvc_seh(self.cx.sess()) {
                // This bitmask is a single `HT_IsStdDotDot` flag, which
                // represents that this is a C++-style `catch (...)` block that
                // only captures programmatic exceptions, not all SEH
                // exceptions. The second `null` points to a non-existent
                // `alloca` instruction, which an LLVM pass would inline into
                // the initial SEH frame allocation.
                let adjectives = bx.const_i32(0x40);
                &[null, adjectives, null] as &[_]
            } else {
                // Specifying more arguments than necessary usually doesn't
                // hurt, but the `WasmEHPrepare` LLVM pass does not recognize
                // anything other than a single `null` as a `catch (...)` block,
                // leading to problems down the line during instruction
                // selection.
                &[null] as &[_]
            };

            funclet = Some(bx.catch_pad(cs, args));
        } else {
            llbb = Bx::append_block(self.cx, self.llfn, "terminate");
            bx = Bx::build(self.cx, llbb);

            let llpersonality = self.cx.eh_personality();
            bx.filter_landing_pad(llpersonality);

            funclet = None;
        }

        self.set_debug_loc(&mut bx, mir::SourceInfo::outermost(self.mir.span));

        let (fn_abi, fn_ptr, instance) =
            common::build_langcall(&bx, self.mir.span, reason.lang_item());
        if is_call_from_compiler_builtins_to_upstream_monomorphization(bx.tcx(), instance) {
            bx.abort();
        } else {
            let fn_ty = bx.fn_decl_backend_type(fn_abi);

            let llret = bx.call(fn_ty, None, Some(fn_abi), fn_ptr, &[], funclet.as_ref(), None);
            bx.apply_attrs_to_cleanup_callsite(llret);
        }

        bx.unreachable();

        self.terminate_block = Some((llbb, reason));
        llbb
    }

    /// Get the backend `BasicBlock` for a MIR `BasicBlock`, either already
    /// cached in `self.cached_llbbs`, or created on demand (and cached).
    // FIXME(eddyb) rename `llbb` and other `ll`-prefixed things to use a
    // more backend-agnostic prefix such as `cg` (i.e. this would be `cgbb`).
    pub fn llbb(&mut self, bb: mir::BasicBlock) -> Bx::BasicBlock {
        self.try_llbb(bb).unwrap()
    }

    /// Like `llbb`, but may fail if the basic block should be skipped.
    pub(crate) fn try_llbb(&mut self, bb: mir::BasicBlock) -> Option<Bx::BasicBlock> {
        match self.cached_llbbs[bb] {
            CachedLlbb::None => {
                let llbb = Bx::append_block(self.cx, self.llfn, &format!("{bb:?}"));
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
                        llargs.push(tmp.val.llval);
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
            self.codegen_place(bx, dest.as_ref())
        };
        if fn_ret.is_indirect() {
            if dest.val.align < dest.layout.align.abi {
                // Currently, MIR code generation does not create calls
                // that store directly to fields of packed structs (in
                // fact, the calls it creates write only to temps).
                //
                // If someone changes that, please update this code path
                // to create a temporary.
                span_bug!(self.mir.span, "can't directly store to unaligned value");
            }
            llargs.push(dest.val.llval);
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
            Store(dst) => bx.store_arg(ret_abi, llval, dst),
            IndirectOperand(tmp, index) => {
                let op = bx.load_operand(tmp);
                tmp.storage_dead(bx);
                self.overwrite_local(index, LocalRef::Operand(op));
                self.debug_introduce_local(bx, index);
            }
            DirectOperand(index) => {
                // If there is a cast, we have to store and reload.
                let op = if let PassMode::Cast { .. } = ret_abi.mode {
                    let tmp = PlaceRef::alloca(bx, ret_abi.layout);
                    tmp.storage_live(bx);
                    bx.store_arg(ret_abi, llval, tmp);
                    let op = bx.load_operand(tmp);
                    tmp.storage_dead(bx);
                    op
                } else {
                    OperandRef::from_immediate_or_packed_pair(bx, llval, ret_abi.layout)
                };
                self.overwrite_local(index, LocalRef::Operand(op));
                self.debug_introduce_local(bx, index);
            }
        }
    }
}

enum ReturnDest<'tcx, V> {
    /// Do nothing; the return value is indirect or ignored.
    Nothing,
    /// Store the return value to the pointer.
    Store(PlaceRef<'tcx, V>),
    /// Store an indirect return value to an operand local place.
    IndirectOperand(PlaceRef<'tcx, V>, mir::Local),
    /// Store a direct return value to an operand local place.
    DirectOperand(mir::Local),
}

fn load_cast<'a, 'tcx, Bx: BuilderMethods<'a, 'tcx>>(
    bx: &mut Bx,
    cast: &CastTarget,
    ptr: Bx::Value,
    align: Align,
) -> Bx::Value {
    let cast_ty = bx.cast_backend_type(cast);
    if let Some(offset_from_start) = cast.rest_offset {
        assert!(cast.prefix[1..].iter().all(|p| p.is_none()));
        assert_eq!(cast.rest.unit.size, cast.rest.total);
        let first_ty = bx.reg_backend_type(&cast.prefix[0].unwrap());
        let second_ty = bx.reg_backend_type(&cast.rest.unit);
        let first = bx.load(first_ty, ptr, align);
        let second_ptr = bx.inbounds_ptradd(ptr, bx.const_usize(offset_from_start.bytes()));
        let second = bx.load(second_ty, second_ptr, align.restrict_for_offset(offset_from_start));
        let res = bx.cx().const_poison(cast_ty);
        let res = bx.insert_value(res, first, 0);
        bx.insert_value(res, second, 1)
    } else {
        bx.load(cast_ty, ptr, align)
    }
}

pub fn store_cast<'a, 'tcx, Bx: BuilderMethods<'a, 'tcx>>(
    bx: &mut Bx,
    cast: &CastTarget,
    value: Bx::Value,
    ptr: Bx::Value,
    align: Align,
) {
    if let Some(offset_from_start) = cast.rest_offset {
        assert!(cast.prefix[1..].iter().all(|p| p.is_none()));
        assert_eq!(cast.rest.unit.size, cast.rest.total);
        assert!(cast.prefix[0].is_some());
        let first = bx.extract_value(value, 0);
        let second = bx.extract_value(value, 1);
        bx.store(first, ptr, align);
        let second_ptr = bx.inbounds_ptradd(ptr, bx.const_usize(offset_from_start.bytes()));
        bx.store(second, second_ptr, align.restrict_for_offset(offset_from_start));
    } else {
        bx.store(value, ptr, align);
    };
}
