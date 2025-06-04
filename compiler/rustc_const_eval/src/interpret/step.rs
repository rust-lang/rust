//! This module contains the `InterpCx` methods for executing a single step of the interpreter.
//!
//! The main entry point is the `step` method.

use either::Either;
use rustc_abi::{FIRST_VARIANT, FieldIdx};
use rustc_index::IndexSlice;
use rustc_middle::ty::layout::FnAbiOf;
use rustc_middle::ty::{self, Instance, Ty};
use rustc_middle::{bug, mir, span_bug};
use rustc_span::source_map::Spanned;
use rustc_target::callconv::FnAbi;
use tracing::{info, instrument, trace};

use super::{
    FnArg, FnVal, ImmTy, Immediate, InterpCx, InterpResult, Machine, MemPlaceMeta, PlaceTy,
    Projectable, Scalar, interp_ok, throw_ub, throw_unsup_format,
};
use crate::util;

struct EvaluatedCalleeAndArgs<'tcx, M: Machine<'tcx>> {
    callee: FnVal<'tcx, M::ExtraFnVal>,
    args: Vec<FnArg<'tcx, M::Provenance>>,
    fn_sig: ty::FnSig<'tcx>,
    fn_abi: &'tcx FnAbi<'tcx, Ty<'tcx>>,
    /// True if the function is marked as `#[track_caller]` ([`ty::InstanceKind::requires_caller_location`])
    with_caller_location: bool,
}

impl<'tcx, M: Machine<'tcx>> InterpCx<'tcx, M> {
    /// Returns `true` as long as there are more things to do.
    ///
    /// This is used by [priroda](https://github.com/oli-obk/priroda)
    ///
    /// This is marked `#inline(always)` to work around adversarial codegen when `opt-level = 3`
    #[inline(always)]
    pub fn step(&mut self) -> InterpResult<'tcx, bool> {
        if self.stack().is_empty() {
            return interp_ok(false);
        }

        let Either::Left(loc) = self.frame().loc else {
            // We are unwinding and this fn has no cleanup code.
            // Just go on unwinding.
            trace!("unwinding: skipping frame");
            self.return_from_current_stack_frame(/* unwinding */ true)?;
            return interp_ok(true);
        };
        let basic_block = &self.body().basic_blocks[loc.block];

        if let Some(stmt) = basic_block.statements.get(loc.statement_index) {
            let old_frames = self.frame_idx();
            self.eval_statement(stmt)?;
            // Make sure we are not updating `statement_index` of the wrong frame.
            assert_eq!(old_frames, self.frame_idx());
            // Advance the program counter.
            self.frame_mut().loc.as_mut().left().unwrap().statement_index += 1;
            return interp_ok(true);
        }

        M::before_terminator(self)?;

        let terminator = basic_block.terminator();
        self.eval_terminator(terminator)?;
        if !self.stack().is_empty() {
            if let Either::Left(loc) = self.frame().loc {
                info!("// executing {:?}", loc.block);
            }
        }
        interp_ok(true)
    }

    /// Runs the interpretation logic for the given `mir::Statement` at the current frame and
    /// statement counter.
    ///
    /// This does NOT move the statement counter forward, the caller has to do that!
    pub fn eval_statement(&mut self, stmt: &mir::Statement<'tcx>) -> InterpResult<'tcx> {
        info!("{:?}", stmt);

        use rustc_middle::mir::StatementKind::*;

        match &stmt.kind {
            Assign(box (place, rvalue)) => self.eval_rvalue_into_place(rvalue, *place)?,

            SetDiscriminant { place, variant_index } => {
                let dest = self.eval_place(**place)?;
                self.write_discriminant(*variant_index, &dest)?;
            }

            Deinit(place) => {
                let dest = self.eval_place(**place)?;
                self.write_uninit(&dest)?;
            }

            // Mark locals as alive
            StorageLive(local) => {
                self.storage_live(*local)?;
            }

            // Mark locals as dead
            StorageDead(local) => {
                self.storage_dead(*local)?;
            }

            // No dynamic semantics attached to `FakeRead`; MIR
            // interpreter is solely intended for borrowck'ed code.
            FakeRead(..) => {}

            // Stacked Borrows.
            Retag(kind, place) => {
                let dest = self.eval_place(**place)?;
                M::retag_place_contents(self, *kind, &dest)?;
            }

            Intrinsic(box intrinsic) => self.eval_nondiverging_intrinsic(intrinsic)?,

            // Evaluate the place expression, without reading from it.
            PlaceMention(box place) => {
                let _ = self.eval_place(*place)?;
            }

            // This exists purely to guide borrowck lifetime inference, and does not have
            // an operational effect.
            AscribeUserType(..) => {}

            // Currently, Miri discards Coverage statements. Coverage statements are only injected
            // via an optional compile time MIR pass and have no side effects. Since Coverage
            // statements don't exist at the source level, it is safe for Miri to ignore them, even
            // for undefined behavior (UB) checks.
            //
            // A coverage counter inside a const expression (for example, a counter injected in a
            // const function) is discarded when the const is evaluated at compile time. Whether
            // this should change, and/or how to implement a const eval counter, is a subject of the
            // following issue:
            //
            // FIXME(#73156): Handle source code coverage in const eval
            Coverage(..) => {}

            ConstEvalCounter => {
                M::increment_const_eval_counter(self)?;
            }

            // Defined to do nothing. These are added by optimization passes, to avoid changing the
            // size of MIR constantly.
            Nop => {}

            // Only used for temporary lifetime lints
            BackwardIncompatibleDropHint { .. } => {}
        }

        interp_ok(())
    }

    /// Evaluate an assignment statement.
    ///
    /// There is no separate `eval_rvalue` function. Instead, the code for handling each rvalue
    /// type writes its results directly into the memory specified by the place.
    pub fn eval_rvalue_into_place(
        &mut self,
        rvalue: &mir::Rvalue<'tcx>,
        place: mir::Place<'tcx>,
    ) -> InterpResult<'tcx> {
        let dest = self.eval_place(place)?;
        // FIXME: ensure some kind of non-aliasing between LHS and RHS?
        // Also see https://github.com/rust-lang/rust/issues/68364.

        use rustc_middle::mir::Rvalue::*;
        match *rvalue {
            ThreadLocalRef(did) => {
                let ptr = M::thread_local_static_pointer(self, did)?;
                self.write_pointer(ptr, &dest)?;
            }

            Use(ref operand) => {
                // Avoid recomputing the layout
                let op = self.eval_operand(operand, Some(dest.layout))?;
                self.copy_op(&op, &dest)?;
            }

            CopyForDeref(place) => {
                let op = self.eval_place_to_op(place, Some(dest.layout))?;
                self.copy_op(&op, &dest)?;
            }

            BinaryOp(bin_op, box (ref left, ref right)) => {
                let layout = util::binop_left_homogeneous(bin_op).then_some(dest.layout);
                let left = self.read_immediate(&self.eval_operand(left, layout)?)?;
                let layout = util::binop_right_homogeneous(bin_op).then_some(left.layout);
                let right = self.read_immediate(&self.eval_operand(right, layout)?)?;
                let result = self.binary_op(bin_op, &left, &right)?;
                assert_eq!(result.layout, dest.layout, "layout mismatch for result of {bin_op:?}");
                self.write_immediate(*result, &dest)?;
            }

            UnaryOp(un_op, ref operand) => {
                // The operand always has the same type as the result.
                let val = self.read_immediate(&self.eval_operand(operand, Some(dest.layout))?)?;
                let result = self.unary_op(un_op, &val)?;
                assert_eq!(result.layout, dest.layout, "layout mismatch for result of {un_op:?}");
                self.write_immediate(*result, &dest)?;
            }

            NullaryOp(null_op, ty) => {
                let ty = self.instantiate_from_current_frame_and_normalize_erasing_regions(ty)?;
                let val = self.nullary_op(null_op, ty)?;
                self.write_immediate(*val, &dest)?;
            }

            Aggregate(box ref kind, ref operands) => {
                self.write_aggregate(kind, operands, &dest)?;
            }

            Repeat(ref operand, _) => {
                self.write_repeat(operand, &dest)?;
            }

            Len(place) => {
                let src = self.eval_place(place)?;
                let len = src.len(self)?;
                self.write_scalar(Scalar::from_target_usize(len, self), &dest)?;
            }

            Ref(_, borrow_kind, place) => {
                let src = self.eval_place(place)?;
                let place = self.force_allocation(&src)?;
                let val = ImmTy::from_immediate(place.to_ref(self), dest.layout);
                // A fresh reference was created, make sure it gets retagged.
                let val = M::retag_ptr_value(
                    self,
                    if borrow_kind.allows_two_phase_borrow() {
                        mir::RetagKind::TwoPhase
                    } else {
                        mir::RetagKind::Default
                    },
                    &val,
                )?;
                self.write_immediate(*val, &dest)?;
            }

            RawPtr(kind, place) => {
                // Figure out whether this is an addr_of of an already raw place.
                let place_base_raw = if place.is_indirect_first_projection() {
                    let ty = self.frame().body.local_decls[place.local].ty;
                    ty.is_raw_ptr()
                } else {
                    // Not a deref, and thus not raw.
                    false
                };

                let src = self.eval_place(place)?;
                let place = self.force_allocation(&src)?;
                let mut val = ImmTy::from_immediate(place.to_ref(self), dest.layout);
                if !place_base_raw && !kind.is_fake() {
                    // If this was not already raw, it needs retagging -- except for "fake"
                    // raw borrows whose defining property is that they do not get retagged.
                    val = M::retag_ptr_value(self, mir::RetagKind::Raw, &val)?;
                }
                self.write_immediate(*val, &dest)?;
            }

            ShallowInitBox(ref operand, _) => {
                let src = self.eval_operand(operand, None)?;
                let v = self.read_immediate(&src)?;
                self.write_immediate(*v, &dest)?;
            }

            Cast(cast_kind, ref operand, cast_ty) => {
                let src = self.eval_operand(operand, None)?;
                let cast_ty =
                    self.instantiate_from_current_frame_and_normalize_erasing_regions(cast_ty)?;
                self.cast(&src, cast_kind, cast_ty, &dest)?;
            }

            Discriminant(place) => {
                let op = self.eval_place_to_op(place, None)?;
                let variant = self.read_discriminant(&op)?;
                let discr = self.discriminant_for_variant(op.layout.ty, variant)?;
                self.write_immediate(*discr, &dest)?;
            }

            WrapUnsafeBinder(ref op, _ty) => {
                // Constructing an unsafe binder acts like a transmute
                // since the operand's layout does not change.
                let op = self.eval_operand(op, None)?;
                self.copy_op_allow_transmute(&op, &dest)?;
            }
        }

        trace!("{:?}", self.dump_place(&dest));

        interp_ok(())
    }

    /// Writes the aggregate to the destination.
    #[instrument(skip(self), level = "trace")]
    fn write_aggregate(
        &mut self,
        kind: &mir::AggregateKind<'tcx>,
        operands: &IndexSlice<FieldIdx, mir::Operand<'tcx>>,
        dest: &PlaceTy<'tcx, M::Provenance>,
    ) -> InterpResult<'tcx> {
        self.write_uninit(dest)?; // make sure all the padding ends up as uninit
        let (variant_index, variant_dest, active_field_index) = match *kind {
            mir::AggregateKind::Adt(_, variant_index, _, _, active_field_index) => {
                let variant_dest = self.project_downcast(dest, variant_index)?;
                (variant_index, variant_dest, active_field_index)
            }
            mir::AggregateKind::RawPtr(..) => {
                // Pointers don't have "fields" in the normal sense, so the
                // projection-based code below would either fail in projection
                // or in type mismatches. Instead, build an `Immediate` from
                // the parts and write that to the destination.
                let [data, meta] = &operands.raw else {
                    bug!("{kind:?} should have 2 operands, had {operands:?}");
                };
                let data = self.eval_operand(data, None)?;
                let data = self.read_pointer(&data)?;
                let meta = self.eval_operand(meta, None)?;
                let meta = if meta.layout.is_zst() {
                    MemPlaceMeta::None
                } else {
                    MemPlaceMeta::Meta(self.read_scalar(&meta)?)
                };
                let ptr_imm = Immediate::new_pointer_with_meta(data, meta, self);
                let ptr = ImmTy::from_immediate(ptr_imm, dest.layout);
                self.copy_op(&ptr, dest)?;
                return interp_ok(());
            }
            _ => (FIRST_VARIANT, dest.clone(), None),
        };
        if active_field_index.is_some() {
            assert_eq!(operands.len(), 1);
        }
        for (field_index, operand) in operands.iter_enumerated() {
            let field_index = active_field_index.unwrap_or(field_index);
            let field_dest = self.project_field(&variant_dest, field_index)?;
            let op = self.eval_operand(operand, Some(field_dest.layout))?;
            self.copy_op(&op, &field_dest)?;
        }
        self.write_discriminant(variant_index, dest)
    }

    /// Repeats `operand` into the destination. `dest` must have array type, and that type
    /// determines how often `operand` is repeated.
    fn write_repeat(
        &mut self,
        operand: &mir::Operand<'tcx>,
        dest: &PlaceTy<'tcx, M::Provenance>,
    ) -> InterpResult<'tcx> {
        let src = self.eval_operand(operand, None)?;
        assert!(src.layout.is_sized());
        let dest = self.force_allocation(&dest)?;
        let length = dest.len(self)?;

        if length == 0 {
            // Nothing to copy... but let's still make sure that `dest` as a place is valid.
            self.get_place_alloc_mut(&dest)?;
        } else {
            // Write the src to the first element.
            let first = self.project_index(&dest, 0)?;
            self.copy_op(&src, &first)?;

            // This is performance-sensitive code for big static/const arrays! So we
            // avoid writing each operand individually and instead just make many copies
            // of the first element.
            let elem_size = first.layout.size;
            let first_ptr = first.ptr();
            let rest_ptr = first_ptr.wrapping_offset(elem_size, self);
            // No alignment requirement since `copy_op` above already checked it.
            self.mem_copy_repeatedly(
                first_ptr,
                rest_ptr,
                elem_size,
                length - 1,
                /*nonoverlapping:*/ true,
            )?;
        }

        interp_ok(())
    }

    /// Evaluate the arguments of a function call
    fn eval_fn_call_argument(
        &self,
        op: &mir::Operand<'tcx>,
    ) -> InterpResult<'tcx, FnArg<'tcx, M::Provenance>> {
        interp_ok(match op {
            mir::Operand::Copy(_) | mir::Operand::Constant(_) => {
                // Make a regular copy.
                let op = self.eval_operand(op, None)?;
                FnArg::Copy(op)
            }
            mir::Operand::Move(place) => {
                // If this place lives in memory, preserve its location.
                // We call `place_to_op` which will be an `MPlaceTy` whenever there exists
                // an mplace for this place. (This is in contrast to `PlaceTy::as_mplace_or_local`
                // which can return a local even if that has an mplace.)
                let place = self.eval_place(*place)?;
                let op = self.place_to_op(&place)?;

                match op.as_mplace_or_imm() {
                    Either::Left(mplace) => FnArg::InPlace(mplace),
                    Either::Right(_imm) => {
                        // This argument doesn't live in memory, so there's no place
                        // to make inaccessible during the call.
                        // We rely on there not being any stray `PlaceTy` that would let the
                        // caller directly access this local!
                        // This is also crucial for tail calls, where we want the `FnArg` to
                        // stay valid when the old stack frame gets popped.
                        FnArg::Copy(op)
                    }
                }
            }
        })
    }

    /// Shared part of `Call` and `TailCall` implementation — finding and evaluating all the
    /// necessary information about callee and arguments to make a call.
    fn eval_callee_and_args(
        &self,
        terminator: &mir::Terminator<'tcx>,
        func: &mir::Operand<'tcx>,
        args: &[Spanned<mir::Operand<'tcx>>],
    ) -> InterpResult<'tcx, EvaluatedCalleeAndArgs<'tcx, M>> {
        let func = self.eval_operand(func, None)?;
        let args = args
            .iter()
            .map(|arg| self.eval_fn_call_argument(&arg.node))
            .collect::<InterpResult<'tcx, Vec<_>>>()?;

        let fn_sig_binder = func.layout.ty.fn_sig(*self.tcx);
        let fn_sig = self.tcx.normalize_erasing_late_bound_regions(self.typing_env, fn_sig_binder);
        let extra_args = &args[fn_sig.inputs().len()..];
        let extra_args =
            self.tcx.mk_type_list_from_iter(extra_args.iter().map(|arg| arg.layout().ty));

        let (callee, fn_abi, with_caller_location) = match *func.layout.ty.kind() {
            ty::FnPtr(..) => {
                let fn_ptr = self.read_pointer(&func)?;
                let fn_val = self.get_ptr_fn(fn_ptr)?;
                (fn_val, self.fn_abi_of_fn_ptr(fn_sig_binder, extra_args)?, false)
            }
            ty::FnDef(def_id, args) => {
                let instance = self.resolve(def_id, args)?;
                (
                    FnVal::Instance(instance),
                    self.fn_abi_of_instance(instance, extra_args)?,
                    instance.def.requires_caller_location(*self.tcx),
                )
            }
            _ => {
                span_bug!(terminator.source_info.span, "invalid callee of type {}", func.layout.ty)
            }
        };

        interp_ok(EvaluatedCalleeAndArgs { callee, args, fn_sig, fn_abi, with_caller_location })
    }

    fn eval_terminator(&mut self, terminator: &mir::Terminator<'tcx>) -> InterpResult<'tcx> {
        info!("{:?}", terminator.kind);

        use rustc_middle::mir::TerminatorKind::*;
        match terminator.kind {
            Return => {
                self.return_from_current_stack_frame(/* unwinding */ false)?
            }

            Goto { target } => self.go_to_block(target),

            SwitchInt { ref discr, ref targets } => {
                let discr = self.read_immediate(&self.eval_operand(discr, None)?)?;
                trace!("SwitchInt({:?})", *discr);

                // Branch to the `otherwise` case by default, if no match is found.
                let mut target_block = targets.otherwise();

                for (const_int, target) in targets.iter() {
                    // Compare using MIR BinOp::Eq, to also support pointer values.
                    // (Avoiding `self.binary_op` as that does some redundant layout computation.)
                    let res = self.binary_op(
                        mir::BinOp::Eq,
                        &discr,
                        &ImmTy::from_uint(const_int, discr.layout),
                    )?;
                    if res.to_scalar().to_bool()? {
                        target_block = target;
                        break;
                    }
                }

                self.go_to_block(target_block);
            }

            Call {
                ref func,
                ref args,
                destination,
                target,
                unwind,
                call_source: _,
                fn_span: _,
            } => {
                let old_stack = self.frame_idx();
                let old_loc = self.frame().loc;

                let EvaluatedCalleeAndArgs { callee, args, fn_sig, fn_abi, with_caller_location } =
                    self.eval_callee_and_args(terminator, func, args)?;

                let destination = self.eval_place(destination)?;
                self.init_fn_call(
                    callee,
                    (fn_sig.abi, fn_abi),
                    &args,
                    with_caller_location,
                    &destination,
                    target,
                    if fn_abi.can_unwind { unwind } else { mir::UnwindAction::Unreachable },
                )?;
                // Sanity-check that `eval_fn_call` either pushed a new frame or
                // did a jump to another block.
                if self.frame_idx() == old_stack && self.frame().loc == old_loc {
                    span_bug!(terminator.source_info.span, "evaluating this call made no progress");
                }
            }

            TailCall { ref func, ref args, fn_span: _ } => {
                let old_frame_idx = self.frame_idx();

                let EvaluatedCalleeAndArgs { callee, args, fn_sig, fn_abi, with_caller_location } =
                    self.eval_callee_and_args(terminator, func, args)?;

                self.init_fn_tail_call(callee, (fn_sig.abi, fn_abi), &args, with_caller_location)?;

                if self.frame_idx() != old_frame_idx {
                    span_bug!(
                        terminator.source_info.span,
                        "evaluating this tail call pushed a new stack frame"
                    );
                }
            }

            Drop { place, target, unwind, replace: _, drop, async_fut } => {
                assert!(
                    async_fut.is_none() && drop.is_none(),
                    "Async Drop must be expanded or reset to sync in runtime MIR"
                );
                let place = self.eval_place(place)?;
                let instance = Instance::resolve_drop_in_place(*self.tcx, place.layout.ty);
                if let ty::InstanceKind::DropGlue(_, None) = instance.def {
                    // This is the branch we enter if and only if the dropped type has no drop glue
                    // whatsoever. This can happen as a result of monomorphizing a drop of a
                    // generic. In order to make sure that generic and non-generic code behaves
                    // roughly the same (and in keeping with Mir semantics) we do nothing here.
                    self.go_to_block(target);
                    return interp_ok(());
                }
                trace!("TerminatorKind::drop: {:?}, type {}", place, place.layout.ty);
                self.init_drop_in_place_call(&place, instance, target, unwind)?;
            }

            Assert { ref cond, expected, ref msg, target, unwind } => {
                let ignored =
                    M::ignore_optional_overflow_checks(self) && msg.is_optional_overflow_check();
                let cond_val = self.read_scalar(&self.eval_operand(cond, None)?)?.to_bool()?;
                if ignored || expected == cond_val {
                    self.go_to_block(target);
                } else {
                    M::assert_panic(self, msg, unwind)?;
                }
            }

            UnwindTerminate(reason) => {
                M::unwind_terminate(self, reason)?;
            }

            // When we encounter Resume, we've finished unwinding
            // cleanup for the current stack frame. We pop it in order
            // to continue unwinding the next frame
            UnwindResume => {
                trace!("unwinding: resuming from cleanup");
                // By definition, a Resume terminator means
                // that we're unwinding
                self.return_from_current_stack_frame(/* unwinding */ true)?;
                return interp_ok(());
            }

            // It is UB to ever encounter this.
            Unreachable => throw_ub!(Unreachable),

            // These should never occur for MIR we actually run.
            FalseEdge { .. } | FalseUnwind { .. } | Yield { .. } | CoroutineDrop => span_bug!(
                terminator.source_info.span,
                "{:#?} should have been eliminated by MIR pass",
                terminator.kind
            ),

            InlineAsm { .. } => {
                throw_unsup_format!("inline assembly is not supported");
            }
        }

        interp_ok(())
    }
}
