//! This module contains the `InterpCx` methods for executing a single step of the interpreter.
//!
//! The main entry point is the `step` method.

use std::iter;

use either::Either;
use rustc_abi::{FIRST_VARIANT, FieldIdx};
use rustc_data_structures::fx::FxHashSet;
use rustc_index::{IndexSlice, IndexVec};
use rustc_middle::ty::layout::TyAndLayout;
use rustc_middle::ty::{self, Instance, Ty};
use rustc_middle::{bug, mir, span_bug};
use rustc_span::Spanned;
use rustc_target::callconv::FnAbi;
use tracing::field::Empty;
use tracing::{info, instrument, trace};

use super::{
    EnteredTraceSpan, FnArg, FnVal, ImmTy, Immediate, InterpCx, InterpResult, Machine,
    MemPlaceMeta, OpTy, PlaceTy, Projectable, Provenance, RetagMode, Scalar, interp_ok, throw_ub,
    throw_unsup_format,
};
use crate::{enter_trace_span, util};

/// An evaluated rvalue, with destination-dependent operations deferred until writing.
enum EvaluatedRvalue<'tcx, Prov: Provenance> {
    Use(OpTy<'tcx, Prov>, mir::WithRetag),
    Immediate(ImmTy<'tcx, Prov>),
    Ref(ImmTy<'tcx, Prov>, RetagMode),
    RawPtr { val: ImmTy<'tcx, Prov>, needs_retag: bool },
    Copy { op: OpTy<'tcx, Prov>, allow_transmute: bool },
    Cast(OpTy<'tcx, Prov>, mir::CastKind, Ty<'tcx>),
    Aggregate(mir::AggregateKind<'tcx>, IndexVec<FieldIdx, OpTy<'tcx, Prov>>),
    Repeat(OpTy<'tcx, Prov>),
}

struct EvaluatedCalleeAndArgs<'tcx, M: Machine<'tcx>> {
    callee: FnVal<'tcx, M::ExtraFnVal>,
    args: Vec<FnArg<'tcx, M::Provenance>>,
    fn_sig: ty::FnSig<'tcx>,
    /// None if LLVM intrinsic
    fn_abi: Option<&'tcx FnAbi<'tcx, Ty<'tcx>>>,
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
        let _trace = enter_trace_span!(
            M,
            step::eval_statement,
            stmt = ?stmt.kind,
            span = ?stmt.source_info.span,
            tracing_separate_thread = Empty,
        )
        .or_if_tracing_disabled(|| info!("{:?}", stmt.kind));

        use rustc_middle::mir::StatementKind::*;

        match &stmt.kind {
            Assign((place, rvalue)) => self.eval_rvalue_into_place(rvalue, *place)?,

            SetDiscriminant { place, variant_index } => {
                let dest =
                    self.eval_place(**place, /* skip_validity_for_simple_deref */ false)?;
                self.write_discriminant(*variant_index, &dest)?;
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

            Intrinsic(intrinsic) => self.eval_nondiverging_intrinsic(intrinsic)?,

            // Evaluate the place expression, without reading from it.
            PlaceMention(place) => {
                let _ =
                    self.eval_place(**place, /* skip_validity_for_simple_deref */ false)?;
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
    pub fn eval_rvalue_into_place(
        &mut self,
        rvalue: &mir::Rvalue<'tcx>,
        place: mir::Place<'tcx>,
    ) -> InterpResult<'tcx> {
        // We can skip validity because we'll write to the place which checks everything we care
        // about for references, and the pointee must be sized so there's nothing to check for raw
        // pointers.
        let dest = self.eval_place(place, /* skip_validity_for_simple_deref */ true)?;
        let value = self.eval_rvalue(rvalue, dest.layout)?;
        self.write_rvalue(value, dest)
    }

    /// Evaluate an rvalue, leaving destination-dependent operations for `write_rvalue`.
    fn eval_rvalue(
        &mut self,
        rvalue: &mir::Rvalue<'tcx>,
        layout: TyAndLayout<'tcx>,
    ) -> InterpResult<'tcx, EvaluatedRvalue<'tcx, M::Provenance>> {
        use rustc_middle::mir::Rvalue::*;
        interp_ok(match *rvalue {
            Use(ref operand, with_retag) => {
                EvaluatedRvalue::Use(self.eval_operand(operand, Some(layout))?, with_retag)
            }
            Repeat(ref operand, _) => EvaluatedRvalue::Repeat(self.eval_operand(operand, None)?),
            Ref(_, borrow_kind, place) => {
                // `x = &*ptr` does not need a validity check on `ptr` because we will already
                // check `x` when writing to the destination.
                let src = self.eval_place(place, /* skip_validity_for_simple_deref */ true)?;
                let place = self.force_allocation(&src)?;
                let val = ImmTy::from_immediate(place.to_ref(self), layout);
                let mode = if borrow_kind.is_two_phase_borrow() {
                    RetagMode::TwoPhase
                } else {
                    RetagMode::Default
                };
                EvaluatedRvalue::Ref(val, mode)
            }
            RawPtr(kind, place) => {
                let place_base_raw = place.is_indirect_first_projection()
                    && self.frame().body.local_decls[place.local].ty.is_raw_ptr();
                let src =
                    self.eval_place(place, /* skip_validity_for_simple_deref */ false)?;
                let place = self.force_allocation(&src)?;
                let val = ImmTy::from_immediate(place.to_ref(self), layout);
                // Retag unless the place was already raw or this is a "fake" raw borrow.
                EvaluatedRvalue::RawPtr { val, needs_retag: !place_base_raw && !kind.is_fake() }
            }
            ThreadLocalRef(did) => {
                let ptr = M::thread_local_static_pointer(self, did)?;
                EvaluatedRvalue::Immediate(ImmTy::from_scalar(
                    Scalar::from_maybe_pointer(ptr.into(), self),
                    layout,
                ))
            }
            Cast(kind, ref operand, ty) => {
                let op = self.eval_operand(operand, None)?;
                let ty = self.instantiate_from_current_frame_and_normalize_erasing_regions(ty)?;
                EvaluatedRvalue::Cast(op, kind, ty)
            }
            BinaryOp(bin_op, (ref left, ref right)) => {
                let operand_layout = util::binop_left_homogeneous(bin_op).then_some(layout);
                let left = self.read_immediate(&self.eval_operand(left, operand_layout)?)?;
                let operand_layout = util::binop_right_homogeneous(bin_op).then_some(left.layout);
                let right = self.read_immediate(&self.eval_operand(right, operand_layout)?)?;
                let result = self.binary_op(bin_op, &left, &right)?;
                assert_eq!(result.layout, layout, "layout mismatch for result of {bin_op:?}");
                EvaluatedRvalue::Immediate(result)
            }
            UnaryOp(un_op, ref operand) => {
                let operand_layout = util::unop_homogeneous(un_op).then_some(layout);
                let val = self.read_immediate(&self.eval_operand(operand, operand_layout)?)?;
                let result = self.unary_op(un_op, &val)?;
                assert_eq!(result.layout, layout, "layout mismatch for result of {un_op:?}");
                EvaluatedRvalue::Immediate(result)
            }
            Discriminant(place) => {
                let op = self.eval_place_to_op(place, None)?;
                let variant = self.read_discriminant(&op)?;
                EvaluatedRvalue::Immediate(self.discriminant_for_variant(op.layout.ty, variant)?)
            }
            Reborrow(_, mutability, place) => {
                // Shared generic reborrows use `CoerceShared`: a bitwise copy into a
                // distinct same-layout target ADT.
                EvaluatedRvalue::Copy {
                    op: self.eval_place_to_op(place, None)?,
                    allow_transmute: mutability.is_not(),
                }
            }
            Aggregate(ref kind, ref operands) => {
                let operands = operands
                    .iter()
                    .map(|operand| self.eval_operand(operand, None))
                    .collect::<InterpResult<'tcx, IndexVec<FieldIdx, _>>>()?;
                EvaluatedRvalue::Aggregate((**kind).clone(), operands)
            }
            WrapUnsafeBinder(ref operand, _) => {
                // Constructing an unsafe binder acts like a transmute
                // since the operand's layout does not change.
                EvaluatedRvalue::Copy {
                    op: self.eval_operand(operand, None)?,
                    allow_transmute: true,
                }
            }
            CopyForDeref(_) => bug!("`CopyForDeref` in runtime MIR"),
        })
    }

    /// Write an evaluated rvalue, performing destination-dependent conversion and validation.
    fn write_rvalue(
        &mut self,
        value: EvaluatedRvalue<'tcx, M::Provenance>,
        dest: PlaceTy<'tcx, M::Provenance>,
    ) -> InterpResult<'tcx> {
        // FIXME: ensure some kind of non-aliasing between LHS and RHS?
        // Also see https://github.com/rust-lang/rust/issues/68364.

        match value {
            EvaluatedRvalue::Use(op, with_retag) => {
                let mode = if with_retag.yes() { RetagMode::Default } else { RetagMode::None };
                M::with_retag_mode(self, mode, |ecx| ecx.copy_op(&op, &dest))?;
            }
            EvaluatedRvalue::Immediate(val) => self.write_immediate(*val, &dest)?,
            EvaluatedRvalue::Ref(mut val, mode) => {
                // A fresh reference was created, make sure it gets retagged with the right mode.
                M::with_retag_mode(self, mode, |ecx| {
                    // If validation is disabled, we still want to do this retag. This is because
                    // const-eval disables validation for performance reasons but wants to retag
                    // shared references. So we add a bit of a hack here to do the retag manually
                    // if the write would not incur validation.
                    if !M::enforce_validity(ecx, val.layout) {
                        if let Some(new_val) = M::retag_ptr_value(ecx, &val, val.layout.ty)? {
                            val = new_val;
                        }
                    }
                    ecx.write_immediate(*val, &dest)
                })?;
            }
            EvaluatedRvalue::RawPtr { mut val, needs_retag } => {
                if needs_retag {
                    val = M::with_retag_mode(self, RetagMode::Raw, |ecx| {
                        interp_ok(M::retag_ptr_value(ecx, &val, val.layout.ty)?.unwrap_or(val))
                    })?;
                }
                // Writing a raw pointer does not retag it during validation.
                self.write_immediate(*val, &dest)?;
            }
            EvaluatedRvalue::Copy { op, allow_transmute } => {
                if allow_transmute {
                    self.copy_op_allow_transmute(&op, &dest)?;
                } else {
                    self.copy_op(&op, &dest)?;
                }
            }
            EvaluatedRvalue::Cast(op, kind, ty) => self.cast(&op, kind, ty, &dest)?,
            EvaluatedRvalue::Aggregate(kind, operands) => {
                self.write_aggregate(&kind, &operands, &dest)?;
            }
            EvaluatedRvalue::Repeat(op) => self.write_repeat(&op, &dest)?,
        }
        interp_ok(())
    }

    /// Writes the aggregate to the destination.
    #[instrument(skip(self), level = "trace")]
    fn write_aggregate(
        &mut self,
        kind: &mir::AggregateKind<'tcx>,
        operands: &IndexSlice<FieldIdx, OpTy<'tcx, M::Provenance>>,
        dest: &PlaceTy<'tcx, M::Provenance>,
    ) -> InterpResult<'tcx> {
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
                let data = self.read_pointer(data)?;
                let meta = if meta.layout.is_zst() {
                    MemPlaceMeta::None
                } else {
                    MemPlaceMeta::Meta(self.read_scalar(meta)?)
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
            // We validate manually below so we don't have to do it here.
            self.copy_op_no_validate(operand, &field_dest, /*allow_transmute*/ false)?;
        }
        self.write_discriminant(variant_index, dest)?;
        // Validate that the entire thing is valid, and reset padding that might be in between the
        // fields.
        if M::enforce_validity(self, dest.layout()) {
            self.validate_place(
                dest,
                M::enforce_validity_recursively(self, dest.layout()),
                /*reset_provenance_and_padding*/ true,
            )?;
        }
        interp_ok(())
    }

    /// Repeats `src` into the destination. `dest` must have array type, and that type
    /// determines how often `src` is repeated.
    fn write_repeat(
        &mut self,
        src: &OpTy<'tcx, M::Provenance>,
        dest: &PlaceTy<'tcx, M::Provenance>,
    ) -> InterpResult<'tcx> {
        assert!(src.layout.is_sized());
        let dest = self.force_allocation(&dest)?;
        let length = dest.len(self)?;

        if length == 0 {
            // Nothing to copy... but let's still make sure that `dest` as a place is valid.
            self.get_place_alloc_mut(&dest)?;
        } else {
            // Write the src to the first element.
            let first = self.project_index(&dest, 0)?;
            self.copy_op(src, &first)?;

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
        &mut self,
        op: &mir::Operand<'tcx>,
        move_definitely_disjoint: bool,
    ) -> InterpResult<'tcx, FnArg<'tcx, M::Provenance>> {
        interp_ok(match op {
            mir::Operand::Copy(_) | mir::Operand::Constant(_) | mir::Operand::RuntimeChecks(_) => {
                // Make a regular copy.
                let op = self.eval_operand(op, None)?;
                FnArg::Copy(op)
            }
            mir::Operand::Move(place) => {
                // We will read from this place, which checks everything there is to check,
                // so we can skip the extra validity check here.
                let place =
                    self.eval_place(*place, /* skip_validity_for_simple_deref */ true)?;
                if move_definitely_disjoint {
                    // We still have to ensure that no *other* pointers are used to access this place,
                    // so *if* it is in memory then we have to treat it as `InPlace`.
                    // Use `place_to_op` to guarantee that we notice it being in memory.
                    let op = self.place_to_op(&place)?;
                    match op.as_mplace_or_imm() {
                        Either::Left(mplace) => FnArg::InPlace(mplace),
                        Either::Right(_imm) => FnArg::Copy(op),
                    }
                } else {
                    // We have to force this into memory to detect aliasing among `Move` arguments.
                    FnArg::InPlace(self.force_allocation(&place)?)
                }
            }
        })
    }

    /// Shared part of `Call` and `TailCall` implementation — finding and evaluating all the
    /// necessary information about callee and arguments to make a call.
    fn eval_callee_and_args(
        &mut self,
        terminator: &mir::Terminator<'tcx>,
        func: &mir::Operand<'tcx>,
        args: &[Spanned<mir::Operand<'tcx>>],
        dest: &mir::Place<'tcx>,
    ) -> InterpResult<'tcx, EvaluatedCalleeAndArgs<'tcx, M>> {
        let func = self.eval_operand(func, None)?;

        // Evaluating function call arguments. The tricky part here is dealing with `Move`
        // arguments: we have to ensure no two such arguments alias. This would be most easily done
        // by just forcing them all into memory and then doing the usual in-place argument
        // protection, but then we'd force *a lot* of arguments into memory. So we do some syntactic
        // pre-processing here where if all `move` arguments are syntactically distinct local
        // variables (and none is indirect), we can skip the in-memory forcing.
        // We have to include `dest` in that list so that we can detect aliasing of an in-place
        // argument with the return place.
        let move_definitely_disjoint = 'move_definitely_disjoint: {
            let mut previous_locals = FxHashSet::<mir::Local>::default();
            for place in args
                .iter()
                .filter_map(|a| {
                    // We only have to care about `Move` arguments.
                    if let mir::Operand::Move(place) = &a.node { Some(place) } else { None }
                })
                .chain(iter::once(dest))
            {
                if place.is_indirect_first_projection() {
                    // An indirect in-place argument could alias with anything else...
                    break 'move_definitely_disjoint false;
                }
                if !previous_locals.insert(place.local) {
                    // This local is the base for two arguments! They might overlap.
                    break 'move_definitely_disjoint false;
                }
            }
            // We found no violation so they are all definitely disjoint.
            true
        };
        let args = args
            .iter()
            .map(|arg| self.eval_fn_call_argument(&arg.node, move_definitely_disjoint))
            .collect::<InterpResult<'tcx, Vec<_>>>()?;

        let fn_sig_binder = {
            let _trace = enter_trace_span!(M, "fn_sig", ty = ?func.layout.ty.kind());
            func.layout.ty.fn_sig(*self.tcx)
        };
        let fn_sig = self.tcx.normalize_erasing_late_bound_regions(self.typing_env, fn_sig_binder);
        let extra_args = &args[fn_sig.inputs().len()..];
        let extra_args =
            self.tcx.mk_type_list_from_iter(extra_args.iter().map(|arg| arg.layout().ty));

        let (callee, fn_abi, with_caller_location) = match *func.layout.ty.kind() {
            ty::FnPtr(..) => {
                let fn_ptr = self.read_pointer(&func)?;
                let fn_val = self.get_ptr_fn(fn_ptr)?;
                (fn_val, Some(self.fn_abi_of_fn_ptr(fn_sig_binder, extra_args)?), false)
            }
            ty::FnDef(def_id, args) => {
                let instance = self.resolve(def_id, args.no_bound_vars().unwrap())?;
                // Don't compute FnAbi for LLVM intrinsics. Trying to do that would panic.
                // Rust intrinsics however *do* need a FnAbi as we may invoke the
                // fallback body like a regular function.
                let has_fn_abi = !matches!(instance.def, ty::InstanceKind::LlvmIntrinsic(_));
                (
                    FnVal::Instance(instance),
                    has_fn_abi
                        .then(|| self.fn_abi_of_instance_no_deduced_attrs(instance, extra_args))
                        .transpose()?,
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
        let _trace = enter_trace_span!(
            M,
            step::eval_terminator,
            terminator = ?terminator.kind,
            span = ?terminator.source_info.span,
            tracing_separate_thread = Empty,
        )
        .or_if_tracing_disabled(|| info!("{:?}", terminator.kind));

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

                // Evaluation order consistent with assignment: destination first.
                let dest_place =
                    self.eval_place(destination, /* skip_validity_for_simple_deref */ false)?;
                let EvaluatedCalleeAndArgs { callee, args, fn_sig, fn_abi, with_caller_location } =
                    self.eval_callee_and_args(terminator, func, args, &destination)?;

                self.init_fn_call(
                    callee,
                    (fn_sig.abi(), fn_abi),
                    &args,
                    with_caller_location,
                    &dest_place,
                    target,
                    if fn_abi.map_or(false, |fn_abi| fn_abi.can_unwind) {
                        unwind
                    } else {
                        mir::UnwindAction::Unreachable
                    },
                )?;

                // Sanity-check that `eval_fn_call` either pushed a new frame or
                // did a jump to another block. We disable the sanity check for functions that
                // can't return, since Miri sometimes does have to keep the location the same
                // for those (which is fine since execution will continue on a different thread).
                if target.is_some() && self.frame_idx() == old_stack && self.frame().loc == old_loc
                {
                    span_bug!(terminator.source_info.span, "evaluating this call made no progress");
                }
            }

            TailCall { ref func, ref args, fn_span: _ } => {
                let old_frame_idx = self.frame_idx();

                let EvaluatedCalleeAndArgs { callee, args, fn_sig, fn_abi, with_caller_location } =
                    self.eval_callee_and_args(terminator, func, args, &mir::Place::return_place())?;

                self.init_fn_tail_call(
                    callee,
                    (fn_sig.abi(), fn_abi),
                    &args,
                    with_caller_location,
                )?;

                if self.frame_idx() != old_frame_idx {
                    span_bug!(
                        terminator.source_info.span,
                        "evaluating this tail call pushed a new stack frame"
                    );
                }
            }

            Drop { place, target, unwind, replace: _, drop } => {
                assert!(
                    drop.is_none(),
                    "Async Drop must be expanded or reset to sync in runtime MIR"
                );
                let place =
                    self.eval_place(place, /* skip_validity_for_simple_deref */ false)?;
                let instance = {
                    let _trace =
                        enter_trace_span!(M, resolve::resolve_drop_glue, ty = ?place.layout.ty);
                    Instance::resolve_drop_glue(*self.tcx, place.layout.ty)
                };
                if let ty::InstanceKind::Shim(ty::ShimKind::DropGlue(_, None)) = instance.def {
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
