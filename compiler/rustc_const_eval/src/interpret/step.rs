//! This module contains the `InterpCx` methods for executing a single step of the interpreter.
//!
//! The main entry point is the `step` method.

use rustc_middle::mir;
use rustc_middle::mir::interpret::{InterpResult, Scalar};
use rustc_middle::ty::layout::LayoutOf;

use super::{InterpCx, Machine};

/// Classify whether an operator is "left-homogeneous", i.e., the LHS has the
/// same type as the result.
#[inline]
fn binop_left_homogeneous(op: mir::BinOp) -> bool {
    use rustc_middle::mir::BinOp::*;
    match op {
        Add | Sub | Mul | Div | Rem | BitXor | BitAnd | BitOr | Offset | Shl | Shr => true,
        Eq | Ne | Lt | Le | Gt | Ge => false,
    }
}
/// Classify whether an operator is "right-homogeneous", i.e., the RHS has the
/// same type as the LHS.
#[inline]
fn binop_right_homogeneous(op: mir::BinOp) -> bool {
    use rustc_middle::mir::BinOp::*;
    match op {
        Add | Sub | Mul | Div | Rem | BitXor | BitAnd | BitOr | Eq | Ne | Lt | Le | Gt | Ge => true,
        Offset | Shl | Shr => false,
    }
}

impl<'mir, 'tcx: 'mir, M: Machine<'mir, 'tcx>> InterpCx<'mir, 'tcx, M> {
    pub fn run(&mut self) -> InterpResult<'tcx> {
        while self.step()? {}
        Ok(())
    }

    /// Returns `true` as long as there are more things to do.
    ///
    /// This is used by [priroda](https://github.com/oli-obk/priroda)
    ///
    /// This is marked `#inline(always)` to work around adverserial codegen when `opt-level = 3`
    #[inline(always)]
    pub fn step(&mut self) -> InterpResult<'tcx, bool> {
        if self.stack().is_empty() {
            return Ok(false);
        }

        let loc = match self.frame().loc {
            Ok(loc) => loc,
            Err(_) => {
                // We are unwinding and this fn has no cleanup code.
                // Just go on unwinding.
                trace!("unwinding: skipping frame");
                self.pop_stack_frame(/* unwinding */ true)?;
                return Ok(true);
            }
        };
        let basic_block = &self.body().basic_blocks()[loc.block];

        let old_frames = self.frame_idx();

        if let Some(stmt) = basic_block.statements.get(loc.statement_index) {
            assert_eq!(old_frames, self.frame_idx());
            self.statement(stmt)?;
            return Ok(true);
        }

        M::before_terminator(self)?;

        let terminator = basic_block.terminator();
        assert_eq!(old_frames, self.frame_idx());
        self.terminator(terminator)?;
        Ok(true)
    }

    /// Runs the interpretation logic for the given `mir::Statement` at the current frame and
    /// statement counter. This also moves the statement counter forward.
    pub fn statement(&mut self, stmt: &mir::Statement<'tcx>) -> InterpResult<'tcx> {
        info!("{:?}", stmt);

        use rustc_middle::mir::StatementKind::*;

        // Some statements (e.g., box) push new stack frames.
        // We have to record the stack frame number *before* executing the statement.
        let frame_idx = self.frame_idx();

        match &stmt.kind {
            Assign(box (place, rvalue)) => self.eval_rvalue_into_place(rvalue, *place)?,

            SetDiscriminant { place, variant_index } => {
                let dest = self.eval_place(**place)?;
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

            // Stacked Borrows.
            Retag(kind, place) => {
                let dest = self.eval_place(**place)?;
                M::retag(self, *kind, &dest)?;
            }

            // Call CopyNonOverlapping
            CopyNonOverlapping(box rustc_middle::mir::CopyNonOverlapping { src, dst, count }) => {
                let src = self.eval_operand(src, None)?;
                let dst = self.eval_operand(dst, None)?;
                let count = self.eval_operand(count, None)?;
                self.copy_intrinsic(&src, &dst, &count, /* nonoverlapping */ true)?;
            }

            // Statements we do not track.
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

            // Defined to do nothing. These are added by optimization passes, to avoid changing the
            // size of MIR constantly.
            Nop => {}

            LlvmInlineAsm { .. } => throw_unsup_format!("inline assembly is not supported"),
        }

        self.stack_mut()[frame_idx].loc.as_mut().unwrap().statement_index += 1;
        Ok(())
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

        use rustc_middle::mir::Rvalue::*;
        match *rvalue {
            ThreadLocalRef(did) => {
                let ptr = M::thread_local_static_base_pointer(self, did)?;
                self.write_pointer(ptr, &dest)?;
            }

            Use(ref operand) => {
                // Avoid recomputing the layout
                let op = self.eval_operand(operand, Some(dest.layout))?;
                self.copy_op(&op, &dest)?;
            }

            BinaryOp(bin_op, box (ref left, ref right)) => {
                let layout = binop_left_homogeneous(bin_op).then_some(dest.layout);
                let left = self.read_immediate(&self.eval_operand(left, layout)?)?;
                let layout = binop_right_homogeneous(bin_op).then_some(left.layout);
                let right = self.read_immediate(&self.eval_operand(right, layout)?)?;
                self.binop_ignore_overflow(bin_op, &left, &right, &dest)?;
            }

            CheckedBinaryOp(bin_op, box (ref left, ref right)) => {
                // Due to the extra boolean in the result, we can never reuse the `dest.layout`.
                let left = self.read_immediate(&self.eval_operand(left, None)?)?;
                let layout = binop_right_homogeneous(bin_op).then_some(left.layout);
                let right = self.read_immediate(&self.eval_operand(right, layout)?)?;
                self.binop_with_overflow(bin_op, &left, &right, &dest)?;
            }

            UnaryOp(un_op, ref operand) => {
                // The operand always has the same type as the result.
                let val = self.read_immediate(&self.eval_operand(operand, Some(dest.layout))?)?;
                let val = self.unary_op(un_op, &val)?;
                assert_eq!(val.layout, dest.layout, "layout mismatch for result of {:?}", un_op);
                self.write_immediate(*val, &dest)?;
            }

            Aggregate(ref kind, ref operands) => {
                // active_field_index is for union initialization.
                let (dest, active_field_index) = match **kind {
                    mir::AggregateKind::Adt(adt_did, variant_index, _, _, active_field_index) => {
                        self.write_discriminant(variant_index, &dest)?;
                        if self.tcx.adt_def(adt_did).is_enum() {
                            assert!(active_field_index.is_none());
                            (self.place_downcast(&dest, variant_index)?, None)
                        } else {
                            if active_field_index.is_some() {
                                assert_eq!(operands.len(), 1);
                            }
                            (dest, active_field_index)
                        }
                    }
                    _ => (dest, None),
                };

                for (i, operand) in operands.iter().enumerate() {
                    let op = self.eval_operand(operand, None)?;
                    let field_index = active_field_index.unwrap_or(i);
                    let field_dest = self.place_field(&dest, field_index)?;
                    self.copy_op(&op, &field_dest)?;
                }
            }

            Repeat(ref operand, _) => {
                let src = self.eval_operand(operand, None)?;
                assert!(!src.layout.is_unsized());
                let dest = self.force_allocation(&dest)?;
                let length = dest.len(self)?;

                if length == 0 {
                    // Nothing to copy... but let's still make sure that `dest` as a place is valid.
                    self.get_alloc_mut(&dest)?;
                } else {
                    // Write the src to the first element.
                    let first = self.mplace_field(&dest, 0)?;
                    self.copy_op(&src, &first.into())?;

                    // This is performance-sensitive code for big static/const arrays! So we
                    // avoid writing each operand individually and instead just make many copies
                    // of the first element.
                    let elem_size = first.layout.size;
                    let first_ptr = first.ptr;
                    let rest_ptr = first_ptr.offset(elem_size, self)?;
                    // For the alignment of `rest_ptr`, we crucially do *not* use `first.align` as
                    // that place might be more aligned than its type mandates (a `u8` array could
                    // be 4-aligned if it sits at the right spot in a struct). Instead we use
                    // `first.layout.align`, i.e., the alignment given by the type.
                    self.memory.copy_repeatedly(
                        first_ptr,
                        first.align,
                        rest_ptr,
                        first.layout.align.abi,
                        elem_size,
                        length - 1,
                        /*nonoverlapping:*/ true,
                    )?;
                }
            }

            Len(place) => {
                let src = self.eval_place(place)?;
                let mplace = self.force_allocation(&src)?;
                let len = mplace.len(self)?;
                self.write_scalar(Scalar::from_machine_usize(len, self), &dest)?;
            }

            AddressOf(_, place) | Ref(_, _, place) => {
                let src = self.eval_place(place)?;
                let place = self.force_allocation(&src)?;
                self.write_immediate(place.to_ref(self), &dest)?;
            }

            NullaryOp(null_op, ty) => {
                let ty = self.subst_from_current_frame_and_normalize_erasing_regions(ty)?;
                let layout = self.layout_of(ty)?;
                if layout.is_unsized() {
                    // FIXME: This should be a span_bug (#80742)
                    self.tcx.sess.delay_span_bug(
                        self.frame().current_span(),
                        &format!("Nullary MIR operator called for unsized type {}", ty),
                    );
                    throw_inval!(SizeOfUnsizedType(ty));
                }
                let val = match null_op {
                    mir::NullOp::SizeOf => layout.size.bytes(),
                    mir::NullOp::AlignOf => layout.align.abi.bytes(),
                };
                self.write_scalar(Scalar::from_machine_usize(val, self), &dest)?;
            }

            ShallowInitBox(ref operand, _) => {
                let src = self.eval_operand(operand, None)?;
                let v = self.read_immediate(&src)?;
                self.write_immediate(*v, &dest)?;
            }

            Cast(cast_kind, ref operand, cast_ty) => {
                let src = self.eval_operand(operand, None)?;
                let cast_ty =
                    self.subst_from_current_frame_and_normalize_erasing_regions(cast_ty)?;
                self.cast(&src, cast_kind, cast_ty, &dest)?;
            }

            Discriminant(place) => {
                let op = self.eval_place_to_op(place, None)?;
                let discr_val = self.read_discriminant(&op)?.0;
                self.write_scalar(discr_val, &dest)?;
            }
        }

        trace!("{:?}", self.dump_place(*dest));

        Ok(())
    }

    fn terminator(&mut self, terminator: &mir::Terminator<'tcx>) -> InterpResult<'tcx> {
        info!("{:?}", terminator.kind);

        self.eval_terminator(terminator)?;
        if !self.stack().is_empty() {
            if let Ok(loc) = self.frame().loc {
                info!("// executing {:?}", loc.block);
            }
        }
        Ok(())
    }
}
