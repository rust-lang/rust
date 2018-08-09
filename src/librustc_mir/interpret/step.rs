//! This module contains the `EvalContext` methods for executing a single step of the interpreter.
//!
//! The main entry point is the `step` method.

use rustc::{mir, ty};
use rustc::ty::layout::LayoutOf;
use rustc::mir::interpret::{EvalResult, Scalar, Value};
use rustc_data_structures::indexed_vec::Idx;

use super::{EvalContext, Machine, PlaceExtra, ValTy};

impl<'a, 'mir, 'tcx, M: Machine<'mir, 'tcx>> EvalContext<'a, 'mir, 'tcx, M> {
    pub fn inc_step_counter_and_detect_loops(&mut self) -> EvalResult<'tcx, ()> {
        /// The number of steps between loop detector snapshots.
        /// Should be a power of two for performance reasons.
        const DETECTOR_SNAPSHOT_PERIOD: isize = 256;

        {
            let steps = &mut self.steps_since_detector_enabled;

            *steps += 1;
            if *steps < 0 {
                return Ok(());
            }

            *steps %= DETECTOR_SNAPSHOT_PERIOD;
            if *steps != 0 {
                return Ok(());
            }
        }

        if self.loop_detector.is_empty() {
            // First run of the loop detector

            // FIXME(#49980): make this warning a lint
            self.tcx.sess.span_warn(self.frame().span,
                "Constant evaluating a complex constant, this might take some time");
        }

        self.loop_detector.observe_and_analyze(&self.machine, &self.stack, &self.memory)
    }

    /// Returns true as long as there are more things to do.
    pub fn step(&mut self) -> EvalResult<'tcx, bool> {
        if self.stack.is_empty() {
            return Ok(false);
        }

        let block = self.frame().block;
        let stmt_id = self.frame().stmt;
        let mir = self.mir();
        let basic_block = &mir.basic_blocks()[block];

        let old_frames = self.cur_frame();

        if let Some(stmt) = basic_block.statements.get(stmt_id) {
            assert_eq!(old_frames, self.cur_frame());
            self.statement(stmt)?;
            return Ok(true);
        }

        self.inc_step_counter_and_detect_loops()?;

        let terminator = basic_block.terminator();
        assert_eq!(old_frames, self.cur_frame());
        self.terminator(terminator)?;
        Ok(true)
    }

    fn statement(&mut self, stmt: &mir::Statement<'tcx>) -> EvalResult<'tcx> {
        trace!("{:?}", stmt);

        use rustc::mir::StatementKind::*;

        // Some statements (e.g. box) push new stack frames.  We have to record the stack frame number
        // *before* executing the statement.
        let frame_idx = self.cur_frame();
        self.tcx.span = stmt.source_info.span;
        self.memory.tcx.span = stmt.source_info.span;

        match stmt.kind {
            Assign(ref place, ref rvalue) => self.eval_rvalue_into_place(rvalue, place)?,

            SetDiscriminant {
                ref place,
                variant_index,
            } => {
                let dest = self.eval_place(place)?;
                let dest_ty = self.place_ty(place);
                self.write_discriminant_value(dest_ty, dest, variant_index)?;
            }

            // Mark locals as alive
            StorageLive(local) => {
                let old_val = self.storage_live(local)?;
                self.deallocate_local(old_val)?;
            }

            // Mark locals as dead
            StorageDead(local) => {
                let old_val = self.frame_mut().storage_dead(local);
                self.deallocate_local(old_val)?;
            }

            // No dynamic semantics attached to `ReadForMatch`; MIR
            // interpreter is solely intended for borrowck'ed code.
            ReadForMatch(..) => {}

            // Validity checks.
            Validate(op, ref places) => {
                for operand in places {
                    M::validation_op(self, op, operand)?;
                }
            }
            EndRegion(ce) => {
                M::end_region(self, Some(ce))?;
            }

            UserAssertTy(..) => {}

            // Defined to do nothing. These are added by optimization passes, to avoid changing the
            // size of MIR constantly.
            Nop => {}

            InlineAsm { .. } => return err!(InlineAsm),
        }

        self.stack[frame_idx].stmt += 1;
        Ok(())
    }

    /// Evaluate an assignment statement.
    ///
    /// There is no separate `eval_rvalue` function. Instead, the code for handling each rvalue
    /// type writes its results directly into the memory specified by the place.
    fn eval_rvalue_into_place(
        &mut self,
        rvalue: &mir::Rvalue<'tcx>,
        place: &mir::Place<'tcx>,
    ) -> EvalResult<'tcx> {
        let dest = self.eval_place(place)?;
        let dest_ty = self.place_ty(place);
        let dest_layout = self.layout_of(dest_ty)?;

        use rustc::mir::Rvalue::*;
        match *rvalue {
            Use(ref operand) => {
                let value = self.eval_operand(operand)?.value;
                let valty = ValTy {
                    value,
                    ty: dest_ty,
                };
                self.write_value(valty, dest)?;
            }

            BinaryOp(bin_op, ref left, ref right) => {
                let left = self.eval_operand(left)?;
                let right = self.eval_operand(right)?;
                self.intrinsic_overflowing(
                    bin_op,
                    left,
                    right,
                    dest,
                    dest_ty,
                )?;
            }

            CheckedBinaryOp(bin_op, ref left, ref right) => {
                let left = self.eval_operand(left)?;
                let right = self.eval_operand(right)?;
                self.intrinsic_with_overflow(
                    bin_op,
                    left,
                    right,
                    dest,
                    dest_ty,
                )?;
            }

            UnaryOp(un_op, ref operand) => {
                let val = self.eval_operand_to_scalar(operand)?;
                let val = self.unary_op(un_op, val, dest_layout)?;
                self.write_scalar(
                    dest,
                    val,
                    dest_ty,
                )?;
            }

            Aggregate(ref kind, ref operands) => {
                let (dest, active_field_index) = match **kind {
                    mir::AggregateKind::Adt(adt_def, variant_index, _, active_field_index) => {
                        self.write_discriminant_value(dest_ty, dest, variant_index)?;
                        if adt_def.is_enum() {
                            (self.place_downcast(dest, variant_index)?, active_field_index)
                        } else {
                            (dest, active_field_index)
                        }
                    }
                    _ => (dest, None)
                };

                let layout = self.layout_of(dest_ty)?;
                for (i, operand) in operands.iter().enumerate() {
                    let value = self.eval_operand(operand)?;
                    // Ignore zero-sized fields.
                    if !self.layout_of(value.ty)?.is_zst() {
                        let field_index = active_field_index.unwrap_or(i);
                        let (field_dest, _) = self.place_field(dest, mir::Field::new(field_index), layout)?;
                        self.write_value(value, field_dest)?;
                    }
                }
            }

            Repeat(ref operand, _) => {
                let (elem_ty, length) = match dest_ty.sty {
                    ty::TyArray(elem_ty, n) => (elem_ty, n.unwrap_usize(self.tcx.tcx)),
                    _ => {
                        bug!(
                            "tried to assign array-repeat to non-array type {:?}",
                            dest_ty
                        )
                    }
                };
                let elem_size = self.layout_of(elem_ty)?.size;
                let value = self.eval_operand(operand)?.value;

                let (dest, dest_align) = self.force_allocation(dest)?.to_ptr_align();

                if length > 0 {
                    let dest = dest.unwrap_or_err()?;
                    //write the first value
                    self.write_value_to_ptr(value, dest, dest_align, elem_ty)?;

                    if length > 1 {
                        let rest = dest.ptr_offset(elem_size * 1 as u64, &self)?;
                        self.memory.copy_repeatedly(dest, dest_align, rest, dest_align, elem_size, length - 1, false)?;
                    }
                }
            }

            Len(ref place) => {
                // FIXME(CTFE): don't allow computing the length of arrays in const eval
                let src = self.eval_place(place)?;
                let ty = self.place_ty(place);
                let (_, len) = src.elem_ty_and_len(ty, self.tcx.tcx);
                let size = self.memory.pointer_size().bytes() as u8;
                self.write_scalar(
                    dest,
                    Scalar::Bits {
                        bits: len as u128,
                        size,
                    },
                    dest_ty,
                )?;
            }

            Ref(_, _, ref place) => {
                let src = self.eval_place(place)?;
                // We ignore the alignment of the place here -- special handling for packed structs ends
                // at the `&` operator.
                let (ptr, _align, extra) = self.force_allocation(src)?.to_ptr_align_extra();

                let val = match extra {
                    PlaceExtra::None => Value::Scalar(ptr),
                    PlaceExtra::Length(len) => ptr.to_value_with_len(len, self.tcx.tcx),
                    PlaceExtra::Vtable(vtable) => ptr.to_value_with_vtable(vtable),
                    PlaceExtra::DowncastVariant(..) => {
                        bug!("attempted to take a reference to an enum downcast place")
                    }
                };
                let valty = ValTy {
                    value: val,
                    ty: dest_ty,
                };
                self.write_value(valty, dest)?;
            }

            NullaryOp(mir::NullOp::Box, ty) => {
                let ty = self.monomorphize(ty, self.substs());
                M::box_alloc(self, ty, dest)?;
            }

            NullaryOp(mir::NullOp::SizeOf, ty) => {
                let ty = self.monomorphize(ty, self.substs());
                let layout = self.layout_of(ty)?;
                assert!(!layout.is_unsized(),
                        "SizeOf nullary MIR operator called for unsized type");
                let size = self.memory.pointer_size().bytes() as u8;
                self.write_scalar(
                    dest,
                    Scalar::Bits {
                        bits: layout.size.bytes() as u128,
                        size,
                    },
                    dest_ty,
                )?;
            }

            Cast(kind, ref operand, cast_ty) => {
                debug_assert_eq!(self.monomorphize(cast_ty, self.substs()), dest_ty);
                let src = self.eval_operand(operand)?;
                self.cast(src, kind, dest_ty, dest)?;
            }

            Discriminant(ref place) => {
                let ty = self.place_ty(place);
                let layout = self.layout_of(ty)?;
                let place = self.eval_place(place)?;
                let discr_val = self.read_discriminant_value(place, layout)?;
                let size = self.layout_of(dest_ty).unwrap().size.bytes() as u8;
                self.write_scalar(dest, Scalar::Bits {
                    bits: discr_val,
                    size,
                }, dest_ty)?;
            }
        }

        self.dump_local(dest);

        Ok(())
    }

    fn terminator(&mut self, terminator: &mir::Terminator<'tcx>) -> EvalResult<'tcx> {
        trace!("{:?}", terminator.kind);
        self.tcx.span = terminator.source_info.span;
        self.memory.tcx.span = terminator.source_info.span;
        self.eval_terminator(terminator)?;
        if !self.stack.is_empty() {
            trace!("// {:?}", self.frame().block);
        }
        Ok(())
    }
}
