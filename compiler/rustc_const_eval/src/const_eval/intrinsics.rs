use rustc_abi::{Align, Size};
use rustc_middle::ty::layout::{HasTypingEnv, ValidityRequirement};
use rustc_middle::ty::{self, FieldInfo};
use rustc_middle::{mir, throw_machine_stop};
use rustc_span::{Symbol, bug, span_bug, sym};

use super::error::*;
use crate::const_eval::{CompileTimeMachine, MemoryKind};
use crate::interpret::{
    self, AllocInit, CtfeProvenance, InterpCx, InterpResult, OpTy, PlaceTy, Scalar,
    ensure_monomorphic_enough, err_inval, interp_ok, throw_ub_format,
};

impl<'tcx> InterpCx<'tcx, CompileTimeMachine<'tcx>> {
    pub(crate) fn call_ctfe_intrinsic(
        &mut self,
        intrinsic_name: Symbol,
        instance: ty::Instance<'tcx>,
        args: &[OpTy<'tcx>],
        dest: &PlaceTy<'tcx, CtfeProvenance>,
        target: Option<mir::BasicBlock>,
    ) -> InterpResult<'tcx, bool> {
        match intrinsic_name {
            sym::abort_immediate => {
                // Note that `abort_immediate` is also hooked separately in Miri.
                throw_machine_stop!(ConstEvalErrKind::Abort);
            }
            sym::ptr_guaranteed_cmp => {
                let a = self.read_scalar(&args[0])?;
                let b = self.read_scalar(&args[1])?;
                let cmp = self.guaranteed_cmp(a, b)?;
                self.write_scalar(Scalar::from_u8(cmp), dest)?;
            }
            sym::const_allocate => {
                let size = self.read_scalar(&args[0])?.to_target_usize(self)?;
                let align = self.read_scalar(&args[1])?.to_target_usize(self)?;

                let align = match Align::from_bytes(align) {
                    Ok(a) => a,
                    Err(err) => {
                        throw_ub_format!("invalid align passed to `const_allocate`: {err}")
                    }
                };

                let ptr = self.allocate_ptr(
                    Size::from_bytes(size),
                    align,
                    interpret::MemoryKind::Machine(MemoryKind::Heap { was_made_global: false }),
                    AllocInit::Uninit,
                )?;
                self.write_pointer(ptr, dest)?;
            }
            sym::const_deallocate => {
                let ptr = self.read_pointer(&args[0])?;
                let size = self.read_scalar(&args[1])?.to_target_usize(self)?;
                let align = self.read_scalar(&args[2])?.to_target_usize(self)?;

                let size = Size::from_bytes(size);
                let align = match Align::from_bytes(align) {
                    Ok(a) => a,
                    Err(err) => {
                        throw_ub_format!("invalid align passed to `const_deallocate`: {err}")
                    }
                };

                // If an allocation is created in an another const,
                // we don't deallocate it.
                let (alloc_id, _, _) = self.ptr_get_alloc_id(ptr, 0)?;
                let is_allocated_in_another_const = matches!(
                    self.tcx.try_get_global_alloc(alloc_id),
                    Some(interpret::GlobalAlloc::Memory(_))
                );

                if !is_allocated_in_another_const {
                    self.deallocate_ptr(
                        ptr,
                        Some((size, align)),
                        interpret::MemoryKind::Machine(MemoryKind::Heap { was_made_global: false }),
                    )?;
                }
            }

            sym::const_make_global => {
                let ptr = self.read_pointer(&args[0])?;
                self.make_const_heap_ptr_global(ptr)?;
                self.write_pointer(ptr, dest)?;
            }

            // The intrinsic represents whether the value is known to the optimizer (LLVM).
            // We're not doing any optimizations here, so there is no optimizer that could know the value.
            // (We know the value here in the machine of course, but this is the runtime of that code,
            // not the optimization stage.)
            sym::is_val_statically_known => self.write_scalar(Scalar::from_bool(false), dest)?,

            // We handle these here since Miri does not want to have them.
            sym::assert_inhabited
            | sym::assert_zero_valid
            | sym::assert_mem_uninitialized_valid => {
                let ty = instance.args.type_at(0);
                let requirement =
                    ValidityRequirement::from_rustc_intrinsic(intrinsic_name).unwrap();

                let should_panic = !self
                    .tcx
                    .check_validity_requirement((requirement, self.typing_env().as_query_input(ty)))
                    .map_err(|_| err_inval!(TooGeneric))?;

                if should_panic {
                    let layout = self.layout_of(ty)?;

                    let msg = Symbol::intern(&match requirement {
                        // For *all* intrinsics we first check `is_uninhabited` to give a more specific
                        // error message.
                        _ if layout.is_uninhabited() => format!(
                            "aborted execution: attempted to instantiate uninhabited type `{ty}`"
                        ),
                        ValidityRequirement::Inhabited => bug!("handled earlier"),
                        ValidityRequirement::Zero => format!(
                            "aborted execution: attempted to zero-initialize type `{ty}`, which is invalid"
                        ),
                        ValidityRequirement::UninitMitigated0x01Fill => format!(
                            "aborted execution: attempted to leave type `{ty}` uninitialized, which is invalid"
                        ),
                        ValidityRequirement::Uninit => bug!("assert_uninit_valid doesn't exist"),
                    });

                    let span = self.caller_location();
                    let (file, line, col) = self.location_triple_for_span(span);

                    InterpResult::from(Err::<(), _>(ConstEvalErrKind::Panic {
                        msg,
                        file,
                        line,
                        col,
                    }))?;

                    // Skip the `return_to_block` at the end (we panicked, we do not return).
                    return interp_ok(true);
                }
            }

            sym::field_offset => {
                let frt_ty = instance.args.type_at(0);
                ensure_monomorphic_enough(frt_ty)?;

                let (ty, variant, field) = if let ty::Adt(def, args) = frt_ty.kind()
                    && let Some(FieldInfo { base, variant_idx, field_idx, .. }) =
                        def.field_representing_type_info(self.tcx.tcx, *args)
                {
                    (base, variant_idx, field_idx)
                } else {
                    span_bug!(self.cur_span(), "expected field representing type, got {frt_ty}")
                };
                let layout = self.layout_of(ty)?;
                let cx = ty::layout::LayoutCx::new(self.tcx.tcx, self.typing_env());

                let layout = layout.for_variant(&cx, variant);
                let offset = layout.fields.offset(field.index()).bytes();

                self.write_scalar(Scalar::from_target_usize(offset, self), dest)?;
            }

            _ => {
                return interp_ok(false);
            }
        }

        // Intrinsic is done, jump to next block.
        self.return_to_block(target)?;
        interp_ok(true)
    }
}
