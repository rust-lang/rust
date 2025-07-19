use rustc_middle::mir::{self, NonDivergingIntrinsic, RETURN_PLACE, StmtDebugInfo};
use rustc_middle::{bug, span_bug};
use rustc_target::callconv::PassMode;
use tracing::instrument;

use super::{FunctionCx, LocalRef};
use crate::common::TypeKind;
use crate::mir::place::PlaceRef;
use crate::traits::*;

impl<'a, 'tcx, Bx: BuilderMethods<'a, 'tcx>> FunctionCx<'a, 'tcx, Bx> {
    #[instrument(level = "debug", skip(self, bx))]
    pub(crate) fn codegen_statement(&mut self, bx: &mut Bx, statement: &mir::Statement<'tcx>) {
        self.codegen_stmt_debuginfos(bx, &statement.debuginfos);
        self.set_debug_loc(bx, statement.source_info);
        match statement.kind {
            mir::StatementKind::Assign(box (ref place, ref rvalue)) => {
                if let Some(index) = place.as_local() {
                    match self.locals[index] {
                        LocalRef::Place(cg_dest) => self.codegen_rvalue(bx, cg_dest, rvalue),
                        LocalRef::UnsizedPlace(cg_indirect_dest) => {
                            let ty = cg_indirect_dest.layout.ty;
                            span_bug!(
                                statement.source_info.span,
                                "cannot reallocate from `UnsizedPlace({ty})` \
                                into `{rvalue:?}`; dynamic alloca is not supported",
                            );
                        }
                        LocalRef::PendingOperand => {
                            let operand = self.codegen_rvalue_operand(bx, rvalue);
                            self.overwrite_local(index, LocalRef::Operand(operand));
                            self.debug_introduce_local(bx, index);
                        }
                        LocalRef::Operand(op) => {
                            if !op.layout.is_zst() {
                                span_bug!(
                                    statement.source_info.span,
                                    "operand {:?} already assigned",
                                    rvalue
                                );
                            }

                            // If the type is zero-sized, it's already been set here,
                            // but we still need to make sure we codegen the operand
                            self.codegen_rvalue_operand(bx, rvalue);
                        }
                    }
                } else {
                    let cg_dest = self.codegen_place(bx, place.as_ref());
                    self.codegen_rvalue(bx, cg_dest, rvalue);
                }
            }
            mir::StatementKind::SetDiscriminant { box ref place, variant_index } => {
                self.codegen_place(bx, place.as_ref()).codegen_set_discr(bx, variant_index);
            }
            mir::StatementKind::Deinit(..) => {
                // For now, don't codegen this to anything. In the future it may be worth
                // experimenting with what kind of information we can emit to LLVM without hurting
                // perf here
            }
            mir::StatementKind::StorageLive(local) => {
                if let LocalRef::Place(cg_place) = self.locals[local] {
                    cg_place.storage_live(bx);
                } else if let LocalRef::UnsizedPlace(cg_indirect_place) = self.locals[local] {
                    cg_indirect_place.storage_live(bx);
                }
            }
            mir::StatementKind::StorageDead(local) => {
                if let LocalRef::Place(cg_place) = self.locals[local] {
                    cg_place.storage_dead(bx);
                } else if let LocalRef::UnsizedPlace(cg_indirect_place) = self.locals[local] {
                    cg_indirect_place.storage_dead(bx);
                }
            }
            mir::StatementKind::Coverage(ref kind) => {
                self.codegen_coverage(bx, kind, statement.source_info.scope);
            }
            mir::StatementKind::Intrinsic(box NonDivergingIntrinsic::Assume(ref op)) => {
                let op_val = self.codegen_operand(bx, op);
                bx.assume(op_val.immediate());
            }
            mir::StatementKind::Intrinsic(box NonDivergingIntrinsic::CopyNonOverlapping(
                mir::CopyNonOverlapping { ref count, ref src, ref dst },
            )) => {
                let dst_val = self.codegen_operand(bx, dst);
                let src_val = self.codegen_operand(bx, src);
                let count = self.codegen_operand(bx, count).immediate();
                let pointee_layout = dst_val
                    .layout
                    .pointee_info_at(bx, rustc_abi::Size::ZERO)
                    .expect("Expected pointer");
                let bytes = bx.mul(count, bx.const_usize(pointee_layout.size.bytes()));

                let align = pointee_layout.align;
                let dst = dst_val.immediate();
                let src = src_val.immediate();
                bx.memcpy(dst, align, src, align, bytes, crate::MemFlags::empty(), None);
            }
            mir::StatementKind::FakeRead(..)
            | mir::StatementKind::Retag { .. }
            | mir::StatementKind::AscribeUserType(..)
            | mir::StatementKind::ConstEvalCounter
            | mir::StatementKind::PlaceMention(..)
            | mir::StatementKind::BackwardIncompatibleDropHint { .. }
            | mir::StatementKind::Nop => {}
        }
    }

    pub(crate) fn codegen_stmt_debuginfo(&mut self, bx: &mut Bx, debuginfo: &StmtDebugInfo<'tcx>) {
        match debuginfo {
            StmtDebugInfo::AssignRef(dest, place) => {
                let local_ref = match self.locals[place.local] {
                    LocalRef::Place(place_ref) | LocalRef::UnsizedPlace(place_ref) => {
                        Some(place_ref)
                    }
                    LocalRef::Operand(operand_ref) => operand_ref
                        .val
                        .try_pointer_parts()
                        .map(|(pointer, _)| PlaceRef::new_sized(pointer, operand_ref.layout)),
                    LocalRef::PendingOperand => None,
                }
                .filter(|place_ref| {
                    // For the reference of an argument (e.x. `&_1`), it's only valid if the pass mode is indirect, and its reference is
                    // llval.
                    let local_ref_pass_mode = place.as_local().and_then(|local| {
                        if local == RETURN_PLACE {
                            None
                        } else {
                            self.fn_abi.args.get(local.as_usize() - 1).map(|arg| &arg.mode)
                        }
                    });
                    matches!(local_ref_pass_mode, Some(&PassMode::Indirect {..}) | None) &&
                    // Drop unsupported projections.
                    place.projection.iter().all(|p| p.can_use_in_debuginfo()) &&
                    // Only pointers can be calculated addresses.
                    bx.type_kind(bx.val_ty(place_ref.val.llval)) == TypeKind::Pointer
                });
                if let Some(local_ref) = local_ref {
                    let (base_layout, projection) = if place.is_indirect_first_projection() {
                        // For `_n = &((*_1).0: i32);`, we are calculating the address of `_1.0`, so
                        // we should drop the deref projection.
                        let projected_ty = local_ref
                            .layout
                            .ty
                            .builtin_deref(true)
                            .unwrap_or_else(|| bug!("deref of non-pointer {:?}", local_ref));
                        let layout = bx.cx().layout_of(projected_ty);
                        (layout, &place.projection[1..])
                    } else {
                        (local_ref.layout, place.projection.as_slice())
                    };
                    self.debug_new_val_to_local(bx, *dest, local_ref.val, base_layout, projection);
                } else {
                    // If the address cannot be computed, use poison to indicate that the value has been optimized out.
                    self.debug_poison_to_local(bx, *dest);
                }
            }
            StmtDebugInfo::InvalidAssign(local) => {
                self.debug_poison_to_local(bx, *local);
            }
        }
    }

    pub(crate) fn codegen_stmt_debuginfos(
        &mut self,
        bx: &mut Bx,
        debuginfos: &[StmtDebugInfo<'tcx>],
    ) {
        for debuginfo in debuginfos {
            self.codegen_stmt_debuginfo(bx, debuginfo);
        }
    }
}
