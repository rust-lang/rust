use rustc_middle::mir::{self, NonDivergingIntrinsic, StmtDebugInfo};
use rustc_middle::span_bug;
use tracing::instrument;

use super::{FunctionCx, LocalRef};
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
                    // For an rvalue like `&(_1.1)`, when `BackendRepr` is `BackendRepr::Memory`, we allocate a block of memory to this place.
                    // The place is an indirect pointer, we can refer to it directly.
                    LocalRef::Place(place_ref) => Some((place_ref, place.projection.as_slice())),
                    // For an rvalue like `&((*_1).1)`, we are calculating the address of `_1.1`.
                    // The deref projection is no-op here.
                    LocalRef::Operand(operand_ref) if place.is_indirect_first_projection() => {
                        Some((operand_ref.deref(bx.cx()), &place.projection[1..]))
                    }
                    // For an rvalue like `&1`, when `BackendRepr` is `BackendRepr::Scalar`,
                    // we cannot get the address.
                    // N.B. `non_ssa_locals` returns that this is an SSA local.
                    LocalRef::Operand(_) => None,
                    LocalRef::UnsizedPlace(_) | LocalRef::PendingOperand => None,
                }
                .filter(|(_, projection)| {
                    // Drop unsupported projections.
                    projection.iter().all(|p| p.can_use_in_debuginfo())
                });
                if let Some((base, projection)) = local_ref {
                    self.debug_new_val_to_local(bx, *dest, base, projection);
                } else {
                    // If the address cannot be calculated, use poison to indicate that the value has been optimized out.
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
