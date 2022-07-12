use rustc_middle::mir;
use rustc_middle::mir::NonDivergingIntrinsic;

use super::FunctionCx;
use super::LocalRef;
use crate::traits::BuilderMethods;
use crate::traits::*;

impl<'a, 'tcx, Bx: BuilderMethods<'a, 'tcx>> FunctionCx<'a, 'tcx, Bx> {
    #[instrument(level = "debug", skip(self, bx))]
    pub fn codegen_statement(&mut self, mut bx: Bx, statement: &mir::Statement<'tcx>) -> Bx {
        self.set_debug_loc(&mut bx, statement.source_info);
        match statement.kind {
            mir::StatementKind::Assign(box (ref place, ref rvalue)) => {
                if let Some(index) = place.as_local() {
                    match self.locals[index] {
                        LocalRef::Place(cg_dest) => self.codegen_rvalue(bx, cg_dest, rvalue),
                        LocalRef::UnsizedPlace(cg_indirect_dest) => {
                            self.codegen_rvalue_unsized(bx, cg_indirect_dest, rvalue)
                        }
                        LocalRef::Operand(None) => {
                            let (mut bx, operand) = self.codegen_rvalue_operand(bx, rvalue);
                            self.locals[index] = LocalRef::Operand(Some(operand));
                            self.debug_introduce_local(&mut bx, index);
                            bx
                        }
                        LocalRef::Operand(Some(op)) => {
                            if !op.layout.is_zst() {
                                span_bug!(
                                    statement.source_info.span,
                                    "operand {:?} already assigned",
                                    rvalue
                                );
                            }

                            // If the type is zero-sized, it's already been set here,
                            // but we still need to make sure we codegen the operand
                            self.codegen_rvalue_operand(bx, rvalue).0
                        }
                    }
                } else {
                    let cg_dest = self.codegen_place(&mut bx, place.as_ref());
                    self.codegen_rvalue(bx, cg_dest, rvalue)
                }
            }
            mir::StatementKind::SetDiscriminant { box ref place, variant_index } => {
                self.codegen_place(&mut bx, place.as_ref())
                    .codegen_set_discr(&mut bx, variant_index);
                bx
            }
            mir::StatementKind::Deinit(..) => {
                // For now, don't codegen this to anything. In the future it may be worth
                // experimenting with what kind of information we can emit to LLVM without hurting
                // perf here
                bx
            }
            mir::StatementKind::StorageLive(local) => {
                if let LocalRef::Place(cg_place) = self.locals[local] {
                    cg_place.storage_live(&mut bx);
                } else if let LocalRef::UnsizedPlace(cg_indirect_place) = self.locals[local] {
                    cg_indirect_place.storage_live(&mut bx);
                }
                bx
            }
            mir::StatementKind::StorageDead(local) => {
                if let LocalRef::Place(cg_place) = self.locals[local] {
                    cg_place.storage_dead(&mut bx);
                } else if let LocalRef::UnsizedPlace(cg_indirect_place) = self.locals[local] {
                    cg_indirect_place.storage_dead(&mut bx);
                }
                bx
            }
            mir::StatementKind::Coverage(box ref coverage) => {
                self.codegen_coverage(&mut bx, coverage.clone(), statement.source_info.scope);
                bx
            }
            mir::StatementKind::Intrinsic(box NonDivergingIntrinsic::Assume(ref op)) => {
                let op_val = self.codegen_operand(&mut bx, op);
                bx.assume(op_val.immediate());
                bx
            }
            mir::StatementKind::Intrinsic(box NonDivergingIntrinsic::CopyNonOverlapping(
                mir::CopyNonOverlapping { ref count, ref src, ref dst },
            )) => {
                let dst_val = self.codegen_operand(&mut bx, dst);
                let src_val = self.codegen_operand(&mut bx, src);
                let count = self.codegen_operand(&mut bx, count).immediate();
                let pointee_layout = dst_val
                    .layout
                    .pointee_info_at(&bx, rustc_target::abi::Size::ZERO)
                    .expect("Expected pointer");
                let bytes = bx.mul(count, bx.const_usize(pointee_layout.size.bytes()));

                let align = pointee_layout.align;
                let dst = dst_val.immediate();
                let src = src_val.immediate();
                bx.memcpy(dst, align, src, align, bytes, crate::MemFlags::empty());
                bx
            }
            mir::StatementKind::FakeRead(..)
            | mir::StatementKind::Retag { .. }
            | mir::StatementKind::AscribeUserType(..)
            | mir::StatementKind::Nop => bx,
        }
    }
}
