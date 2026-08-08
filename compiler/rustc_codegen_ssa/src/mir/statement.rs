use rustc_middle::mir::{self, NonDivergingIntrinsic};
use rustc_middle::{bug, span_bug, ty};
use tracing::instrument;

use super::{FunctionCx, LocalRef};
use crate::mir::retag;
use crate::traits::*;

impl<'a, 'tcx, Bx: BuilderMethods<'a, 'tcx>> FunctionCx<'a, 'tcx, Bx> {
    #[instrument(level = "debug", skip(self, bx))]
    pub(crate) fn codegen_statement(&mut self, bx: &mut Bx, statement: &mir::Statement<'tcx>) {
        self.codegen_stmt_debuginfos(bx, &statement.debuginfos);
        self.set_debug_loc(bx, statement.source_info);
        match statement.kind {
            mir::StatementKind::Assign((ref place, ref rvalue)) => {
                let needs_retag = bx.tcx().sess.opts.unstable_opts.codegen_emit_retag.is_some()
                    && retag::rvalue_needs_retag(rvalue);

                if let Some(index) = place.as_local() {
                    match self.locals[index] {
                        LocalRef::Place(cg_dest) => {
                            self.codegen_rvalue(bx, cg_dest, rvalue);
                            if needs_retag {
                                self.codegen_retag_place(bx, cg_dest, false);
                            }
                        }
                        LocalRef::UnsizedPlace(cg_indirect_dest) => {
                            let ty = cg_indirect_dest.layout.ty;
                            span_bug!(
                                statement.source_info.span,
                                "cannot reallocate from `UnsizedPlace({ty})` \
                                into `{rvalue:?}`; dynamic alloca is not supported",
                            );
                        }
                        LocalRef::PendingOperand => {
                            let mut operand = self.codegen_rvalue_operand(bx, rvalue);
                            if needs_retag {
                                operand = self.codegen_retag_operand(bx, operand, false);
                            }
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
                            // and emit a retag.
                            let operand = self.codegen_rvalue_operand(bx, rvalue);
                            if needs_retag {
                                self.codegen_retag_operand(bx, operand, false);
                            }
                        }
                    }
                } else {
                    let cg_dest = self.codegen_place(bx, place.as_ref());
                    self.codegen_rvalue(bx, cg_dest, rvalue);
                    if needs_retag {
                        self.codegen_retag_place(bx, cg_dest, false);
                    }
                }
            }
            mir::StatementKind::SetDiscriminant { ref place, variant_index } => {
                self.codegen_place(bx, (**place).as_ref()).codegen_set_discr(bx, variant_index);
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
            mir::StatementKind::Intrinsic(NonDivergingIntrinsic::Assume(ref op)) => {
                let op_val = self.codegen_operand(bx, op);
                bx.assume(op_val.immediate());
            }
            mir::StatementKind::Intrinsic(NonDivergingIntrinsic::CopyNonOverlapping(
                mir::CopyNonOverlapping { ref count, ref src, ref dst },
            )) => {
                let dst_val = self.codegen_operand(bx, dst);
                let src_val = self.codegen_operand(bx, src);
                let count = self.codegen_operand(bx, count).immediate();

                let &ty::RawPtr(pointee, _) = dst_val.layout.ty.kind() else {
                    bug!("expected pointer")
                };
                let pointee_layout = bx
                    .tcx()
                    .layout_of(bx.typing_env().as_query_input(pointee))
                    .expect("expected pointee to have a layout");
                let elem_size = pointee_layout.layout.size().bytes();
                let bytes = bx.unchecked_sumul(count, bx.const_usize(elem_size));

                let align = pointee_layout.layout.align.abi;
                let dst = dst_val.immediate();
                let src = src_val.immediate();

                bx.memcpy(dst, align, src, align, bytes, crate::MemFlags::empty(), None);
            }
            mir::StatementKind::Intrinsic(NonDivergingIntrinsic::CodeviewAnnotation(
                ref operands,
            )) => {
                if operands.is_empty() {
                    bug!("expected at least one operand in codeview annotation");
                }

                let strings = operands
                    .iter()
                    .map(|op| {
                        if let mir::Operand::Constant(c) = op {
                            let val = self.eval_mir_constant(c);
                            let mir::ConstValue::Slice { alloc_id, meta } = val else {
                                bug!("`CodeviewAnnotation` operand is not a `ConstValue::Slice`");
                            };
                            bx.tcx()
                                .global_alloc(alloc_id)
                                .unwrap_memory()
                                .inner()
                                .inspect_with_uninit_and_ptr_outside_interpreter(0..meta as usize)
                        } else {
                            bug!("`CodeviewAnnotation` operand is not a constant");
                        }
                    })
                    .collect::<Vec<_>>();

                bx.codeview_annotation(&strings);
            }
            mir::StatementKind::FakeRead(..)
            | mir::StatementKind::AscribeUserType(..)
            | mir::StatementKind::ConstEvalCounter
            | mir::StatementKind::PlaceMention(..)
            | mir::StatementKind::BackwardIncompatibleDropHint { .. }
            | mir::StatementKind::Nop => {}
        }
    }
}
