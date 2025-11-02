//! This module provides a pass that removes parts of MIR that are no longer relevant after
//! analysis phase and borrowck. In particular, it removes false edges, user type annotations and
//! replaces following statements with [`Nop`]s:
//!
//!   - [`AscribeUserType`]
//!   - [`FakeRead`]
//!   - [`Assign`] statements with a [`Fake`] borrow
//!   - [`Coverage`] statements of kind [`BlockMarker`] or [`SpanMarker`]
//!
//! [`AscribeUserType`]: rustc_middle::mir::StatementKind::AscribeUserType
//! [`Assign`]: rustc_middle::mir::StatementKind::Assign
//! [`FakeRead`]: rustc_middle::mir::StatementKind::FakeRead
//! [`Nop`]: rustc_middle::mir::StatementKind::Nop
//! [`Fake`]: rustc_middle::mir::BorrowKind::Fake
//! [`Coverage`]: rustc_middle::mir::StatementKind::Coverage
//! [`BlockMarker`]: rustc_middle::mir::coverage::CoverageKind::BlockMarker
//! [`SpanMarker`]: rustc_middle::mir::coverage::CoverageKind::SpanMarker

use rustc_middle::mir::coverage::CoverageKind;
use rustc_middle::mir::*;
use rustc_middle::ty::TyCtxt;
use rustc_middle::ty::adjustment::PointerCoercion;

pub(super) struct CleanupPostBorrowck;

impl<'tcx> crate::MirPass<'tcx> for CleanupPostBorrowck {
    fn run_pass(&self, _tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        // Manually invalidate CFG caches if we actually change a terminator's edges.
        let mut invalidate_cfg = false;
        for basic_block in body.basic_blocks.as_mut_preserves_cfg().iter_mut() {
            for statement in basic_block.statements.iter_mut() {
                match statement.kind {
                    StatementKind::AscribeUserType(..)
                    | StatementKind::Assign(box (_, Rvalue::Ref(_, BorrowKind::Fake(_), _)))
                    | StatementKind::Coverage(
                        // These kinds of coverage statements are markers inserted during
                        // MIR building, and are not needed after InstrumentCoverage.
                        CoverageKind::BlockMarker { .. } | CoverageKind::SpanMarker { .. },
                    )
                    | StatementKind::FakeRead(..)
                    | StatementKind::BackwardIncompatibleDropHint { .. } => {
                        statement.make_nop(true)
                    }
                    StatementKind::Assign(box (
                        _,
                        Rvalue::Cast(
                            ref mut cast_kind @ CastKind::PointerCoercion(
                                PointerCoercion::ArrayToPointer
                                | PointerCoercion::MutToConstPointer,
                                _,
                            ),
                            ..,
                        ),
                    )) => {
                        // BorrowCk needed to track whether these cases were coercions or casts,
                        // to know whether to check lifetimes in their pointees,
                        // but from now on that distinction doesn't matter,
                        // so just make them ordinary pointer casts instead.
                        *cast_kind = CastKind::PtrToPtr;
                    }
                    _ => (),
                }
            }

            // If we change any terminator, we need to ensure that we invalidated the CFG cache.
            let terminator = basic_block.terminator_mut();
            match terminator.kind {
                TerminatorKind::FalseEdge { real_target, .. }
                | TerminatorKind::FalseUnwind { real_target, .. } => {
                    invalidate_cfg = true;
                    terminator.kind = TerminatorKind::Goto { target: real_target };
                }
                _ => {}
            }
        }

        if invalidate_cfg {
            body.basic_blocks.invalidate_cfg_cache();
        }

        body.user_type_annotations.raw.clear();

        for decl in &mut body.local_decls {
            decl.user_ty = None;
        }
    }

    fn is_required(&self) -> bool {
        true
    }
}
