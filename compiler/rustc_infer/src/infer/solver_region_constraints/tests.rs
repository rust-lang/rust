use rustc_data_structures::undo_log::UndoLogs;
use rustc_middle::ty::TyCtxt;
use rustc_span::{BytePos, Span};
use rustc_type_ir::region_constraint::{And, LeafRegionConstraint, Or};

use super::SolverRegionConstraint;
use crate::infer::snapshot::undo_log::UndoLog;

#[test]
fn canonicalization_preserves_only_one_ambiguity() {
    let first = Span::with_root_ctxt(BytePos(1), BytePos(2));
    let second = Span::with_root_ctxt(BytePos(3), BytePos(4));

    let first = LeafRegionConstraint::Ambiguity::<TyCtxt<'_>, _>(first);
    let second = LeafRegionConstraint::Ambiguity::<TyCtxt<'_>, _>(second);

    let c = And::new([first.clone(), second.clone()]);
    assert_eq!(c.0.len(), 1);

    let c = Or::new([And::new([first]), And::new([second])]);
    assert_eq!(c.0.len(), 1);
}

#[test]
fn solver_constraint_batches_rollback() {
    let constraint = |start| {
        SolverRegionConstraint::new_leaf(LeafRegionConstraint::Ambiguity(Span::with_root_ctxt(
            BytePos(start),
            BytePos(start + 1),
        )))
    };
    let first = constraint(1);
    let second = constraint(3);
    let mut inner = crate::infer::InferCtxtInner::new();
    inner.solver_region_constraint_storage.push(first.clone());

    let outer = inner.undo_log.start_snapshot();
    inner.undo_log.push(UndoLog::PushSolverRegionConstraint);
    inner.solver_region_constraint_storage.push(second.clone());

    let nested = inner.undo_log.start_snapshot();
    let old_constraints = inner.solver_region_constraint_storage.take();
    inner.undo_log.push(UndoLog::OverwriteSolverRegionConstraints { old_constraints });
    inner.solver_region_constraint_storage.push(constraint(5));
    inner.undo_log.push(UndoLog::PushSolverRegionConstraint);
    inner.solver_region_constraint_storage.push(constraint(7));

    inner.rollback_to(nested);
    assert_eq!(inner.solver_region_constraint_storage.constraints(), &[first.clone(), second]);
    inner.rollback_to(outer);
    assert_eq!(inner.solver_region_constraint_storage.constraints(), &[first]);
}
