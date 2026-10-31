use rustc_middle::infer::canonical::QueryRegionConstraints;
use rustc_middle::ty::TyCtxt;
use rustc_span::{BytePos, DUMMY_SP, Span};
use rustc_type_ir::region_constraint::{And, LeafRegionConstraint, NoSpan, Or};

use super::{RegionConstraintCollector, RegionConstraintStorage, SolverRegionConstraint};
use crate::infer::snapshot::undo_log::InferCtxtUndoLogs;

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
fn trivial_solver_constraints_keep_query_response_empty() {
    let mut constraints = QueryRegionConstraints::<'static>::default();
    assert!(constraints.is_empty());
    constraints.extend(&QueryRegionConstraints::default());
    assert!(constraints.solver_constraints.is_none());

    constraints
        .add_solver_constraints(rustc_type_ir::region_constraint::RegionConstraint::new_true());
    assert!(constraints.is_empty());
    constraints.extend(&constraints.clone());
    assert!(constraints.is_empty());
}

#[test]
fn extending_query_response_preserves_solver_constraints() {
    let mut constraints = QueryRegionConstraints::<'static>::default();
    constraints.add_solver_constraints(
        rustc_type_ir::region_constraint::RegionConstraint::new_ambig(NoSpan),
    );
    assert!(!constraints.is_empty());

    let mut output = QueryRegionConstraints::default();
    output.extend(&constraints);
    assert!(output.solver_constraints.as_ref().unwrap().is_ambig());
    output.extend(&QueryRegionConstraints::default());
    assert!(output.solver_constraints.as_ref().unwrap().is_ambig());

    constraints
        .add_solver_constraints(rustc_type_ir::region_constraint::RegionConstraint::new_false());
    output.extend(&constraints);
    assert!(output.solver_constraints.as_ref().unwrap().is_false());
}

#[test]
fn taking_solver_constraints_clears_storage() {
    let mut storage = RegionConstraintStorage::default();
    let mut undo_log = InferCtxtUndoLogs::default();
    let mut collector =
        RegionConstraintCollector { storage: &mut storage, undo_log: &mut undo_log };
    collector.register_solver_constraint(SolverRegionConstraint::new_ambig(DUMMY_SP));

    assert!(collector.take_and_reset_solver_data().is_ambig());
    assert!(collector.solver_data().get_constraint().is_true());
    assert!(collector.take_and_reset_solver_data().is_true());
}
