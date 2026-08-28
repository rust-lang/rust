use rustc_middle::infer::canonical::QueryRegionConstraints;
use rustc_middle::ty::TyCtxt;
use rustc_span::{BytePos, Span};
use rustc_type_ir::region_constraint::{And, LeafRegionConstraint, Or};

use super::{SolverRegionConstraint, SolverRegionConstraintStorage};

#[test]
fn true_constraint_keeps_query_response_empty() {
    // Mirrors `register_solver_region_constraint`: anding a trivially true
    // constraint into an empty store has to leave the store trivially true,
    // as the resulting query response would otherwise no longer be empty.
    let mut storage = SolverRegionConstraintStorage::<'static>::new();
    storage.overwrite(SolverRegionConstraint::build_and(
        SolverRegionConstraint::new_true(),
        storage.get_constraint(),
    ));

    let constraints = QueryRegionConstraints {
        solver_constraints: storage.get_constraint().without_spans(),
        ..Default::default()
    };
    assert!(constraints.is_empty());
}

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
