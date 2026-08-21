use rustc_span::{BytePos, DUMMY_SP, Span};
use rustc_type_ir::region_constraint::evaluate_solver_constraint;

use super::SolverRegionConstraint;

fn and(constraints: Vec<SolverRegionConstraint<'static>>) -> SolverRegionConstraint<'static> {
    SolverRegionConstraint::And(constraints.into_boxed_slice())
}

fn or(constraints: Vec<SolverRegionConstraint<'static>>) -> SolverRegionConstraint<'static> {
    SolverRegionConstraint::Or(constraints.into_boxed_slice())
}

fn ambiguity() -> SolverRegionConstraint<'static> {
    SolverRegionConstraint::Ambiguity(DUMMY_SP)
}

#[test]
fn evaluation_is_span_agnostic() {
    let constraints = [
        ambiguity(),
        and(vec![]),
        or(vec![]),
        and(vec![and(vec![]), ambiguity()]),
        and(vec![ambiguity(), or(vec![])]),
        or(vec![or(vec![]), ambiguity()]),
        or(vec![ambiguity(), and(vec![])]),
        and(vec![or(vec![or(vec![]), ambiguity()]), or(vec![ambiguity(), and(vec![])])]),
    ];

    for constraint in constraints {
        let expected = evaluate_solver_constraint(&constraint.clone().without_spans());
        let actual = evaluate_solver_constraint(&constraint).without_spans();
        assert_eq!(actual, expected);
    }
}

#[test]
fn evaluation_preserves_first_ambiguity_span() {
    let first = Span::with_root_ctxt(BytePos(1), BytePos(2));
    let second = Span::with_root_ctxt(BytePos(3), BytePos(4));

    for constraint in [
        and(vec![
            SolverRegionConstraint::Ambiguity(first),
            SolverRegionConstraint::Ambiguity(second),
        ]),
        or(vec![
            SolverRegionConstraint::Ambiguity(first),
            SolverRegionConstraint::Ambiguity(second),
        ]),
    ] {
        assert!(matches!(
            evaluate_solver_constraint(&constraint),
            SolverRegionConstraint::Ambiguity(span) if span == first
        ));
    }
}
