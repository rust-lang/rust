use super::*;

fn constraint_vars(input: Vec<Vec<usize>>) -> IndexVec<ConstraintIndex, Vec<VarIndex>> {
    input
        .into_iter()
        .map(|constraint| constraint.into_iter().map(|x| VarIndex::new(x)).collect())
        .collect()
}
fn constraint_cliques(input: Vec<Vec<usize>>) -> IndexVec<CliqueIndex, Vec<ConstraintIndex>> {
    input
        .into_iter()
        .map(|clique| clique.into_iter().map(|x| ConstraintIndex::new(x)).collect())
        .collect()
}
fn constraint_set<const N: usize>(input: [usize; N]) -> FxIndexSet<ConstraintIndex> {
    FxIndexSet::from_iter(input.into_iter().map(|x| ConstraintIndex::new(x)))
}

#[test]
fn test_noop() {
    let deduped = DedupSolver::dedup(
        constraint_vars(vec![vec![1], vec![1, 1], vec![2], vec![3]]),
        constraint_cliques(vec![vec![0, 1], vec![2], vec![3]]),
        FxIndexSet::default(),
    );
    assert!(deduped.removed_constraints.is_empty());
}
#[test]
fn test_simple() {
    let deduped = DedupSolver::dedup(
        constraint_vars(vec![vec![1], vec![2], vec![3]]),
        constraint_cliques(vec![vec![0, 1], vec![2]]),
        FxIndexSet::default(),
    );
    assert!([constraint_set([0]), constraint_set([1])].contains(&deduped.removed_constraints));
}
#[test]
fn test_dependencies() {
    let deduped = DedupSolver::dedup(
        constraint_vars(vec![
            vec![1, 2, 13],
            vec![4, 5, 16],
            vec![1, 2, 23],
            vec![4, 5, 26],
            vec![1, 2],
            vec![4, 5],
        ]),
        constraint_cliques(vec![vec![0, 1], vec![2, 3], vec![4, 5]]),
        FxIndexSet::default(),
    );
    assert!(
        [constraint_set([0, 2, 4]), constraint_set([1, 3, 5])]
            .contains(&deduped.removed_constraints)
    );
}
#[test]
fn test_unremovable_var() {
    fn try_dedup(unremovable_vars: FxIndexSet<VarIndex>) -> FxIndexSet<ConstraintIndex> {
        DedupSolver::dedup(
            constraint_vars(vec![
                vec![1, 2, 13],
                vec![4, 5, 16],
                vec![1, 2, 23],
                vec![4, 5, 26],
                vec![1, 2],
                vec![4, 5],
            ]),
            constraint_cliques(vec![vec![0, 1], vec![2, 3], vec![4, 5]]),
            unremovable_vars,
        )
        .removed_constraints
    }
    assert_eq!(try_dedup(FxIndexSet::from_iter([VarIndex::new(13)])), constraint_set([1, 3, 5]));
    assert_eq!(try_dedup(FxIndexSet::from_iter([VarIndex::new(16)])), constraint_set([0, 2, 4]));
}
#[test]
fn test_dependencies_unresolvable() {
    let deduped = DedupSolver::dedup(
        constraint_vars(vec![
            vec![1, 2, 13],
            vec![4, 5, 16],
            vec![1, 2, 23],
            vec![4, 6, 26],
            vec![1, 2],
            vec![4, 5],
        ]),
        constraint_cliques(vec![vec![0, 1], vec![2, 3], vec![4, 5]]),
        FxIndexSet::default(),
    );
    assert!(deduped.removed_constraints.is_empty());
}
#[test]
fn test_gh_issues_example() {
    let deduped = DedupSolver::dedup(
        constraint_vars(vec![vec![1], vec![2], vec![3]]),
        constraint_cliques(vec![vec![0, 1, 2]]),
        FxIndexSet::default(),
    );
    assert!(
        [constraint_set([0, 1]), constraint_set([0, 2]), constraint_set([1, 2])]
            .contains(&deduped.removed_constraints)
    );
}
