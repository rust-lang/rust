use rustc_middle::traits::solve::inspect::GoalEvaluation;

pub fn print_tree(tree: &GoalEvaluation<'_>) {
    debug!(?tree);
}
