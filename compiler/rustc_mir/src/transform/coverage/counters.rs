use super::Error;

use super::debug;
use super::graph;
use super::spans;

use debug::{DebugCounters, NESTED_INDENT};
use graph::{BasicCoverageBlock, BcbBranch, CoverageGraph, TraverseCoverageGraphWithLoops};
use spans::CoverageSpan;

use rustc_data_structures::graph::WithNumNodes;
use rustc_index::bit_set::BitSet;
use rustc_middle::mir::coverage::*;

/// Manages the counter and expression indexes/IDs to generate `CoverageKind` components for MIR
/// `Coverage` statements.
pub(super) struct CoverageCounters {
    function_source_hash: u64,
    next_counter_id: u32,
    num_expressions: u32,
    pub debug_counters: DebugCounters,
}

impl CoverageCounters {
    pub fn new(function_source_hash: u64) -> Self {
        Self {
            function_source_hash,
            next_counter_id: CounterValueReference::START.as_u32(),
            num_expressions: 0,
            debug_counters: DebugCounters::new(),
        }
    }

    /// Activate the `DebugCounters` data structures, to provide additional debug formatting
    /// features when formating `CoverageKind` (counter) values.
    pub fn enable_debug(&mut self) {
        self.debug_counters.enable();
    }

    /// Makes `CoverageKind` `Counter`s and `Expressions` for the `BasicCoverageBlock`s directly or
    /// indirectly associated with `CoverageSpans`, and returns additional `Expression`s
    /// representing intermediate values.
    pub fn make_bcb_counters(
        &mut self,
        basic_coverage_blocks: &mut CoverageGraph,
        coverage_spans: &Vec<CoverageSpan>,
    ) -> Result<Vec<CoverageKind>, Error> {
        let mut bcb_counters = BcbCounters::new(self, basic_coverage_blocks);
        bcb_counters.make_bcb_counters(coverage_spans)
    }

    fn make_counter<F>(&mut self, debug_block_label_fn: F) -> CoverageKind
    where
        F: Fn() -> Option<String>,
    {
        let counter = CoverageKind::Counter {
            function_source_hash: self.function_source_hash,
            id: self.next_counter(),
        };
        if self.debug_counters.is_enabled() {
            self.debug_counters.add_counter(&counter, (debug_block_label_fn)());
        }
        counter
    }

    fn make_expression<F>(
        &mut self,
        lhs: ExpressionOperandId,
        op: Op,
        rhs: ExpressionOperandId,
        debug_block_label_fn: F,
    ) -> CoverageKind
    where
        F: Fn() -> Option<String>,
    {
        let id = self.next_expression();
        let expression = CoverageKind::Expression { id, lhs, op, rhs };
        if self.debug_counters.is_enabled() {
            self.debug_counters.add_counter(&expression, (debug_block_label_fn)());
        }
        expression
    }

    pub fn make_identity_counter(&mut self, counter_operand: ExpressionOperandId) -> CoverageKind {
        let some_debug_block_label = if self.debug_counters.is_enabled() {
            self.debug_counters.some_block_label(counter_operand).cloned()
        } else {
            None
        };
        self.make_expression(counter_operand, Op::Add, ExpressionOperandId::ZERO, || {
            some_debug_block_label.clone()
        })
    }

    /// Counter IDs start from one and go up.
    fn next_counter(&mut self) -> CounterValueReference {
        assert!(self.next_counter_id < u32::MAX - self.num_expressions);
        let next = self.next_counter_id;
        self.next_counter_id += 1;
        CounterValueReference::from(next)
    }

    /// Expression IDs start from u32::MAX and go down because a Expression can reference
    /// (add or subtract counts) of both Counter regions and Expression regions. The counter
    /// expression operand IDs must be unique across both types.
    fn next_expression(&mut self) -> InjectedExpressionId {
        assert!(self.next_counter_id < u32::MAX - self.num_expressions);
        let next = u32::MAX - self.num_expressions;
        self.num_expressions += 1;
        InjectedExpressionId::from(next)
    }
}

/// Traverse the `CoverageGraph` and add either a `Counter` or `Expression` to every BCB, to be
/// injected with `CoverageSpan`s. `Expressions` have no runtime overhead, so if a viable expression
/// (adding or subtracting two other counters or expressions) can compute the same result as an
/// embedded counter, an `Expression` should be used.
struct BcbCounters<'a> {
    coverage_counters: &'a mut CoverageCounters,
    basic_coverage_blocks: &'a mut CoverageGraph,
}

impl<'a> BcbCounters<'a> {
    fn new(
        coverage_counters: &'a mut CoverageCounters,
        basic_coverage_blocks: &'a mut CoverageGraph,
    ) -> Self {
        Self { coverage_counters, basic_coverage_blocks }
    }

    /// If two `BasicCoverageBlock`s branch from another `BasicCoverageBlock`, one of the branches
    /// can be counted by `Expression` by subtracting the other branch from the branching
    /// block. Otherwise, the `BasicCoverageBlock` executed the least should have the `Counter`.
    /// One way to predict which branch executes the least is by considering loops. A loop is exited
    /// at a branch, so the branch that jumps to a `BasicCoverageBlock` outside the loop is almost
    /// always executed less than the branch that does not exit the loop.
    ///
    /// Returns any non-code-span expressions created to represent intermediate values (such as to
    /// add two counters so the result can be subtracted from another counter), or an Error with
    /// message for subsequent debugging.
    fn make_bcb_counters(
        &mut self,
        coverage_spans: &[CoverageSpan],
    ) -> Result<Vec<CoverageKind>, Error> {
        debug!("make_bcb_counters(): adding a counter or expression to each BasicCoverageBlock");
        let num_bcbs = self.basic_coverage_blocks.num_nodes();
        let mut collect_intermediate_expressions = Vec::with_capacity(num_bcbs);

        let mut bcbs_with_coverage = BitSet::new_empty(num_bcbs);
        for covspan in coverage_spans {
            bcbs_with_coverage.insert(covspan.bcb);
        }

        // Walk the `CoverageGraph`. For each `BasicCoverageBlock` node with an associated
        // `CoverageSpan`, add a counter. If the `BasicCoverageBlock` branches, add a counter or
        // expression to each branch `BasicCoverageBlock` (if the branch BCB has only one incoming
        // edge) or edge from the branching BCB to the branch BCB (if the branch BCB has multiple
        // incoming edges).
        //
        // The `TraverseCoverageGraphWithLoops` traversal ensures that, when a loop is encountered,
        // all `BasicCoverageBlock` nodes in the loop are visited before visiting any node outside
        // the loop. The `traversal` state includes a `context_stack`, providing a way to know if
        // the current BCB is in one or more nested loops or not.
        let mut traversal = TraverseCoverageGraphWithLoops::new(&self.basic_coverage_blocks);
        while let Some(bcb) = traversal.next(self.basic_coverage_blocks) {
            if bcbs_with_coverage.contains(bcb) {
                debug!("{:?} has at least one `CoverageSpan`. Get or make its counter", bcb);
                let branching_counter_operand =
                    self.get_or_make_counter_operand(bcb, &mut collect_intermediate_expressions)?;

                if self.bcb_needs_branch_counters(bcb) {
                    self.make_branch_counters(
                        &mut traversal,
                        bcb,
                        branching_counter_operand,
                        &mut collect_intermediate_expressions,
                    )?;
                }
            } else {
                debug!(
                    "{:?} does not have any `CoverageSpan`s. A counter will only be added if \
                    and when a covered BCB has an expression dependency.",
                    bcb,
                );
            }
        }

        if traversal.is_complete() {
            Ok(collect_intermediate_expressions)
        } else {
            Error::from_string(format!(
                "`TraverseCoverageGraphWithLoops` missed some `BasicCoverageBlock`s: {:?}",
                traversal.unvisited(),
            ))
        }
    }

    fn make_branch_counters(
        &mut self,
        traversal: &mut TraverseCoverageGraphWithLoops,
        branching_bcb: BasicCoverageBlock,
        branching_counter_operand: ExpressionOperandId,
        collect_intermediate_expressions: &mut Vec<CoverageKind>,
    ) -> Result<(), Error> {
        let branches = self.bcb_branches(branching_bcb);
        debug!(
            "{:?} has some branch(es) without counters:\n  {}",
            branching_bcb,
            branches
                .iter()
                .map(|branch| {
                    format!("{:?}: {:?}", branch, branch.counter(&self.basic_coverage_blocks))
                })
                .collect::<Vec<_>>()
                .join("\n  "),
        );

        // Use the `traversal` state to decide if a subset of the branches exit a loop, making it
        // likely that branch is executed less than branches that do not exit the same loop. In this
        // case, any branch that does not exit the loop (and has not already been assigned a
        // counter) should be counted by expression, if possible. (If a preferred expression branch
        // is not selected based on the loop context, select any branch without an existing
        // counter.)
        let expression_branch = self.choose_preferred_expression_branch(traversal, &branches);

        // Assign a Counter or Expression to each branch, plus additional `Expression`s, as needed,
        // to sum up intermediate results.
        let mut some_sumup_counter_operand = None;
        for branch in branches {
            // Skip the selected `expression_branch`, if any. It's expression will be assigned after
            // all others.
            if branch != expression_branch {
                let branch_counter_operand = if branch.is_only_path_to_target() {
                    debug!(
                        "  {:?} has only one incoming edge (from {:?}), so adding a \
                        counter",
                        branch, branching_bcb
                    );
                    self.get_or_make_counter_operand(
                        branch.target_bcb,
                        collect_intermediate_expressions,
                    )?
                } else {
                    debug!("  {:?} has multiple incoming edges, so adding an edge counter", branch);
                    self.get_or_make_edge_counter_operand(
                        branching_bcb,
                        branch.target_bcb,
                        collect_intermediate_expressions,
                    )?
                };
                if let Some(sumup_counter_operand) =
                    some_sumup_counter_operand.replace(branch_counter_operand)
                {
                    let intermediate_expression = self.coverage_counters.make_expression(
                        branch_counter_operand,
                        Op::Add,
                        sumup_counter_operand,
                        || None,
                    );
                    debug!(
                        "  [new intermediate expression: {}]",
                        self.format_counter(&intermediate_expression)
                    );
                    let intermediate_expression_operand = intermediate_expression.as_operand_id();
                    collect_intermediate_expressions.push(intermediate_expression);
                    some_sumup_counter_operand.replace(intermediate_expression_operand);
                }
            }
        }

        // Assign the final expression to the `expression_branch` by subtracting the total of all
        // other branches from the counter of the branching BCB.
        let sumup_counter_operand =
            some_sumup_counter_operand.expect("sumup_counter_operand should have a value");
        debug!(
            "Making an expression for the selected expression_branch: {:?} \
            (expression_branch predecessors: {:?})",
            expression_branch,
            self.bcb_predecessors(expression_branch.target_bcb),
        );
        let expression = self.coverage_counters.make_expression(
            branching_counter_operand,
            Op::Subtract,
            sumup_counter_operand,
            || Some(format!("{:?}", expression_branch)),
        );
        debug!("{:?} gets an expression: {}", expression_branch, self.format_counter(&expression));
        let bcb = expression_branch.target_bcb;
        if expression_branch.is_only_path_to_target() {
            self.basic_coverage_blocks[bcb].set_counter(expression)?;
        } else {
            self.basic_coverage_blocks[bcb].set_edge_counter_from(branching_bcb, expression)?;
        }
        Ok(())
    }

    fn get_or_make_counter_operand(
        &mut self,
        bcb: BasicCoverageBlock,
        collect_intermediate_expressions: &mut Vec<CoverageKind>,
    ) -> Result<ExpressionOperandId, Error> {
        self.recursive_get_or_make_counter_operand(bcb, collect_intermediate_expressions, 1)
    }

    fn recursive_get_or_make_counter_operand(
        &mut self,
        bcb: BasicCoverageBlock,
        collect_intermediate_expressions: &mut Vec<CoverageKind>,
        debug_indent_level: usize,
    ) -> Result<ExpressionOperandId, Error> {
        // If the BCB already has a counter, return it.
        if let Some(counter_kind) = self.basic_coverage_blocks[bcb].counter() {
            debug!(
                "{}{:?} already has a counter: {}",
                NESTED_INDENT.repeat(debug_indent_level),
                bcb,
                self.format_counter(counter_kind),
            );
            return Ok(counter_kind.as_operand_id());
        }

        // A BCB with only one incoming edge gets a simple `Counter` (via `make_counter()`).
        // Also, a BCB that loops back to itself gets a simple `Counter`. This may indicate the
        // program results in a tight infinite loop, but it should still compile.
        let one_path_to_target = self.bcb_has_one_path_to_target(bcb);
        if one_path_to_target || self.bcb_predecessors(bcb).contains(&bcb) {
            let counter_kind = self.coverage_counters.make_counter(|| Some(format!("{:?}", bcb)));
            if one_path_to_target {
                debug!(
                    "{}{:?} gets a new counter: {}",
                    NESTED_INDENT.repeat(debug_indent_level),
                    bcb,
                    self.format_counter(&counter_kind),
                );
            } else {
                debug!(
                    "{}{:?} has itself as its own predecessor. It can't be part of its own \
                    Expression sum, so it will get its own new counter: {}. (Note, the compiled \
                    code will generate an infinite loop.)",
                    NESTED_INDENT.repeat(debug_indent_level),
                    bcb,
                    self.format_counter(&counter_kind),
                );
            }
            return self.basic_coverage_blocks[bcb].set_counter(counter_kind);
        }

        // A BCB with multiple incoming edges can compute its count by `Expression`, summing up the
        // counters and/or expressions of its incoming edges. This will recursively get or create
        // counters for those incoming edges first, then call `make_expression()` to sum them up,
        // with additional intermediate expressions as needed.
        let mut predecessors = self.bcb_predecessors(bcb).clone().into_iter();
        debug!(
            "{}{:?} has multiple incoming edges and will get an expression that sums them up...",
            NESTED_INDENT.repeat(debug_indent_level),
            bcb,
        );
        let first_edge_counter_operand = self.recursive_get_or_make_edge_counter_operand(
            predecessors.next().unwrap(),
            bcb,
            collect_intermediate_expressions,
            debug_indent_level + 1,
        )?;
        let mut some_sumup_edge_counter_operand = None;
        for predecessor in predecessors {
            let edge_counter_operand = self.recursive_get_or_make_edge_counter_operand(
                predecessor,
                bcb,
                collect_intermediate_expressions,
                debug_indent_level + 1,
            )?;
            if let Some(sumup_edge_counter_operand) =
                some_sumup_edge_counter_operand.replace(edge_counter_operand)
            {
                let intermediate_expression = self.coverage_counters.make_expression(
                    sumup_edge_counter_operand,
                    Op::Add,
                    edge_counter_operand,
                    || None,
                );
                debug!(
                    "{}new intermediate expression: {}",
                    NESTED_INDENT.repeat(debug_indent_level),
                    self.format_counter(&intermediate_expression)
                );
                let intermediate_expression_operand = intermediate_expression.as_operand_id();
                collect_intermediate_expressions.push(intermediate_expression);
                some_sumup_edge_counter_operand.replace(intermediate_expression_operand);
            }
        }
        let counter_kind = self.coverage_counters.make_expression(
            first_edge_counter_operand,
            Op::Add,
            some_sumup_edge_counter_operand.unwrap(),
            || Some(format!("{:?}", bcb)),
        );
        debug!(
            "{}{:?} gets a new counter (sum of predecessor counters): {}",
            NESTED_INDENT.repeat(debug_indent_level),
            bcb,
            self.format_counter(&counter_kind)
        );
        self.basic_coverage_blocks[bcb].set_counter(counter_kind)
    }

    fn get_or_make_edge_counter_operand(
        &mut self,
        from_bcb: BasicCoverageBlock,
        to_bcb: BasicCoverageBlock,
        collect_intermediate_expressions: &mut Vec<CoverageKind>,
    ) -> Result<ExpressionOperandId, Error> {
        self.recursive_get_or_make_edge_counter_operand(
            from_bcb,
            to_bcb,
            collect_intermediate_expressions,
            1,
        )
    }

    fn recursive_get_or_make_edge_counter_operand(
        &mut self,
        from_bcb: BasicCoverageBlock,
        to_bcb: BasicCoverageBlock,
        collect_intermediate_expressions: &mut Vec<CoverageKind>,
        debug_indent_level: usize,
    ) -> Result<ExpressionOperandId, Error> {
        // If the source BCB has only one successor (assumed to be the given target), an edge
        // counter is unnecessary. Just get or make a counter for the source BCB.
        let successors = self.bcb_successors(from_bcb).iter();
        if successors.len() == 1 {
            return self.recursive_get_or_make_counter_operand(
                from_bcb,
                collect_intermediate_expressions,
                debug_indent_level + 1,
            );
        }

        // If the edge already has a counter, return it.
        if let Some(counter_kind) = self.basic_coverage_blocks[to_bcb].edge_counter_from(from_bcb) {
            debug!(
                "{}Edge {:?}->{:?} already has a counter: {}",
                NESTED_INDENT.repeat(debug_indent_level),
                from_bcb,
                to_bcb,
                self.format_counter(counter_kind)
            );
            return Ok(counter_kind.as_operand_id());
        }

        // Make a new counter to count this edge.
        let counter_kind =
            self.coverage_counters.make_counter(|| Some(format!("{:?}->{:?}", from_bcb, to_bcb)));
        debug!(
            "{}Edge {:?}->{:?} gets a new counter: {}",
            NESTED_INDENT.repeat(debug_indent_level),
            from_bcb,
            to_bcb,
            self.format_counter(&counter_kind)
        );
        self.basic_coverage_blocks[to_bcb].set_edge_counter_from(from_bcb, counter_kind)
    }

    /// Select a branch for the expression, either the recommended `reloop_branch`, or if none was
    /// found, select any branch.
    fn choose_preferred_expression_branch(
        &self,
        traversal: &TraverseCoverageGraphWithLoops,
        branches: &[BcbBranch],
    ) -> BcbBranch {
        let branch_needs_a_counter =
            |branch: &BcbBranch| branch.counter(&self.basic_coverage_blocks).is_none();

        let some_reloop_branch = self.find_some_reloop_branch(traversal, &branches);
        if let Some(reloop_branch_without_counter) =
            some_reloop_branch.filter(branch_needs_a_counter)
        {
            debug!(
                "Selecting reloop_branch={:?} that still needs a counter, to get the \
                `Expression`",
                reloop_branch_without_counter
            );
            reloop_branch_without_counter
        } else {
            let &branch_without_counter = branches
                .iter()
                .find(|&&branch| branch.counter(&self.basic_coverage_blocks).is_none())
                .expect(
                    "needs_branch_counters was `true` so there should be at least one \
                    branch",
                );
            debug!(
                "Selecting any branch={:?} that still needs a counter, to get the \
                `Expression` because there was no `reloop_branch`, or it already had a \
                counter",
                branch_without_counter
            );
            branch_without_counter
        }
    }

    /// At most, one of the branches (or its edge, from the branching_bcb, if the branch has
    /// multiple incoming edges) can have a counter computed by expression.
    ///
    /// If at least one of the branches leads outside of a loop (`found_loop_exit` is
    /// true), and at least one other branch does not exit the loop (the first of which
    /// is captured in `some_reloop_branch`), it's likely any reloop branch will be
    /// executed far more often than loop exit branch, making the reloop branch a better
    /// candidate for an expression.
    fn find_some_reloop_branch(
        &self,
        traversal: &TraverseCoverageGraphWithLoops,
        branches: &[BcbBranch],
    ) -> Option<BcbBranch> {
        let branch_needs_a_counter =
            |branch: &BcbBranch| branch.counter(&self.basic_coverage_blocks).is_none();

        let mut some_reloop_branch: Option<BcbBranch> = None;
        for context in traversal.context_stack.iter().rev() {
            if let Some((backedge_from_bcbs, _)) = &context.loop_backedges {
                let mut found_loop_exit = false;
                for &branch in branches.iter() {
                    if backedge_from_bcbs.iter().any(|&backedge_from_bcb| {
                        self.bcb_is_dominated_by(backedge_from_bcb, branch.target_bcb)
                    }) {
                        if let Some(reloop_branch) = some_reloop_branch {
                            if reloop_branch.counter(&self.basic_coverage_blocks).is_none() {
                                // we already found a candidate reloop_branch that still
                                // needs a counter
                                continue;
                            }
                        }
                        // The path from branch leads back to the top of the loop. Set this
                        // branch as the `reloop_branch`. If this branch already has a
                        // counter, and we find another reloop branch that doesn't have a
                        // counter yet, that branch will be selected as the `reloop_branch`
                        // instead.
                        some_reloop_branch = Some(branch);
                    } else {
                        // The path from branch leads outside this loop
                        found_loop_exit = true;
                    }
                    if found_loop_exit
                        && some_reloop_branch.filter(branch_needs_a_counter).is_some()
                    {
                        // Found both a branch that exits the loop and a branch that returns
                        // to the top of the loop (`reloop_branch`), and the `reloop_branch`
                        // doesn't already have a counter.
                        break;
                    }
                }
                if !found_loop_exit {
                    debug!(
                        "No branches exit the loop, so any branch without an existing \
                        counter can have the `Expression`."
                    );
                    break;
                }
                if some_reloop_branch.is_some() {
                    debug!(
                        "Found a branch that exits the loop and a branch the loops back to \
                        the top of the loop (`reloop_branch`). The `reloop_branch` will \
                        get the `Expression`, as long as it still needs a counter."
                    );
                    break;
                }
                // else all branches exited this loop context, so run the same checks with
                // the outer loop(s)
            }
        }
        some_reloop_branch
    }

    #[inline]
    fn bcb_predecessors(&self, bcb: BasicCoverageBlock) -> &Vec<BasicCoverageBlock> {
        &self.basic_coverage_blocks.predecessors[bcb]
    }

    #[inline]
    fn bcb_successors(&self, bcb: BasicCoverageBlock) -> &Vec<BasicCoverageBlock> {
        &self.basic_coverage_blocks.successors[bcb]
    }

    #[inline]
    fn bcb_branches(&self, from_bcb: BasicCoverageBlock) -> Vec<BcbBranch> {
        self.bcb_successors(from_bcb)
            .iter()
            .map(|&to_bcb| BcbBranch::from_to(from_bcb, to_bcb, &self.basic_coverage_blocks))
            .collect::<Vec<_>>()
    }

    fn bcb_needs_branch_counters(&self, bcb: BasicCoverageBlock) -> bool {
        let branch_needs_a_counter =
            |branch: &BcbBranch| branch.counter(&self.basic_coverage_blocks).is_none();
        let branches = self.bcb_branches(bcb);
        branches.len() > 1 && branches.iter().any(branch_needs_a_counter)
    }

    /// Returns true if the BasicCoverageBlock has zero or one incoming edge. (If zero, it should be
    /// the entry point for the function.)
    #[inline]
    fn bcb_has_one_path_to_target(&self, bcb: BasicCoverageBlock) -> bool {
        self.bcb_predecessors(bcb).len() <= 1
    }

    #[inline]
    fn bcb_is_dominated_by(&self, node: BasicCoverageBlock, dom: BasicCoverageBlock) -> bool {
        self.basic_coverage_blocks.is_dominated_by(node, dom)
    }

    #[inline]
    fn format_counter(&self, counter_kind: &CoverageKind) -> String {
        self.coverage_counters.debug_counters.format_counter(counter_kind)
    }
}
