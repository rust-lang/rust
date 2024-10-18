use std::collections::VecDeque;

use rustc_data_structures::fx::{FxHashMap, FxHashSet, FxIndexMap, FxIndexSet};
use rustc_middle::bug;
use rustc_middle::mir::coverage::{
    BlockMarkerId, CandidateCovId, ConditionId, ConditionInfo, DecisionId, MCDCBranchSpan,
    MCDCDecisionSpan, MatchCoverageInfo, MatchKey, MatchPairId, SubcandidateId,
};
use rustc_middle::mir::{BasicBlock, SourceInfo, TerminatorKind};
use rustc_middle::ty::TyCtxt;
use rustc_span::Span;

use crate::build::coverageinfo::mcdc::{DecisionCtx, MCDCInfoBuilder, MCDCTargetInfo};
use crate::build::matches::Candidate;
use crate::build::{Builder, CFG};

/// Represent a matched target. This target might contain several
/// patterns from different candidates.
///
/// We represents the process of matching as a graph.
///
/// For instance, pattern decision `(A, B, C | D(E), F | G)` generates graph like:
///            *
///            |
///            A
///            |
///            B
///     ∨------∧------∨
///     C             D
///     ｜             ｜
///     ｜             E
///     ∧------∨------∧
///            *
///     ∨------∧------∨
///     F             G
/// `*` represents a virtual node. Matching graph have following properties:
/// * Every subcandidate is represented as a branch in the graph.
/// * Start with a virtual node, because we always start from the root subcandidate. (Remember that the root subcandidate might have no pattern)
/// * If two or more branches are merged, they are merged into a virtual node.
///
/// With the help of matching graph we can construct decision tree of the pattern:
/// * The true next of a node is its leftmost successor.
/// * The false next of a node is its sibling node at right sharing same predecssor.
/// * If (true/false) next of a node is a virtual node, the actual next is the (true/false) next of the virtual node.
///
/// Specific to this graph,
/// * `A` and `B` are located in root subcandidate.
/// * `C` is located in subcandidate 1.
/// * `D`, `E` are located in subcandidate 2.
/// * `F` is located in subcandidate 3.
/// * `G` is located in subcandidate 4.
///
/// Thus the true next of `A` is `B`, the false next is none (this pattern is not matched if failed to match `A`).
/// The true next of `B` is `C`, the false next is none.
/// The true next of `C` is `F` (the true next of a virtual node), the false next is `D`.
/// The true next of `D` is `E`, the false next is none.
/// The true next of `E` is `F` (through a virtual node), the false next is none.
/// The true next of `F` is none (end matching here), the false next is `G`.
/// The true and false next of `G` both are none.
#[derive(Debug, Default)]
struct MatchNode {
    matched_keys: Vec<MatchKey>,
    // Index in matching graph of the next tested node if this node is evaluated to true.
    true_next: Option<usize>,
    // Index in matching graph of the next tested node if this node is evaluated to false.
    false_next: Option<usize>,
    // A match node is virtual if it is not related to any patterns.
    is_virtual: bool,
}

impl MatchNode {
    fn real_node(matched_keys: Vec<MatchKey>) -> Self {
        Self { matched_keys, true_next: None, false_next: None, is_virtual: false }
    }

    fn virtual_node(subcandidate_id: SubcandidateId) -> Self {
        let key = MatchKey {
            decision_id: DecisionId::ZERO,
            subcandidate_id,
            match_id: MatchPairId::INVALID,
        };
        Self { matched_keys: vec![key], true_next: None, false_next: None, is_virtual: true }
    }

    fn node_id(&self) -> MatchPairId {
        self.matched_keys.first().expect("MatchNode must have at least one key").match_id
    }
}

// Information about patterns generating conditions.
#[derive(Debug)]
struct MatchPairInfo {
    span: Span,
    fully_matched_block_pairs: Vec<(BasicBlock, BasicBlock)>,
}

// Context to process a pattern decision (without guard).
#[derive(Debug)]
struct CandidateCtx {
    span: Span,
    latest_subcandidate_id: SubcandidateId,
    parent_subcandidate_map: FxHashMap<SubcandidateId, SubcandidateId>,
    last_child_node_map: FxHashMap<usize, usize>,
    match_pairs_info: FxHashMap<MatchPairId, MatchPairInfo>,
    matching_graph: Vec<MatchNode>,
}

impl CandidateCtx {
    fn new(span: Span) -> Self {
        Self {
            span,
            latest_subcandidate_id: SubcandidateId::ROOT,
            parent_subcandidate_map: FxHashMap::default(),
            last_child_node_map: FxHashMap::default(),
            match_pairs_info: FxHashMap::default(),
            matching_graph: vec![MatchNode::virtual_node(SubcandidateId::ROOT)],
        }
    }
    fn next_subcandidate_in(&mut self, parent_subcandidate_id: SubcandidateId) -> SubcandidateId {
        let id = self.latest_subcandidate_id.next_subcandidate_id();
        self.latest_subcandidate_id = id;
        self.parent_subcandidate_map.insert(id, parent_subcandidate_id);
        id
    }

    fn on_visiting_patterns(&mut self, matched_info: Vec<MatchCoverageInfo>) {
        let keys = matched_info
            .into_iter()
            .filter_map(|info| info.fully_matched.then_some(info.key))
            .collect::<Vec<_>>();

        if keys.is_empty() {
            return;
        }

        let parent_subcandidate_id =
            self.parent_subcandidate_map.get(&keys[0].subcandidate_id).copied();
        assert!(
            keys.iter().skip(1).all(|key| self
                .parent_subcandidate_map
                .get(&key.subcandidate_id)
                .copied()
                == parent_subcandidate_id),
            "Patterns simultaneously matched should have same parent subcandidate"
        );
        let current_node = MatchNode::real_node(keys);
        let is_predecessor_of_current = |node: &MatchNode| {
            if current_node.matched_keys.iter().all(|this_matched| {
                node.matched_keys
                    .iter()
                    .any(|prev| prev.subcandidate_id == this_matched.subcandidate_id)
            }) {
                return true;
            }
            parent_subcandidate_id.is_some_and(|parent_subcandidate| {
                node.matched_keys.iter().any(|prev| prev.subcandidate_id == parent_subcandidate)
            })
        };
        if let Some(predecessor_idx) = self
            .matching_graph
            .iter()
            .rev()
            .position(is_predecessor_of_current)
            .map(|rev_pos| self.matching_graph.len() - (1 + rev_pos))
        {
            let current_idx = self.matching_graph.len();
            if self.matching_graph[predecessor_idx].true_next.is_none() {
                self.matching_graph[predecessor_idx].true_next = Some(current_idx);
            }
            if let Some(elder_sibling_idx) =
                self.last_child_node_map.insert(predecessor_idx, current_idx)
            {
                self.matching_graph[elder_sibling_idx].false_next = Some(current_idx);
            }
        }
        self.matching_graph.push(current_node);
    }

    fn on_matching_patterns(
        &mut self,
        test_block: BasicBlock,
        matched_block: BasicBlock,
        matched_infos: Vec<MatchCoverageInfo>,
    ) {
        for matched in matched_infos {
            let info = self.match_pairs_info.entry(matched.key.match_id).or_insert_with(|| {
                MatchPairInfo { span: matched.span, fully_matched_block_pairs: vec![] }
            });
            if matched.fully_matched {
                info.fully_matched_block_pairs.push((test_block, matched_block));
            }
        }
    }

    fn on_merging_subcandidates(
        &mut self,
        next_subcandidate_id: SubcandidateId,
        merging_subcandidate_ids: impl Iterator<Item = SubcandidateId>,
    ) {
        let merged_node = MatchNode::virtual_node(next_subcandidate_id);
        let current_idx = self.matching_graph.len();
        let mut merging_subcandidates = merging_subcandidate_ids.collect::<FxHashSet<_>>();
        'r: for prev_node in self.matching_graph.iter_mut().rev() {
            for key in &prev_node.matched_keys {
                if merging_subcandidates.remove(&key.subcandidate_id) {
                    assert!(
                        prev_node.true_next.is_none(),
                        "merged node must be the tail of its branch"
                    );
                    prev_node.true_next = Some(current_idx);
                    if merging_subcandidates.is_empty() {
                        break 'r;
                    }
                }
            }
        }
        self.matching_graph.push(merged_node);
    }

    fn generate_condition_info(&self) -> FxIndexMap<MatchPairId, ConditionInfo> {
        let mut condition_infos_map = FxIndexMap::<MatchPairId, ConditionInfo>::default();
        let mut condition_counter = 0;
        let mut new_condition_info = || {
            let condition_id = ConditionId::from_usize(condition_counter);
            condition_counter += 1;
            ConditionInfo { condition_id, true_next_id: None, false_next_id: None }
        };
        let find_next_node = |next_idx: Option<usize>, branch: bool| -> Option<&MatchNode> {
            let mut next_node = &self.matching_graph[next_idx?];
            while next_node.is_virtual {
                let next_idx = if branch { next_node.true_next } else { next_node.false_next };
                next_node = &self.matching_graph[next_idx?];
            }
            Some(next_node)
        };

        for node in &self.matching_graph {
            if node.is_virtual {
                continue;
            }
            condition_infos_map.entry(node.node_id()).or_insert_with(&mut new_condition_info);

            let [true_next_id, false_next_id] = [(true, node.true_next), (false, node.false_next)]
                .map(|(branch, next_idx)| {
                    find_next_node(next_idx, branch).map(|next_node| {
                        condition_infos_map
                            .entry(next_node.node_id())
                            .or_insert_with(&mut new_condition_info)
                            .condition_id
                    })
                });

            let condition_info =
                condition_infos_map.get_mut(&node.node_id()).expect("ensured to be inserted above");

            assert!(
                condition_info.true_next_id.is_none()
                    || condition_info.true_next_id == true_next_id,
                "only has one true next"
            );
            assert!(
                condition_info.false_next_id.is_none()
                    || condition_info.false_next_id == false_next_id,
                "only has one false next"
            );

            condition_info.true_next_id = true_next_id;
            condition_info.false_next_id = false_next_id;
        }
        condition_infos_map
    }

    fn into_matching_decision(
        self,
        mut into_branch_blocks: impl FnMut(
            Span,
            &[(BasicBlock, BasicBlock)],
            &FxIndexSet<BasicBlock>,
        ) -> (Vec<BlockMarkerId>, Vec<BlockMarkerId>),
    ) -> MCDCTargetInfo {
        // Note. Conditions tested in same blocks are from different subcandidates.
        // Since one condition might be the `false next` of another, it can not update
        // condbitmap in another's matched block as if it is evaluated to `false` (though we know it is in fact),
        // otherwise we may not update test vector index right. `matched_blocks_in_decision` here is used to record
        // such blocks.
        // In future we could differentiate blocks updating condbitmap from blocks increment counts to get
        // better results.
        let mut matched_blocks_in_decision = FxIndexSet::default();
        let conditions: Vec<_> = self
            .generate_condition_info()
            .into_iter()
            .map(|(match_id, condition_info)| {
                let &MatchPairInfo { span, ref fully_matched_block_pairs } =
                    self.match_pairs_info.get(&match_id).expect("all match pairs are recorded");

                matched_blocks_in_decision
                    .extend(fully_matched_block_pairs.iter().map(|pair| pair.1));
                let (true_markers, false_markers) = into_branch_blocks(
                    span,
                    fully_matched_block_pairs,
                    &matched_blocks_in_decision,
                );

                MCDCBranchSpan { span, condition_info, true_markers, false_markers }
            })
            .collect();
        let decision = MCDCDecisionSpan::new(self.span);
        MCDCTargetInfo { decision, conditions, nested_decisions_id: vec![] }
    }
}

#[derive(Debug)]
pub(super) struct MatchingDecisionCtx {
    candidates: FxIndexMap<DecisionId, CandidateCtx>,
    test_blocks: FxIndexMap<BasicBlock, FxIndexSet<BasicBlock>>,
}

impl MatchingDecisionCtx {
    pub(super) fn new(candidates: &[(Span, DecisionId)]) -> Self {
        Self {
            candidates: candidates
                .into_iter()
                .map(|&(span, id)| (id, CandidateCtx::new(span)))
                .collect(),
            test_blocks: FxIndexMap::default(),
        }
    }

    fn visit_conditions(&mut self, patterns: &Vec<MatchCoverageInfo>) {
        for (decision_id, infos) in group_match_info_by_decision(patterns) {
            let candidate_ctx = self.candidates.get_mut(&decision_id).expect("unknown candidate");
            candidate_ctx.on_visiting_patterns(infos);
        }
    }

    fn match_conditions(
        &mut self,
        cfg: &mut CFG<'_>,
        test_block: BasicBlock,
        target_patterns: impl Iterator<Item = (BasicBlock, Vec<MatchCoverageInfo>)>,
    ) {
        let mut otherwise_blocks = FxIndexSet::default();
        let mut matched_blocks = Vec::with_capacity(target_patterns.size_hint().0);
        for (block, patterns) in target_patterns {
            if patterns.is_empty() {
                otherwise_blocks.insert(block);
                continue;
            }
            for (decision_id, infos) in group_match_info_by_decision(&patterns) {
                let candidate_ctx =
                    self.candidates.get_mut(&decision_id).expect("unknown candidate");
                candidate_ctx.on_matching_patterns(test_block, block, infos);
            }
            matched_blocks.push(block);
        }
        let fail_block =
            find_fail_block(cfg, test_block, matched_blocks.iter().copied(), otherwise_blocks);
        self.test_blocks
            .insert(test_block, matched_blocks.into_iter().chain(fail_block.into_iter()).collect());
    }

    fn finish_matching_tree(
        self,
        mut inject_block_marker: impl FnMut(Span, BasicBlock) -> BlockMarkerId,
    ) -> FxIndexMap<DecisionId, MCDCTargetInfo> {
        let mut block_markers_map = FxHashMap::<BasicBlock, BlockMarkerId>::default();

        let mut into_branch_blocks =
            |span: Span,
             test_match_pairs: &[(BasicBlock, BasicBlock)],
             excluded_unmatched_blocks: &FxIndexSet<BasicBlock>| {
                let mut true_markers = Vec::with_capacity(test_match_pairs.len());
                let mut false_markers = Vec::with_capacity(test_match_pairs.len());
                let mut into_marker = |block: BasicBlock| {
                    *block_markers_map
                        .entry(block)
                        .or_insert_with(|| inject_block_marker(span, block))
                };
                for &(test_block, matched_block) in test_match_pairs {
                    true_markers.push(into_marker(matched_block));
                    false_markers.extend(
                        self.test_blocks
                            .get(&test_block)
                            .expect("all test blocks must be recorded")
                            .into_iter()
                            .copied()
                            .filter_map(|block| {
                                (block != matched_block
                                    && !excluded_unmatched_blocks.contains(&block))
                                .then(|| into_marker(block))
                            }),
                    );
                }
                (true_markers, false_markers)
            };
        self.candidates
            .into_iter()
            .map(|(decision_id, candidate_ctx)| {
                (decision_id, candidate_ctx.into_matching_decision(&mut into_branch_blocks))
            })
            .collect()
    }
}

fn group_match_info_by_decision(
    matched_infos: &[MatchCoverageInfo],
) -> impl IntoIterator<Item = (DecisionId, Vec<MatchCoverageInfo>)> {
    let mut keys_by_decision = FxIndexMap::<DecisionId, Vec<MatchCoverageInfo>>::default();
    for pattern_info in matched_infos {
        keys_by_decision
            .entry(pattern_info.key.decision_id)
            .or_default()
            .push(pattern_info.clone());
    }
    keys_by_decision
}

/// Upon testing there probably is a block to go if all patterns failed to match.
/// We should increment false count and update condbitmap index of all tested conditions in this block.
/// Considering the `otherwise_block` might be reused by several tests, we inject a `fail_block`
/// here to do such stuff.
fn find_fail_block(
    cfg: &mut CFG<'_>,
    test_block: BasicBlock,
    matched_blocks: impl Iterator<Item = BasicBlock>,
    otherwise_blocks: FxIndexSet<BasicBlock>,
) -> Vec<BasicBlock> {
    let matched_blocks = FxIndexSet::from_iter(matched_blocks);
    let mut prev_fail_blocks = vec![];
    let mut blocks = VecDeque::from([test_block]);
    // Some tests might contain multiple sub tests. For example, to test range pattern `0..5`, first
    // test if the value >= 0, then test if the value < 5. So `matched_block` might not be a successor of
    // the `test_block` and there are two blocks which both are predecessors of `fail_block`.
    while let Some(block) = blocks.pop_front() {
        for successor in cfg.block_data(block).terminator().successors() {
            if matched_blocks.contains(&successor) {
                continue;
            } else if otherwise_blocks.contains(&successor) {
                prev_fail_blocks.push(block);
            } else {
                blocks.push_back(successor);
            }
        }
    }
    otherwise_blocks
        .into_iter()
        .map(|otherwise_block| {
            let fail_block = cfg.start_new_block();
            cfg.terminate(
                fail_block,
                cfg.block_data(test_block).terminator().source_info,
                TerminatorKind::Goto { target: otherwise_block },
            );

            for &prev_block in &prev_fail_blocks {
                let otherwise_ref = cfg
                    .block_data_mut(prev_block)
                    .terminator_mut()
                    .successors_mut()
                    .find(|block| **block == otherwise_block)
                    .expect("otherwise_block is ensured to be one of successors above");
                *otherwise_ref = fail_block;
            }
            fail_block
        })
        .collect()
}

/// Context handling matching decisions with if guards.
/// After lowering matching tree, rustc build mir for if guards and code in arms candidate by candidate.
/// In case decisions in arm blocks are taken as nested decisions, the unfinished candidates are moved into
/// this context to wait for their guards.
#[derive(Debug)]
struct LateMatchingCtx {
    candidates: FxIndexMap<DecisionId, MCDCTargetInfo>,
    finished_arms_count: usize,
    otherwise_block: Option<BlockMarkerId>,
    nested_decisions_in_guards: Vec<DecisionId>,
}

impl LateMatchingCtx {
    fn finish_arm(
        &mut self,
        decision_id: DecisionId,
        mut inject_block_marker: impl FnMut(Span) -> BlockMarkerId,
        guard_info: Option<MCDCTargetInfo>,
    ) {
        let Some(MCDCTargetInfo { decision, conditions, .. }) =
            self.candidates.get_mut(&decision_id)
        else {
            return;
        };

        let arm_block = inject_block_marker(decision.span);
        decision.update_end_markers.push(arm_block);
        if let Some(mut guard) = guard_info {
            decision.span = decision.span.to(guard.decision.span);
            let rebase_condition_id =
                |id: ConditionId| ConditionId::from_usize(id.as_usize() + conditions.len());
            for branch in &mut guard.conditions {
                let ConditionInfo { condition_id, true_next_id, false_next_id } =
                    &mut branch.condition_info;
                *condition_id = rebase_condition_id(*condition_id);
                *true_next_id = true_next_id.map(rebase_condition_id);
                *false_next_id = false_next_id.map(rebase_condition_id);
            }
            let guard_entry_id = rebase_condition_id(ConditionId::START);
            conditions
                .iter_mut()
                .filter(|branch| branch.condition_info.true_next_id.is_none())
                .for_each(|branch| branch.condition_info.true_next_id = Some(guard_entry_id));
            conditions.extend(guard.conditions);
            self.nested_decisions_in_guards.extend(guard.nested_decisions_id);
        }
        self.finished_arms_count += 1;
    }

    fn all_arms_finished(&self) -> bool {
        self.finished_arms_count == self.candidates.len()
    }

    fn into_done(mut self) -> (FxIndexMap<DecisionId, MCDCTargetInfo>, Vec<DecisionId>) {
        let Some(all_unmatched_block) = self.otherwise_block.or_else(|| {
            self.candidates.pop().map(|(_, target_info)| target_info.decision.update_end_markers[0])
        }) else {
            return (Default::default(), vec![]);
        };
        // Update test vector bits of a candidate in arm blocks of it and all candidates below it.
        let mut unmatched_blocks: Vec<_> = self
            .candidates
            .values()
            .skip(1)
            .map(|target| target.decision.update_end_markers[0])
            .chain(std::iter::once(all_unmatched_block))
            .rev()
            .collect();
        // Discard condition bitmap of a candidate in arm blocks of all candidates above it.
        // This is to avoid weird result if multiple candidates all match the value.
        let mut discard_blocks = vec![];
        for target in self.candidates.values_mut() {
            target.decision.update_end_markers.extend(unmatched_blocks.clone());
            target.decision.discard_end_markers.extend(discard_blocks.clone());
            unmatched_blocks.pop();
            discard_blocks.push(target.decision.update_end_markers[0]);
        }
        (self.candidates, self.nested_decisions_in_guards)
    }
}

#[derive(Debug, Default)]
pub(super) struct LateMatchingState {
    matching_ctx: Vec<LateMatchingCtx>,
    // Map decision id to guard decision id.
    matching_guard_map: FxIndexMap<DecisionId, DecisionId>,
    guard_decisions_info: FxIndexMap<DecisionId, Option<MCDCTargetInfo>>,
}

impl LateMatchingState {
    pub(super) fn is_empty(&self) -> bool {
        self.matching_ctx.is_empty()
            && self.guard_decisions_info.is_empty()
            && self.matching_guard_map.is_empty()
    }

    fn push_ctx(&mut self, ctx: LateMatchingCtx) {
        self.matching_ctx.push(ctx);
    }

    fn check_decision_exist(&self, decision_id: DecisionId) -> bool {
        let Some(ctx) = self.matching_ctx.last() else { return false };
        let [min, max] = [ctx.candidates.first(), ctx.candidates.last()]
            .map(|opt| *opt.expect("ctx must have candidates").0);
        min <= decision_id && decision_id <= max
    }

    fn declare_guard_for(
        &mut self,
        decision: DecisionId,
        new_guard_id: impl FnOnce() -> DecisionId,
    ) -> Option<DecisionId> {
        if !self.check_decision_exist(decision) {
            return None;
        }

        let guard_id = self.matching_guard_map.entry(decision).or_insert_with(|| {
            let guard_id = new_guard_id();
            self.guard_decisions_info.insert(guard_id, None);
            guard_id
        });

        Some(*guard_id)
    }

    pub(super) fn is_guard_decision(&self, boolean_decision_id: DecisionId) -> bool {
        self.guard_decisions_info.contains_key(&boolean_decision_id)
    }

    pub(super) fn add_guard_decision(
        &mut self,
        boolean_decision_id: DecisionId,
        info: MCDCTargetInfo,
    ) {
        if let Some(Some(guard)) = self.guard_decisions_info.get_mut(&boolean_decision_id) {
            assert_eq!(
                guard.decision.span, info.decision.span,
                "Guard for sub branches must have the same span"
            );
            guard.decision.update_end_markers.extend(info.decision.update_end_markers);
            guard.decision.discard_end_markers.extend(info.decision.discard_end_markers);
            for (this, other) in guard.conditions.iter_mut().zip(info.conditions.into_iter()) {
                assert_eq!(
                    this.condition_info, other.condition_info,
                    "Guard for sub branches must have decision tree"
                );
                assert_eq!(
                    this.span, other.span,
                    "Conditions of guard for sub branches must have the same span"
                );
                this.true_markers.extend(other.true_markers);
                this.false_markers.extend(other.false_markers);
            }
        } else {
            self.guard_decisions_info.insert(boolean_decision_id, Some(info));
        }
    }

    fn finish_arm(
        &mut self,
        id: DecisionId,
        inject_block_marker: impl FnMut(Span) -> BlockMarkerId,
    ) -> Option<LateMatchingCtx> {
        let ctx = self.matching_ctx.last_mut()?;
        let guard = self
            .matching_guard_map
            .swap_remove(&id)
            .and_then(|guard_id| self.guard_decisions_info.swap_remove(&guard_id))
            .flatten();
        ctx.finish_arm(id, inject_block_marker, guard);
        if ctx.all_arms_finished() { self.matching_ctx.pop() } else { None }
    }
}

impl MCDCInfoBuilder {
    fn create_pattern_decision(&mut self, spans: impl Iterator<Item = Span>) -> Vec<DecisionId> {
        self.ensure_active_state();
        let state = self.state_stack.last_mut().expect("ensured just now");
        let decision_info: Vec<_> =
            spans.map(|span| (span, self.decision_id_gen.next_decision_id())).collect();
        assert!(state.is_empty(), "active state for new pattern decision must be empty");
        state.current_ctx = Some(DecisionCtx::new_matching(&decision_info));
        decision_info.into_iter().map(|(_, decision_id)| decision_id).collect()
    }

    fn finish_matching_decision_tree(
        &mut self,
        otherwise_block: Option<BasicBlock>,
        mut inject_block_marker: impl FnMut(Span, BasicBlock) -> BlockMarkerId,
    ) {
        let state = self.current_state_mut();

        let Some((DecisionCtx::Matching(ctx), nested_decisions_id)) = state.take_ctx() else {
            bug!("no processing pattern decision")
        };
        assert!(
            nested_decisions_id.is_empty(),
            "no other decisions can be nested in matching tree"
        );
        let otherwise_block =
            otherwise_block.map(|block| inject_block_marker(Span::default(), block));
        let candidates = ctx.finish_matching_tree(inject_block_marker);
        let late_ctx = LateMatchingCtx {
            candidates,
            finished_arms_count: 0,
            otherwise_block,
            nested_decisions_in_guards: nested_decisions_id,
        };

        self.late_matching_state.push_ctx(late_ctx);
    }

    fn prepare_matching_guard(&mut self, decision_id: DecisionId) {
        assert!(
            self.current_state_mut().current_ctx.is_none(),
            "When visit matching guard there should be no processing decisions"
        );
        if let Some(guard_id) = self
            .late_matching_state
            .declare_guard_for(decision_id, || self.decision_id_gen.next_decision_id())
        {
            self.current_state_mut().current_ctx = Some(DecisionCtx::new_boolean(guard_id));
        }
    }

    fn visit_evaluated_matching_candidate(
        &mut self,
        tcx: TyCtxt<'_>,
        decision_id: DecisionId,
        inject_block_marker: impl FnMut(Span) -> BlockMarkerId,
    ) {
        let Some(ctx) = self.late_matching_state.finish_arm(decision_id, inject_block_marker)
        else {
            return;
        };

        let (targets, mut nested_decisions_id) = ctx.into_done();
        let mut entry_id = None;
        for (id, mut info) in targets.into_iter().rev() {
            info.nested_decisions_id = nested_decisions_id.clone();
            if self.append_mcdc_info(tcx, id, info) {
                nested_decisions_id = vec![id];
                entry_id = Some(id);
            }
        }
        self.on_ctx_finished(tcx, entry_id);
    }
}

impl Builder<'_, '_> {
    /// Prepare for matching decisions if mcdc is enabled. The returned decision ids should be used to generate [`CandidateCovId`] for candidates.
    /// Do nothing if mcdc is not enabled or the candidates is not proper for mcdc.
    pub(crate) fn mcdc_create_matching_decisions(
        &mut self,
        candidates: &mut [&mut Candidate<'_, '_>],
        refutable: bool,
    ) {
        let can_mapped_decisions = refutable || candidates.len() > 1;
        if can_mapped_decisions
            && let Some(coverage_info) = self.coverage_info.as_mut()
            && let Some(mcdc_info) = coverage_info.mcdc_info.as_mut()
        {
            let ids = mcdc_info
                .create_pattern_decision(candidates.iter().map(|candidate| candidate.span()));
            assert_eq!(ids.len(), candidates.len(), "every candidate must get a decision id");
            candidates.iter_mut().zip(ids.into_iter()).for_each(|(candidate, decision_id)| {
                candidate.set_coverage_id(CandidateCovId {
                    decision_id,
                    subcandidate_id: SubcandidateId::ROOT,
                })
            });
        }
    }

    /// Create and assign [`CandidateCovId`] for subcandidates if mcdc is enabled.
    /// Do nothing if mcdc is not enabled or the candidate is ignored.
    pub(crate) fn mcdc_create_subcandidates(
        &mut self,
        candidate_id: CandidateCovId,
        subcandidates: &mut [Candidate<'_, '_>],
    ) {
        if candidate_id.is_valid()
            && let Some(coverage_info) = self.coverage_info.as_mut()
            && let Some(mcdc_info) = coverage_info.mcdc_info.as_mut()
            && let Some(DecisionCtx::Matching(ctx)) = mcdc_info.current_processing_ctx_mut()
            && let Some(candidate_ctx) = ctx.candidates.get_mut(&candidate_id.decision_id)
        {
            subcandidates.iter_mut().for_each(|subcandidate| {
                let id = CandidateCovId {
                    decision_id: candidate_id.decision_id,
                    subcandidate_id: candidate_ctx
                        .next_subcandidate_in(candidate_id.subcandidate_id),
                };
                subcandidate.set_coverage_id(id);
            });
        }
    }

    /// Notify the mcdc builder that some candidates are merged. This can happen on or patterns without bindings.
    /// Do nothing if mcdc is not enabled or the candidates is not proper for mcdc.
    pub(crate) fn mcdc_merge_subcandidates(
        &mut self,
        candidate_id: CandidateCovId,
        subcandidates: impl Iterator<Item = SubcandidateId>,
    ) {
        if candidate_id.is_valid()
            && let Some(coverage_info) = self.coverage_info.as_mut()
            && let Some(mcdc_info) = coverage_info.mcdc_info.as_mut()
            && let Some(DecisionCtx::Matching(ctx)) = mcdc_info.current_processing_ctx_mut()
            && let Some(candidate_ctx) = ctx.candidates.get_mut(&candidate_id.decision_id)
        {
            candidate_ctx.on_merging_subcandidates(candidate_id.subcandidate_id, subcandidates);
        }
    }

    /// Notify the mcdc builder some patterns are preparing for test. This function must be called in same order as match targets are determined.
    /// Do nothing if mcdc is not enabled.
    pub(crate) fn mcdc_visit_pattern_conditions<'a>(
        &mut self,
        patterns: impl Iterator<Item = &'a Vec<MatchCoverageInfo>>,
    ) {
        if let Some(coverage_info) = self.coverage_info.as_mut()
            && let Some(mcdc_info) = coverage_info.mcdc_info.as_mut()
            && let Some(DecisionCtx::Matching(ctx)) = mcdc_info.current_processing_ctx_mut()
        {
            patterns.for_each(|infos| ctx.visit_conditions(infos));
        }
    }

    /// Inform the mcdc builder where the patterns are tested and the blocks to go if the patterns are matched.
    /// Before this call `test_block` must be injected terminator.
    /// If mcdc is not enabled do nothing.
    pub(crate) fn mcdc_match_pattern_conditions(
        &mut self,
        test_block: BasicBlock,
        target_patterns: impl Iterator<Item = (BasicBlock, Vec<MatchCoverageInfo>)>,
    ) {
        if let Some(coverage_info) = self.coverage_info.as_mut()
            && let Some(mcdc_info) = coverage_info.mcdc_info.as_mut()
            && let Some(DecisionCtx::Matching(ctx)) = mcdc_info.current_processing_ctx_mut()
        {
            ctx.match_conditions(&mut self.cfg, test_block, target_patterns);
        }
    }

    /// Notify the mcdc builder matching tree is finished lowering.
    /// This function should be called before diving into arms and guards.
    /// The `otherwise_block` should be provided if and only if the candidates are from refutable statements (`if let` or `let else`).
    /// Do nothing if mcdc is not enabled.
    pub(crate) fn mcdc_finish_matching_tree(
        &mut self,
        mut candidates: impl Iterator<Item = CandidateCovId>,
        otherwise_block: Option<BasicBlock>,
    ) {
        let has_decision = candidates.any(CandidateCovId::is_valid);
        if has_decision
            && let Some(coverage_info) = self.coverage_info.as_mut()
            && let Some(mcdc_info) = coverage_info.mcdc_info.as_mut()
        {
            let inject_block_marker = |span: Span, block: BasicBlock| {
                coverage_info.markers.inject_block_marker(
                    &mut self.cfg,
                    SourceInfo { span, scope: self.source_scope },
                    block,
                )
            };
            mcdc_info.finish_matching_decision_tree(otherwise_block, inject_block_marker);
        }
    }

    /// Notify the mcdc builder a guard is to be lowered.
    /// Do nothing if mcdc is not enabled or the candidates is not proper for mcdc.
    pub(crate) fn mcdc_visit_matching_guard(&mut self, coverage_id: CandidateCovId) {
        if coverage_id.is_valid()
            && let Some(coverage_info) = self.coverage_info.as_mut()
            && let Some(mcdc_info) = coverage_info.mcdc_info.as_mut()
        {
            mcdc_info.prepare_matching_guard(coverage_id.decision_id);
        }
    }

    /// Notify the mcdc builder a candidate has been totally processed.
    /// Do nothing if mcdc is not enabled or the candidates is not proper for mcdc.
    pub(crate) fn mcdc_visit_matching_decision_end(
        &mut self,
        coverage_id: CandidateCovId,
        arm_block: BasicBlock,
    ) {
        if coverage_id.is_valid()
            && let Some(coverage_info) = self.coverage_info.as_mut()
            && let Some(mcdc_info) = coverage_info.mcdc_info.as_mut()
        {
            let inject_block_marker = |span: Span| {
                coverage_info.markers.inject_block_marker(
                    &mut self.cfg,
                    SourceInfo { span, scope: self.source_scope },
                    arm_block,
                )
            };
            mcdc_info.visit_evaluated_matching_candidate(
                self.tcx,
                coverage_id.decision_id,
                inject_block_marker,
            );
        }
    }
}
