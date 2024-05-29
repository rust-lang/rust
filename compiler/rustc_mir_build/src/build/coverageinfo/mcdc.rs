use std::cell::RefCell;
use std::collections::{BTreeSet, VecDeque};
use std::rc::Rc;

use rustc_data_structures::fx::FxIndexMap;
use rustc_middle::bug;
use rustc_middle::mir::coverage::{
    BlockMarkerId, ConditionId, ConditionInfo, MCDCBranchMarkers, MCDCBranchSpan, MCDCDecisionSpan,
};
use rustc_middle::mir::{BasicBlock, SourceInfo};
use rustc_middle::thir::LogicalOp;
use rustc_middle::ty::TyCtxt;
use rustc_span::Span;

use crate::build::{Builder, CFG};
use crate::errors::{MCDCExceedsConditionNumLimit, MCDCExceedsDecisionDepth};

/// The MCDC bitmap scales exponentially (2^n) based on the number of conditions seen,
/// So llvm sets a maximum value prevents the bitmap footprint from growing too large without the user's knowledge.
/// This limit may be relaxed if the [upstream change](https://github.com/llvm/llvm-project/pull/82448) is merged.
const MAX_CONDITIONS_NUM_IN_DECISION: usize = 6;
const MAX_DECISION_DEPTH: u16 = 0xFF;

struct BooleanDecisionCtx {
    /// To construct condition evaluation tree.
    decision_info: MCDCDecisionSpan,
    decision_stack: VecDeque<ConditionInfo>,
    conditions: Vec<MCDCBranchSpan>,
}

impl BooleanDecisionCtx {
    fn new(decision_depth: u16) -> Self {
        Self {
            decision_stack: VecDeque::new(),
            decision_info: MCDCDecisionSpan {
                span: Span::default(),
                conditions_num: 0,
                end_markers: vec![],
                decision_depth,
            },
            conditions: vec![],
        }
    }

    // At first we assign ConditionIds for each sub expression.
    // If the sub expression is composite, re-assign its ConditionId to its LHS and generate a new ConditionId for its RHS.
    //
    // Example: "x = (A && B) || (C && D) || (D && F)"
    //
    //      Visit Depth1:
    //              (A && B) || (C && D) || (D && F)
    //              ^-------LHS--------^    ^-RHS--^
    //                      ID=1              ID=2
    //
    //      Visit LHS-Depth2:
    //              (A && B) || (C && D)
    //              ^-LHS--^    ^-RHS--^
    //                ID=1        ID=3
    //
    //      Visit LHS-Depth3:
    //               (A && B)
    //               LHS   RHS
    //               ID=1  ID=4
    //
    //      Visit RHS-Depth3:
    //                         (C && D)
    //                         LHS   RHS
    //                         ID=3  ID=5
    //
    //      Visit RHS-Depth2:              (D && F)
    //                                     LHS   RHS
    //                                     ID=2  ID=6
    //
    //      Visit Depth1:
    //              (A && B)  || (C && D)  || (D && F)
    //              ID=1  ID=4   ID=3  ID=5   ID=2  ID=6
    //
    // A node ID of '0' always means MC/DC isn't being tracked.
    //
    // If a "next" node ID is '0', it means it's the end of the test vector.
    //
    // As the compiler tracks expression in pre-order, we can ensure that condition info of parents are always properly assigned when their children are visited.
    // - If the op is AND, the "false_next" of LHS and RHS should be the parent's "false_next". While "true_next" of the LHS is the RHS, the "true next" of RHS is the parent's "true_next".
    // - If the op is OR, the "true_next" of LHS and RHS should be the parent's "true_next". While "false_next" of the LHS is the RHS, the "false next" of RHS is the parent's "false_next".
    fn record_conditions(&mut self, op: LogicalOp) {
        let parent_condition = self.decision_stack.pop_back().unwrap_or_default();
        let lhs_id = if parent_condition.condition_id == ConditionId::NONE {
            self.decision_info.conditions_num += 1;
            ConditionId::from(self.decision_info.conditions_num)
        } else {
            parent_condition.condition_id
        };

        self.decision_info.conditions_num += 1;
        let rhs_condition_id = ConditionId::from(self.decision_info.conditions_num);

        let (lhs, rhs) = match op {
            LogicalOp::And => {
                let lhs = ConditionInfo {
                    condition_id: lhs_id,
                    true_next_id: rhs_condition_id,
                    false_next_id: parent_condition.false_next_id,
                };
                let rhs = ConditionInfo {
                    condition_id: rhs_condition_id,
                    true_next_id: parent_condition.true_next_id,
                    false_next_id: parent_condition.false_next_id,
                };
                (lhs, rhs)
            }
            LogicalOp::Or => {
                let lhs = ConditionInfo {
                    condition_id: lhs_id,
                    true_next_id: parent_condition.true_next_id,
                    false_next_id: rhs_condition_id,
                };
                let rhs = ConditionInfo {
                    condition_id: rhs_condition_id,
                    true_next_id: parent_condition.true_next_id,
                    false_next_id: parent_condition.false_next_id,
                };
                (lhs, rhs)
            }
        };
        // We visit expressions tree in pre-order, so place the left-hand side on the top.
        self.decision_stack.push_back(rhs);
        self.decision_stack.push_back(lhs);
    }

    fn finish_two_way_branch(
        &mut self,
        span: Span,
        true_marker: BlockMarkerId,
        false_marker: BlockMarkerId,
    ) {
        let condition_info = self.decision_stack.pop_back().unwrap_or_default();
        if condition_info.true_next_id == ConditionId::NONE {
            self.decision_info.end_markers.push(true_marker);
        }
        if condition_info.false_next_id == ConditionId::NONE {
            self.decision_info.end_markers.push(false_marker);
        }

        self.conditions.push(MCDCBranchSpan {
            span,
            condition_info: Some(condition_info),
            markers: MCDCBranchMarkers::Boolean(true_marker, false_marker),
            decision_depth: self.decision_info.decision_depth,
        });
        // In case this decision had only one condition
        self.decision_info.conditions_num = self.decision_info.conditions_num.max(1);
    }

    fn finished(&self) -> bool {
        self.decision_stack.is_empty()
    }
}

#[derive(Default)]
struct MatchArmRecord {
    matched_spans: Vec<Span>,
    true_next_pattern: Option<Span>,
    false_next_pattern: Option<Span>,
    // The key is test block while the value is matched block.
    test_pairs: FxIndexMap<BasicBlock, BasicBlock>,
}
struct PatternDecisionCtx {
    decision_depth_base: u16,
    // All patterns leading to branches.
    matching_arms: FxIndexMap<Span, MatchArmRecord>,
    // Patterns unknown for their predecessors yet.
    start_tests: FxIndexMap<BasicBlock, Span>,
    // Patterns ending the process once they are matched.
    end_tests: FxIndexMap<BasicBlock, Span>,
    // Patterns as fallback of patterns in another decision.
    // For example,
    // ```rust
    // (Patter::A | PatternB, Pattern::C | Pattern::D) => { /* ... */}
    // (Patter::A,            Pattern::C | Pattern::D) => { /* ... */}
    // ```
    // The first part of second arm is tested only if `Patter::A | Pattern::B` failed.
    // Also they might match different places:
    // ```rust
    // (Patter::A, Pattern::B) => { /* ... */}
    // (_,         Pattern::C)
    // ```
    // In such case `Pattern::C` is tested as long as `Pattern::A` failed, hence it's
    // a remainder of `Pattern::A`.
    remainder_ends: FxIndexMap<BasicBlock, Span>,
}

impl PatternDecisionCtx {
    fn new(decision_depth_base: u16) -> Self {
        Self {
            decision_depth_base,
            matching_arms: FxIndexMap::default(),
            start_tests: FxIndexMap::default(),
            end_tests: FxIndexMap::default(),
            remainder_ends: FxIndexMap::default(),
        }
    }

    // This method is valid when following invariants hold:
    // 1. In same decision, patterns called this in different time shall be matched all together.
    // 2. Targets passed to this method in same time are joint with `|`
    fn visit_matching_arms(
        &mut self,
        test_block: BasicBlock,
        mut targets: Vec<(BasicBlock, Vec<Span>)>,
        otherwise_block: BasicBlock,
    ) {
        let mut visiting_arms: FxIndexMap<_, _> = targets
            .iter()
            .filter_map(|(_, matched_spans)| {
                self.matching_arms.swap_remove_entry(&matched_spans[0])
            })
            .collect();

        let entry_span = targets.first().expect("targets mut be non empty").1[0];

        let mut false_next_arm = self.start_tests.swap_remove(&otherwise_block);
        // Traverse conditions in reverse order to assign "false next" properly.
        while let Some((matched_block, matched_spans)) = targets.pop() {
            let key_span = matched_spans[0];
            let record = visiting_arms
                .entry(key_span)
                .or_insert_with(|| MatchArmRecord { matched_spans, ..Default::default() });
            // Same pattern tests might appear at different blocks as predecessors of same tests.
            // Thus always update `start_tests` even though the `next_test` of the record has already been known.
            let next_test = self.start_tests.swap_remove(&matched_block);
            assert!(
                record.true_next_pattern.is_none()
                    || next_test.is_none()
                    || record.true_next_pattern == next_test,
                "any record should have unique true next test pattern"
            );
            // `partial_matched` means the pattern is not _full matched_ here. That is, it should be matched
            // at some succeeding blocks, while acts as an “otherwise” target branch to its full matched peers here.
            // See `Builder::sort_candidate` for which patterns may be not full matched.
            let partial_matched = next_test.is_some_and(|next_span| next_span == key_span);
            if !partial_matched {
                record.test_pairs.insert(test_block, matched_block);
                record.true_next_pattern = record.true_next_pattern.or(next_test);
            }

            if record.true_next_pattern.is_some() {
                self.end_tests.swap_remove(&matched_block);
            } else {
                self.end_tests.insert(matched_block, key_span);
            }

            if false_next_arm.is_none() {
                self.remainder_ends.insert(otherwise_block, key_span);
            }

            // If the record is `partial_matched` at this test, it might already have been assigned a "false next"
            // when it was `full_matched`.
            assert!(
                record.false_next_pattern.is_none()
                    || false_next_arm.is_none()
                    || record.false_next_pattern == false_next_arm,
                "any record should have unique false next test pattern"
            );
            record.false_next_pattern = record.false_next_pattern.or(false_next_arm);

            false_next_arm = Some(key_span);
        }
        // There are only one of two occasions when we visit this test:
        // 1. The direct predecessor has been visited before. In this case, the predecessor
        //    must be in `end_tests` or `remainder_ends` and should set its true next to the test entry.
        // 2. Its direct predecessor has not been visited yet. Then we insert the test entry to `start_tests`.
        if let Some(span) = self.end_tests.swap_remove(&test_block) {
            let predecessor = self.matching_arms.get_mut(&span).expect("record must exist here");
            if span != entry_span {
                predecessor.true_next_pattern = Some(entry_span);
            } else if let Some(false_test_block) = predecessor
                .test_pairs
                .iter()
                .find_map(|(&tblk, &mblk)| (mblk == test_block).then_some(tblk))
            {
                predecessor.test_pairs.swap_remove(&false_test_block);
            }
        } else if let Some(span) = self.remainder_ends.swap_remove(&test_block) {
            let predecessor = self.matching_arms.get_mut(&span).expect("record must exist here");
            predecessor.false_next_pattern = Some(entry_span);
        } else {
            self.start_tests.insert(test_block, entry_span);
        }

        self.matching_arms.extend(visiting_arms);
    }

    fn finish(
        mut self,
        cfg: &mut CFG<'_>,
        entry_block: BasicBlock,
        decisions: impl Iterator<Item = Span>,
        mut inject_block_marker: impl FnMut(&mut CFG<'_>, Span, BasicBlock) -> BlockMarkerId,
    ) -> Vec<(MCDCDecisionSpan, Vec<MCDCBranchSpan>)> {
        self.link_condition_gaps(entry_block, cfg);
        let Self {
            decision_depth_base,
            mut matching_arms,
            start_tests,
            end_tests: _,
            remainder_ends: _,
        } = self;

        let mut decisions: FxIndexMap<Span, Vec<MCDCBranchSpan>> =
            decisions.into_iter().map(|span| (span, Vec::new())).collect();

        let mut block_markers_map: FxIndexMap<BasicBlock, BlockMarkerId> = Default::default();

        let mut get_block_marker = move |span: Span, block: BasicBlock| {
            *block_markers_map.entry(block).or_insert_with(|| inject_block_marker(cfg, span, block))
        };

        // LLVM implementation requires the entry condition should be assigned to id 1.
        // See https://github.com/rust-lang/rust/issues/79649#issuecomment-2099808951 for details.
        // We assign condition id at last stage because the entry test could appear
        // in any order during recording.
        let mut condition_id_counter: usize = 0;
        let mut next_condition_id = || {
            condition_id_counter += 1;
            ConditionId::from(condition_id_counter)
        };
        let mut test_entries = VecDeque::from_iter(
            start_tests.into_values().zip(std::iter::once(next_condition_id())),
        );

        let mut assigned_entries = FxIndexMap::from_iter(test_entries.iter().copied());
        while let Some((span, condition_id)) = test_entries.pop_front() {
            let MatchArmRecord { matched_spans, true_next_pattern, false_next_pattern, test_pairs } =
                matching_arms.swap_remove(&span).expect("entries must have been recorded");
            let mut condition_info = ConditionInfo {
                condition_id,
                true_next_id: ConditionId::NONE,
                false_next_id: ConditionId::NONE,
            };

            // Assign "global" condition id here. Normalize them in `PatternDecisionCtx::formalize_decision`.
            if let Some(false_next) = false_next_pattern {
                condition_info.false_next_id = next_condition_id();
                assigned_entries.insert(false_next, condition_info.false_next_id);
                test_entries.push_back((false_next, condition_info.false_next_id));
            }

            if let Some(true_next) = true_next_pattern {
                // Multiple patterns joint with `|` may or may not share same `true_next`.
                // That's why `assigned_entries` is used to trace the conditions already assigned a condition id.
                condition_info.true_next_id =
                    assigned_entries.get(&true_next).copied().unwrap_or_else(|| {
                        let id = next_condition_id();
                        test_entries.push_back((true_next, id));
                        assigned_entries.insert(true_next, id);
                        id
                    });
            }

            let (test_markers, matched_markers): (Vec<_>, Vec<_>) = test_pairs
                .into_iter()
                .map(|(test_block, matched_block)| {
                    (get_block_marker(span, test_block), get_block_marker(span, matched_block))
                })
                .unzip();

            for condition_span in matched_spans {
                let Some((_, conditions)) =
                    decisions.iter_mut().find(|(span, _)| span.contains(condition_span))
                else {
                    bug!("all conditions should be contained in a decision");
                };
                conditions.push(MCDCBranchSpan {
                    span: condition_span,
                    decision_depth: decision_depth_base,
                    condition_info: Some(condition_info),
                    markers: MCDCBranchMarkers::PatternMatching(
                        test_markers.clone(),
                        matched_markers.clone(),
                    ),
                });
            }
        }

        let mut arm_offset = decision_depth_base;
        let mut decision_offset = move |conditions: &[MCDCBranchSpan]| {
            // It's common that many of matching arms contain only one pattern, we won't instrument mcdc for these arms
            // at last. So don't increase decision depth for those arms in case there were unused mcdc parameters.
            if conditions.len() > 1 && conditions.len() <= MAX_CONDITIONS_NUM_IN_DECISION {
                let depth = arm_offset;
                arm_offset = arm_offset.saturating_add(1);
                depth
            } else {
                0
            }
        };
        decisions
            .into_iter()
            // Irrefutable patterns like `_` have no conditions
            .filter(|(_, conditions)| !conditions.is_empty())
            .map(|(decision_span, conditions)| {
                let depth = decision_offset(&conditions);
                Self::formalize_decision(decision_span, depth, conditions)
            })
            .collect()
    }

    // Gaps appear when the matched value is ignored. For instance, `Some(_) | Ok(_), IpAddr::V4(_) | IpAddr::V6(_)`.
    // In such cases the matched blocks of `Some(_)` and `Ok(_)` will have empty successors which converge
    // to the unique test block of  `IpAddr::V4(_) | IpAddr::V6(_)`. As a result the test block cannot find its direct
    // predecessors, which are not matched blocks of any patterns, at `PatternDecisionCtx::visit_matching_arms`
    // (in fact they were not connected at that moment).
    fn link_condition_gaps(&mut self, entry_block: BasicBlock, cfg: &mut CFG<'_>) {
        if self.start_tests.len() < 2 {
            return;
        }

        for (&end, span) in &self.end_tests {
            let mut successor = Some(end);
            while let Some(block) = successor.take() {
                let Some(next_blocks) = cfg
                    .block_data(block)
                    .terminator
                    .as_ref()
                    .map(|term| term.successors().collect::<Vec<_>>())
                else {
                    break;
                };
                match next_blocks.as_slice() {
                    &[unique_successor] => {
                        if let Some(&next_span) = self.start_tests.get(&unique_successor) {
                            let end_record = self
                                .matching_arms
                                .get_mut(span)
                                .expect("end tests must be recorded");
                            end_record.true_next_pattern = Some(next_span);
                        } else {
                            successor = Some(unique_successor);
                        }
                    }
                    _ => break,
                }
            }
        }

        // There might be unreached arms in match guard. E.g
        // ```rust
        // match pattern {
        //     (A | B, C | D) => {},
        //     (B, D) => {},
        //     _ => {}
        // }
        // ```
        // Clearly the arm `(B, D)` is covered by the first arm so it should be unreached.
        // Such arms can cause unresolved start tests here, just ignore them.
        self.start_tests.retain(|block, _| *block == entry_block);
        // In case no block in `start_tests` is `entry_block`.
        assert_eq!(self.start_tests.len(), 1, "still some gaps exist in mcdc pattern decision");
    }

    fn formalize_decision(
        decision_span: Span,
        decision_depth: u16,
        mut conditions: Vec<MCDCBranchSpan>,
    ) -> (MCDCDecisionSpan, Vec<MCDCBranchSpan>) {
        conditions.sort_by(|lhs, rhs| {
            if lhs.span.contains(rhs.span) {
                std::cmp::Ordering::Less
            } else {
                lhs.span.cmp(&rhs.span)
            }
        });

        // Suppose the pattern is `A(Some(_))`, it generate two conditions: `A` and `Some`. The span of
        // `A` would be `A(Some(_))`, including the span of `Some`. To make it look nicer we extract span of sub patterns
        // so that span of `A` would be `A(` while `Some` is `Some(_)`.
        for idx in 0..conditions.len().saturating_sub(1) {
            let mut span = conditions[idx].span;
            for sub_branch in conditions.iter().skip(idx + 1) {
                if span.contains(sub_branch.span) {
                    span = span.until(sub_branch.span);
                } else {
                    break;
                }
            }
            conditions[idx].span = span;
        }

        let mut conditions_info: Vec<_> = conditions
            .iter_mut()
            .map(|branch| {
                branch.decision_depth = decision_depth;
                branch.condition_info.as_mut().expect("condition info has been created before")
            })
            .collect();
        conditions_info.sort_by_key(|info| info.condition_id);

        let reassign_condition_id_map: FxIndexMap<ConditionId, ConditionId> = conditions_info
            .iter()
            .enumerate()
            .map(|(idx, info)| (info.condition_id, ConditionId::from_usize(idx + 1)))
            .collect();

        let mut start_conditions: BTreeSet<ConditionId> =
            reassign_condition_id_map.values().copied().collect();
        for info in &mut conditions_info {
            info.condition_id = *reassign_condition_id_map
                .get(&info.condition_id)
                .expect("all conditions have been assigned a new id above");

            [&mut info.true_next_id, &mut info.false_next_id].into_iter().for_each(|next_id| {
                if *next_id != ConditionId::NONE
                    && let Some(next_in_decision) = reassign_condition_id_map.get(&*next_id)
                {
                    *next_id = *next_in_decision;
                    start_conditions.remove(next_in_decision);
                } else {
                    *next_id = ConditionId::NONE;
                }
            });
        }
        assert!(
            start_conditions.remove(&ConditionId::from_usize(1)),
            "Condition 1 should be always the start"
        );
        // Some conditions might not find their predecessors due to gaps between tests.
        // For instance,
        // ```rust
        // (Pat::A(Some(_)), _ ) => { /* ... */ },
        // (Pat::A(None),    _) => { /* ... */ },
        // ```
        // Say `Pat::A` is C1, `Some(_)` is C2, `None` is C3. In the second decision we would
        // reassign id of `Pat::A` with true next `None` because C2 is not condition of this decision.
        // `None` becomes "fake start" here as a result of no condition's next linking to it.
        // Considering condition id of `fake_start` is always less than its first predecessor
        // due to traverse order in `PatternDecisionCtx::finish`, we can fix these gaps in following way.
        while let Some(fake_start) = start_conditions.pop_first() {
            let Some(first_predecessor) = conditions_info.iter_mut().find(|info| {
                info.condition_id < fake_start && info.true_next_id == ConditionId::NONE
            }) else {
                bug!("except condition 1 all conditions must have at least one predecessor");
            };
            first_predecessor.true_next_id = fake_start;

            // Note that sibling predecessors probably have larger condition ids.
            let mut sibling_predecessor = first_predecessor.false_next_id;
            while sibling_predecessor != ConditionId::NONE {
                let predecessor_info = conditions_info
                    .get_mut(sibling_predecessor.as_usize() - 1)
                    .expect("condition ids are properly assigned with the index");
                assert_eq!(
                    predecessor_info.true_next_id,
                    ConditionId::NONE,
                    "condition is joint with its false next by '|' in same decision, so they should have same true next"
                );
                predecessor_info.true_next_id = fake_start;
                sibling_predecessor = predecessor_info.false_next_id;
            }
        }

        let decision = MCDCDecisionSpan {
            span: decision_span,
            decision_depth,
            conditions_num: conditions.len(),
            end_markers: vec![],
        };

        (decision, conditions)
    }
}

enum DecisionCtx {
    Boolean(BooleanDecisionCtx),
    Pattern(PatternDecisionCtx),
}

pub struct MCDCInfoBuilder {
    normal_branch_spans: Vec<MCDCBranchSpan>,
    mcdc_spans: Vec<(MCDCDecisionSpan, Vec<MCDCBranchSpan>)>,
    decision_ctx_stack: Vec<DecisionCtx>,
    decision_ends: FxIndexMap<Span, Rc<RefCell<Vec<BlockMarkerId>>>>,
    base_depth: u16,
}

impl MCDCInfoBuilder {
    pub fn new() -> Self {
        Self {
            normal_branch_spans: vec![],
            mcdc_spans: vec![],
            decision_ctx_stack: vec![],
            decision_ends: FxIndexMap::default(),
            base_depth: 0,
        }
    }

    fn next_decision_depth(&self) -> u16 {
        u16::try_from(self.decision_ctx_stack.len()).expect(
            "decision depth did not fit in u16, this is likely to be an instrumentation error",
        )
    }

    fn ensure_boolean_decision(&mut self, span: Span) -> &mut BooleanDecisionCtx {
        if self.base_depth == self.next_decision_depth()
            || self
                .decision_ctx_stack
                .last()
                .is_some_and(|ctx| !matches!(ctx, DecisionCtx::Boolean(_)))
        {
            let depth = self.next_decision_depth();
            self.decision_ctx_stack.push(DecisionCtx::Boolean(BooleanDecisionCtx::new(depth)));
        } else {
            assert!(
                self.base_depth <= self.next_decision_depth(),
                "expected depth shall not skip next decision depth"
            );
        }
        let Some(DecisionCtx::Boolean(ctx)) = self.decision_ctx_stack.last_mut() else {
            unreachable!("ensured above")
        };

        if ctx.decision_info.span == Span::default() {
            ctx.decision_info.span = span;
        } else {
            ctx.decision_info.span = ctx.decision_info.span.to(span);
        }
        ctx
    }

    fn try_finish_boolean_decision(&mut self, tcx: TyCtxt<'_>) {
        if !self.decision_ctx_stack.last().is_some_and(|decision| match decision {
            DecisionCtx::Boolean(ctx) => ctx.finished(),
            _ => false,
        }) {
            return;
        }
        let Some(DecisionCtx::Boolean(BooleanDecisionCtx {
            decision_info,
            decision_stack: _,
            conditions,
        })) = self.decision_ctx_stack.pop()
        else {
            unreachable!("has checked above");
        };
        self.append_mcdc_info(tcx, decision_info, conditions);
    }

    fn create_pattern_decision(&mut self) {
        if self
            .decision_ctx_stack
            .last()
            .is_some_and(|decision| matches!(decision, DecisionCtx::Pattern(_)))
        {
            bug!("has been processing a pattern decision");
        }
        let depth = self.next_decision_depth();
        self.decision_ctx_stack.push(DecisionCtx::Pattern(PatternDecisionCtx::new(depth)));
    }

    fn current_pattern_decision(&mut self) -> Option<&mut PatternDecisionCtx> {
        self.decision_ctx_stack.last_mut().and_then(|decision| match decision {
            DecisionCtx::Pattern(ctx) => Some(ctx),
            _ => None,
        })
    }

    fn finish_pattern_decision(
        &mut self,
        cfg: &mut CFG<'_>,
        tcx: TyCtxt<'_>,
        entry_block: BasicBlock,
        candidates: impl Iterator<Item = Span>,
        inject_block_marker: impl FnMut(&mut CFG<'_>, Span, BasicBlock) -> BlockMarkerId,
    ) {
        if !self
            .decision_ctx_stack
            .last()
            .is_some_and(|decision| matches!(decision, DecisionCtx::Pattern(_)))
        {
            bug!("no processing pattern decision");
        }
        let Some(DecisionCtx::Pattern(decision_ctx)) = self.decision_ctx_stack.pop() else {
            unreachable!("has checked above");
        };

        let end_blocks = Rc::new(RefCell::new(vec![]));

        // Some candidates such as wildcard `_` are not taken as decisions but they still lead to some end blocks.
        // So we trace them in `decision_ends` too.
        let decision_ends = candidates.map(|span| (span, end_blocks.clone())).collect::<Vec<_>>();

        let decisions = decision_ctx.finish(
            cfg,
            entry_block,
            decision_ends.iter().map(|(span, _)| *span),
            inject_block_marker,
        );
        self.decision_ends.extend(decision_ends);
        for (decision, conditions) in decisions {
            self.append_mcdc_info(tcx, decision, conditions);
        }
    }

    fn add_ends_to_decision(
        &mut self,
        decision_span: Span,
        end_markers: impl Iterator<Item = BlockMarkerId>,
    ) {
        let Some(markers) = self.decision_ends.get_mut(&decision_span) else {
            bug!("unknown decision");
        };
        markers.borrow_mut().extend(end_markers);
    }

    fn append_normal_branches(&mut self, mut conditions: Vec<MCDCBranchSpan>) {
        conditions.iter_mut().for_each(|branch| branch.condition_info = None);
        self.normal_branch_spans.extend(conditions);
    }

    fn append_mcdc_info(
        &mut self,
        tcx: TyCtxt<'_>,
        decision: MCDCDecisionSpan,
        conditions: Vec<MCDCBranchSpan>,
    ) {
        match (decision.conditions_num, decision.decision_depth) {
            (0, _) => {
                unreachable!("Decision with no condition is not expected");
            }
            (2..=MAX_CONDITIONS_NUM_IN_DECISION, 0..=MAX_DECISION_DEPTH) => {
                self.mcdc_spans.push((decision, conditions));
            }
            // MCDC is equivalent to normal branch coverage if number of conditions is 1, so ignore these decisions.
            // See comment of `MAX_CONDITIONS_NUM_IN_DECISION` for why decisions with oversized conditions are ignored.
            _ => {
                // Nested decisions with only one condition should be taken as condition of its outer decision directly
                // in case there were duplicate branches in branch reports.
                if decision.decision_depth == 0 || conditions.len() > 1 {
                    self.append_normal_branches(conditions);
                }
                if decision.conditions_num > MAX_CONDITIONS_NUM_IN_DECISION {
                    tcx.dcx().emit_warn(MCDCExceedsConditionNumLimit {
                        span: decision.span,
                        conditions_num: decision.conditions_num,
                        max_conditions_num: MAX_CONDITIONS_NUM_IN_DECISION,
                    });
                }

                if decision.decision_depth > MAX_DECISION_DEPTH {
                    tcx.dcx().emit_warn(MCDCExceedsDecisionDepth {
                        span: decision.span,
                        max_decision_depth: MAX_DECISION_DEPTH.into(),
                    });
                }
            }
        }
    }

    pub fn visit_evaluated_condition(
        &mut self,
        tcx: TyCtxt<'_>,
        span: Span,
        true_block: BasicBlock,
        false_block: BasicBlock,
        mut inject_block_marker: impl FnMut(BasicBlock) -> BlockMarkerId,
    ) {
        let true_marker = inject_block_marker(true_block);
        let false_marker = inject_block_marker(false_block);
        let decision = self.ensure_boolean_decision(span);
        decision.finish_two_way_branch(span, true_marker, false_marker);
        self.try_finish_boolean_decision(tcx);
    }

    pub fn into_done(self) -> (Vec<MCDCDecisionSpan>, Vec<MCDCBranchSpan>) {
        let Self {
            normal_branch_spans: mut branch_spans,
            mcdc_spans,
            decision_ctx_stack: _,
            mut decision_ends,
            base_depth: _,
        } = self;
        let mut decisions = vec![];
        for (mut decision, conditions) in mcdc_spans {
            if let Some(end_markers) = decision_ends.swap_remove(&decision.span) {
                decision.end_markers = end_markers.borrow().clone();
            }
            decisions.push(decision);
            branch_spans.extend(conditions);
        }
        (decisions, branch_spans)
    }
}

impl Builder<'_, '_> {
    pub(crate) fn visit_coverage_branch_operation(&mut self, logical_op: LogicalOp, span: Span) {
        if let Some(branch_info) = self.coverage_branch_info.as_mut()
            && let Some(mcdc_info) = branch_info.mcdc_info.as_mut()
        {
            let decision = mcdc_info.ensure_boolean_decision(span);
            decision.record_conditions(logical_op);
        }
    }

    pub(crate) fn visit_mcdc_pattern_matching_conditions(
        &mut self,
        test_block: BasicBlock,
        pattern_targets: Vec<(BasicBlock, Vec<Span>)>,
        otherwise_block: BasicBlock,
    ) {
        if let Some(branch_info) = self.coverage_branch_info.as_mut()
            && let Some(mcdc_info) = branch_info.mcdc_info.as_mut()
            && let Some(decision_ctx) = mcdc_info.current_pattern_decision()
        {
            decision_ctx.visit_matching_arms(test_block, pattern_targets, otherwise_block);
        }
    }

    pub(crate) fn mcdc_create_pattern_matching_decision(&mut self) {
        if let Some(branch_info) = self.coverage_branch_info.as_mut()
            && let Some(mcdc_info) = branch_info.mcdc_info.as_mut()
        {
            mcdc_info.create_pattern_decision();
        }
    }

    pub(crate) fn mcdc_finish_pattern_matching_decision(
        &mut self,
        entry_block: BasicBlock,
        candidates: impl Iterator<Item = Span>,
    ) {
        if let Some(branch_info) = self.coverage_branch_info.as_mut()
            && let Some(mcdc_info) = branch_info.mcdc_info.as_mut()
        {
            let inject_block_marker = |cfg: &mut CFG<'_>, span, block| {
                branch_info.markers.inject_block_marker(
                    cfg,
                    SourceInfo { span, scope: self.source_scope },
                    block,
                )
            };
            mcdc_info.finish_pattern_decision(
                &mut self.cfg,
                self.tcx,
                entry_block,
                candidates,
                inject_block_marker,
            );
        }
    }

    pub(crate) fn mcdc_visit_decision_ends(
        &mut self,
        decision_span: Span,
        end_blocks: impl IntoIterator<Item = BasicBlock>,
    ) {
        if let Some(branch_info) = self.coverage_branch_info.as_mut()
            && let Some(mcdc_info) = branch_info.mcdc_info.as_mut()
        {
            let end_markers = end_blocks.into_iter().map(|block| {
                branch_info.markers.inject_block_marker(
                    &mut self.cfg,
                    SourceInfo { span: decision_span, scope: self.source_scope },
                    block,
                )
            });
            mcdc_info.add_ends_to_decision(decision_span, end_markers);
        }
    }

    pub(crate) fn mcdc_increment_depth_if_enabled(&mut self) {
        if let Some(branch_info) = self.coverage_branch_info.as_mut()
            && let Some(mcdc_info) = branch_info.mcdc_info.as_mut()
        {
            mcdc_info.base_depth = mcdc_info.next_decision_depth().max(mcdc_info.base_depth + 1);
        };
    }

    pub(crate) fn mcdc_decrement_depth_if_enabled(&mut self) {
        if let Some(branch_info) = self.coverage_branch_info.as_mut()
            && let Some(mcdc_info) = branch_info.mcdc_info.as_mut()
        {
            mcdc_info.base_depth -= 1;
        };
    }
}
