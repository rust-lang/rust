use std::ops::Index;

use derive_where::derive_where;
use rustc_index::IndexVec;

use crate::search_graph::{
    AvailableDepth, CandidateHeadUsages, Cx, CycleHeads, HeadUsages, NestedGoals, PathKind,
};

rustc_index::newtype_index! {
    #[orderable]
    #[gate_rustc_only]
    pub(super) struct StackDepth {}
}

/// Stack entries of the evaluation stack. Its fields tend to be lazily updated
/// when popping a child goal or completely immutable.
#[derive_where(Debug; X: Cx)]
pub(super) struct StackEntry<X: Cx> {
    pub input: X::Input,

    /// Whether proving this goal is a coinductive step.
    ///
    /// This is used when encountering a trait solver cycle to
    /// decide whether the initial provisional result of the cycle.
    pub step_kind_from_parent: PathKind,

    /// The available depth of a given goal, immutable.
    pub available_depth: AvailableDepth,

    /// The maximum depth required while evaluating this goal.
    pub required_depth: usize,

    /// Starts out as `None` and gets set when rerunning this
    /// goal in case we encounter a cycle.
    pub provisional_result: Option<X::Result>,

    /// All cycle heads this goal depends on. Lazily updated and only
    /// up-to date for the top of the stack.
    pub heads: CycleHeads,

    /// Whether evaluating this goal encountered overflow. Lazily updated.
    pub encountered_overflow: bool,

    /// Whether and how this goal has been used as a cycle head. Lazily updated.
    pub usages: Option<HeadUsages>,

    /// We want to be able to ignore head usages if they happen inside of candidates
    /// which don't impact the result of a goal. This enables us to avoid rerunning goals
    /// and is also used when rebasing provisional cache entries.
    ///
    /// To implement this, we track all usages while evaluating a candidate. If this candidate
    /// then ends up ignored, we manually remove its usages from `usages` and `heads`.
    pub candidate_usages: Option<CandidateHeadUsages>,

    /// The nested goals of this goal, see the doc comment of the type.
    pub nested_goals: NestedGoals<X>,
}

/// The stack of goals currently being computed.
///
/// An element is *deeper* in the stack if its index is *lower*.
///
/// Only the last entry of the stack is mutable. All other entries get
/// lazily updated in `update_parent_goal`.
#[derive_where(Default; X: Cx)]
pub(super) struct Stack<X: Cx> {
    entries: IndexVec<StackDepth, StackEntry<X>>,
}

impl<X: Cx> Stack<X> {
    pub(super) fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub(super) fn len(&self) -> usize {
        self.entries.len()
    }

    pub(super) fn last(&self) -> Option<&StackEntry<X>> {
        self.entries.raw.last()
    }

    pub(super) fn last_mut(&mut self) -> Option<&mut StackEntry<X>> {
        self.entries.raw.last_mut()
    }

    pub(super) fn last_mut_with_index(&mut self) -> Option<(StackDepth, &mut StackEntry<X>)> {
        self.entries.last_index().map(|idx| (idx, &mut self.entries[idx]))
    }

    pub(super) fn next_index(&self) -> StackDepth {
        self.entries.next_index()
    }

    pub(super) fn push(&mut self, entry: StackEntry<X>) -> StackDepth {
        if cfg!(debug_assertions) && self.entries.iter().any(|e| e.input == entry.input) {
            panic!("pushing duplicate entry on stack: {entry:?} {:?}", self.entries);
        }
        self.entries.push(entry)
    }

    pub(super) fn pop(&mut self) -> StackEntry<X> {
        self.entries.pop().unwrap()
    }

    pub(super) fn cycle_step_kinds(&self, head: StackDepth) -> impl Iterator<Item = PathKind> {
        self.entries.raw[head.index() + 1..].iter().map(|entry| entry.step_kind_from_parent)
    }

    pub(super) fn iter(&self) -> impl Iterator<Item = &StackEntry<X>> {
        self.entries.iter()
    }

    pub(super) fn find(&self, input: X::Input) -> Option<StackDepth> {
        self.entries.iter_enumerated().find(|(_, e)| e.input == input).map(|(idx, _)| idx)
    }
}

impl<X: Cx> Index<StackDepth> for Stack<X> {
    type Output = StackEntry<X>;
    fn index(&self, index: StackDepth) -> &StackEntry<X> {
        &self.entries[index]
    }
}
