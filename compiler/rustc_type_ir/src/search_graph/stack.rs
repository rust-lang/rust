use std::ops::{Index, IndexMut};

use derive_where::derive_where;
use rustc_index::IndexVec;

use super::{AvailableDepth, Cx, CycleHeads, NestedGoals, PathKind, UsageKind};

rustc_index::newtype_index! {
    #[orderable]
    #[gate_rustc_only]
    pub(super) struct StackDepth {}
}

/// Stack entries of the evaluation stack. Its fields tend to be lazily
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

    /// Starts out as `None` and gets set when rerunning this
    /// goal in case we encounter a cycle.
    pub provisional_result: Option<X::Result>,

    /// The maximum depth required while evaluating this goal.
    pub required_depth: usize,

    /// All cycle heads this goal depends on. Lazily updated and only
    /// up-to date for the top of the stack.
    pub heads: CycleHeads,

    /// Whether evaluating this goal encountered overflow. Lazily updated.
    pub encountered_overflow: bool,

    /// Whether this goal has been used as the root of a cycle. This gets
    /// eagerly updated when encountering a cycle.
    pub has_been_used: Option<UsageKind>,

    /// The nested goals of this goal, see the doc comment of the type.
    pub nested_goals: NestedGoals<X>,
}

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

    pub(super) fn last_index(&self) -> Option<StackDepth> {
        self.entries.last_index()
    }

    pub(super) fn last(&self) -> Option<&StackEntry<X>> {
        self.entries.raw.last()
    }

    pub(super) fn last_mut(&mut self) -> Option<&mut StackEntry<X>> {
        self.entries.raw.last_mut()
    }

    pub(super) fn next_index(&self) -> StackDepth {
        self.entries.next_index()
    }

    pub(super) fn push(&mut self, entry: StackEntry<X>) -> StackDepth {
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

impl<X: Cx> IndexMut<StackDepth> for Stack<X> {
    fn index_mut(&mut self, index: StackDepth) -> &mut Self::Output {
        &mut self.entries[index]
    }
}
