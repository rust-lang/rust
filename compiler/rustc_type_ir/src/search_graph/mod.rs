//! The search graph is responsible for caching and cycle detection in the trait
//! solver. Making sure that caching doesn't result in soundness bugs or unstable
//! query results is very challenging and makes this one of the most-involved
//! self-contained components of the compiler.
//!
//! We added fuzzing support to test its correctness. The fuzzers used to verify
//! the current implementation can be found in <https://github.com/lcnr/search_graph_fuzz>.
//!
//! This is just a quick overview of the general design, please check out the relevant
//! [rustc-dev-guide chapter](https://rustc-dev-guide.rust-lang.org/solve/caching.html) for
//! more details. Caching is split between a global cache and the per-cycle `provisional_cache`.
//! The global cache has to be completely unobservable, while the per-cycle cache may impact
//! behavior as long as the resulting behavior is still correct.
use std::cmp::Ordering;
use std::collections::hash_map::Entry;
use std::collections::{BTreeMap, btree_map};
use std::fmt::Debug;
use std::hash::Hash;
use std::iter;
use std::marker::PhantomData;

use derive_where::derive_where;
#[cfg(feature = "nightly")]
use rustc_macros::{Decodable_NoContext, Encodable_NoContext, HashStable_NoContext};
use rustc_type_ir::data_structures::HashMap;
use tracing::{debug, instrument};

mod stack;
use stack::{Stack, StackDepth, StackEntry};
mod global_cache;
use global_cache::CacheData;
pub use global_cache::GlobalCache;

/// The search graph does not simply use `Interner` directly
/// to enable its fuzzing without having to stub the rest of
/// the interner. We don't make this a super trait of `Interner`
/// as users of the shared type library shouldn't have to care
/// about `Input` and `Result` as they are implementation details
/// of the search graph.
pub trait Cx: Copy {
    type Input: Debug + Eq + Hash + Copy;
    type Result: Debug + Eq + Hash + Copy;

    type DepNodeIndex;
    type Tracked<T: Debug + Clone>: Debug;
    fn mk_tracked<T: Debug + Clone>(
        self,
        data: T,
        dep_node_index: Self::DepNodeIndex,
    ) -> Self::Tracked<T>;
    fn get_tracked<T: Debug + Clone>(self, tracked: &Self::Tracked<T>) -> T;
    fn with_cached_task<T>(self, task: impl FnOnce() -> T) -> (T, Self::DepNodeIndex);

    fn with_global_cache<R>(self, f: impl FnOnce(&mut GlobalCache<Self>) -> R) -> R;

    fn assert_evaluation_is_concurrent(&self);
}

pub trait Delegate: Sized {
    type Cx: Cx;
    /// Whether to use the provisional cache. Set to `false` by a fuzzer when
    /// validating the search graph.
    const ENABLE_PROVISIONAL_CACHE: bool;
    type ValidationScope;
    /// Returning `Some` disables the global cache for the current goal.
    ///
    /// The `ValidationScope` is used when fuzzing the search graph to track
    /// for which goals the global cache has been disabled. This is necessary
    /// as we may otherwise ignore the global cache entry for some goal `G`
    /// only to later use it, failing to detect a cycle goal and potentially
    /// changing the result.
    fn enter_validation_scope(
        cx: Self::Cx,
        input: <Self::Cx as Cx>::Input,
    ) -> Option<Self::ValidationScope>;

    const FIXPOINT_STEP_LIMIT: usize;

    type ProofTreeBuilder;
    fn inspect_is_noop(inspect: &mut Self::ProofTreeBuilder) -> bool;

    const DIVIDE_AVAILABLE_DEPTH_ON_OVERFLOW: usize;

    fn initial_provisional_result(
        cx: Self::Cx,
        kind: PathKind,
        input: <Self::Cx as Cx>::Input,
    ) -> <Self::Cx as Cx>::Result;
    fn is_initial_provisional_result(result: <Self::Cx as Cx>::Result) -> Option<PathKind>;
    fn stack_overflow_result(
        cx: Self::Cx,
        input: <Self::Cx as Cx>::Input,
    ) -> <Self::Cx as Cx>::Result;
    fn fixpoint_overflow_result(
        cx: Self::Cx,
        input: <Self::Cx as Cx>::Input,
    ) -> <Self::Cx as Cx>::Result;

    fn is_ambiguous_result(result: <Self::Cx as Cx>::Result) -> bool;
    fn propagate_ambiguity(
        cx: Self::Cx,
        for_input: <Self::Cx as Cx>::Input,
        from_result: <Self::Cx as Cx>::Result,
    ) -> <Self::Cx as Cx>::Result;

    fn compute_goal(
        search_graph: &mut SearchGraph<Self>,
        cx: Self::Cx,
        input: <Self::Cx as Cx>::Input,
        inspect: &mut Self::ProofTreeBuilder,
    ) -> <Self::Cx as Cx>::Result;
}

/// In the initial iteration of a cycle, we do not yet have a provisional
/// result. In the case we return an initial provisional result depending
/// on the kind of cycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(
    feature = "nightly",
    derive(Decodable_NoContext, Encodable_NoContext, HashStable_NoContext)
)]
pub enum PathKind {
    /// A path consisting of only inductive/unproductive steps. Their initial
    /// provisional result is `Err(NoSolution)`. We currently treat them as
    /// `PathKind::Unknown` during coherence until we're fully confident in
    /// our approach.
    Inductive,
    /// A path which is not be coinductive right now but we may want
    /// to change of them to be so in the future. We return an ambiguous
    /// result in this case to prevent people from relying on this.
    Unknown,
    /// A path with at least one coinductive step. Such cycles hold.
    Coinductive,
    /// A path which is treated as ambiguous. Once a path has this path kind
    /// any other segment does not change its kind.
    ///
    /// This is currently only used when fuzzing to support negative reasoning.
    /// For more details, see #143054.
    ForcedAmbiguity,
}

impl PathKind {
    /// Returns the path kind when merging `self` with `rest`.
    ///
    /// Given an inductive path `self` and a coinductive path `rest`,
    /// the path `self -> rest` would be coinductive.
    ///
    /// This operation represents an ordering and would be equivalent
    /// to `max(self, rest)`.
    fn extend(self, rest: PathKind) -> PathKind {
        match (self, rest) {
            (PathKind::ForcedAmbiguity, _) | (_, PathKind::ForcedAmbiguity) => {
                PathKind::ForcedAmbiguity
            }
            (PathKind::Coinductive, _) | (_, PathKind::Coinductive) => PathKind::Coinductive,
            (PathKind::Unknown, _) | (_, PathKind::Unknown) => PathKind::Unknown,
            (PathKind::Inductive, PathKind::Inductive) => PathKind::Inductive,
        }
    }
}

/// The kinds of cycles a cycle head was involved in.
///
/// This is used to avoid rerunning a cycle if there's
/// just a single usage kind and the final result matches
/// its provisional result.
///
/// While it tracks the amount of usages using `u32`, we only ever
/// care whether there are any. We only count them to be able to ignore
/// usages from irrelevant candidates while evaluating a goal.
///
/// This cares about how nested goals relied on a cycle head. It does
/// not care about how frequently the nested goal relied on it.
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
struct HeadUsages {
    inductive: u32,
    unknown: u32,
    coinductive: u32,
    forced_ambiguity: u32,
}

impl HeadUsages {
    fn add_usage(&mut self, path: PathKind) {
        match path {
            PathKind::Inductive => self.inductive += 1,
            PathKind::Unknown => self.unknown += 1,
            PathKind::Coinductive => self.coinductive += 1,
            PathKind::ForcedAmbiguity => self.forced_ambiguity += 1,
        }
    }

    /// This adds the usages which occurred while computing a nested goal.
    ///
    /// We don't actually care about how frequently the nested goal relied
    /// on its cycle heads, only whether it did.
    fn add_usages_from_nested(&mut self, usages: HeadUsages) {
        let HeadUsages { inductive, unknown, coinductive, forced_ambiguity } = usages;
        self.inductive += if inductive == 0 { 0 } else { 1 };
        self.unknown += if unknown == 0 { 0 } else { 1 };
        self.coinductive += if coinductive == 0 { 0 } else { 1 };
        self.forced_ambiguity += if forced_ambiguity == 0 { 0 } else { 1 };
    }

    fn ignore_usages(&mut self, usages: HeadUsages) {
        let HeadUsages { inductive, unknown, coinductive, forced_ambiguity } = usages;
        self.inductive = self.inductive.checked_sub(inductive).unwrap();
        self.unknown = self.unknown.checked_sub(unknown).unwrap();
        self.coinductive = self.coinductive.checked_sub(coinductive).unwrap();
        self.forced_ambiguity = self.forced_ambiguity.checked_sub(forced_ambiguity).unwrap();
    }

    fn is_empty(self) -> bool {
        let HeadUsages { inductive, unknown, coinductive, forced_ambiguity } = self;
        inductive == 0 && unknown == 0 && coinductive == 0 && forced_ambiguity == 0
    }

    fn is_single(self, path_kind: PathKind) -> bool {
        match path_kind {
            PathKind::Inductive => matches!(
                self,
                HeadUsages { inductive: _, unknown: 0, coinductive: 0, forced_ambiguity: 0 },
            ),
            PathKind::Unknown => matches!(
                self,
                HeadUsages { inductive: 0, unknown: _, coinductive: 0, forced_ambiguity: 0 },
            ),
            PathKind::Coinductive => matches!(
                self,
                HeadUsages { inductive: 0, unknown: 0, coinductive: _, forced_ambiguity: 0 },
            ),
            PathKind::ForcedAmbiguity => matches!(
                self,
                HeadUsages { inductive: 0, unknown: 0, coinductive: 0, forced_ambiguity: _ },
            ),
        }
    }
}

#[derive(Debug, Default)]
pub struct CandidateHeadUsages {
    usages: Option<Box<HashMap<StackDepth, HeadUsages>>>,
}
impl CandidateHeadUsages {
    pub fn merge_usages(&mut self, other: CandidateHeadUsages) {
        if let Some(other_usages) = other.usages {
            if let Some(ref mut self_usages) = self.usages {
                #[allow(rustc::potential_query_instability)]
                for (head_index, head) in other_usages.into_iter() {
                    let HeadUsages { inductive, unknown, coinductive, forced_ambiguity } = head;
                    let self_usages = self_usages.entry(head_index).or_default();
                    self_usages.inductive += inductive;
                    self_usages.unknown += unknown;
                    self_usages.coinductive += coinductive;
                    self_usages.forced_ambiguity += forced_ambiguity;
                }
            } else {
                self.usages = Some(other_usages);
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct AvailableDepth(usize);
impl AvailableDepth {
    /// Returns the remaining depth allowed for nested goals.
    ///
    /// This is generally simply one less than the current depth.
    /// However, if we encountered overflow, we significantly reduce
    /// the remaining depth of all nested goals to prevent hangs
    /// in case there is exponential blowup.
    fn allowed_depth_for_nested<D: Delegate>(
        root_depth: AvailableDepth,
        stack: &Stack<D::Cx>,
    ) -> Option<AvailableDepth> {
        if let Some(last) = stack.last() {
            if last.available_depth.0 == 0 {
                return None;
            }

            Some(if last.encountered_overflow {
                AvailableDepth(last.available_depth.0 / D::DIVIDE_AVAILABLE_DEPTH_ON_OVERFLOW)
            } else {
                AvailableDepth(last.available_depth.0 - 1)
            })
        } else {
            Some(root_depth)
        }
    }

    /// Whether we're allowed to use a global cache entry which required
    /// the given depth.
    fn cache_entry_is_applicable(self, additional_depth: usize) -> bool {
        self.0 >= additional_depth
    }
}

#[derive(Clone, Copy, Debug)]
struct CycleHead {
    paths_to_head: PathsToNested,
    /// If the `usages` are empty, the result of that head does not matter
    /// for the current goal. However, we still don't completely drop this
    /// cycle head as whether or not it exists impacts which queries we
    /// access, so ignoring it would cause incremental compilation verification
    /// failures or hide query cycles.
    usages: HeadUsages,
}

/// All cycle heads a given goal depends on, ordered by their stack depth.
///
/// We also track all paths from this goal to that head. This is necessary
/// when rebasing provisional cache results.
#[derive(Clone, Debug, Default)]
struct CycleHeads {
    heads: BTreeMap<StackDepth, CycleHead>,
}

impl CycleHeads {
    fn is_empty(&self) -> bool {
        self.heads.is_empty()
    }

    fn highest_cycle_head(&self) -> (StackDepth, CycleHead) {
        self.heads.last_key_value().map(|(k, v)| (*k, *v)).unwrap()
    }

    fn highest_cycle_head_index(&self) -> StackDepth {
        self.opt_highest_cycle_head_index().unwrap()
    }

    fn opt_highest_cycle_head_index(&self) -> Option<StackDepth> {
        self.heads.last_key_value().map(|(k, _)| *k)
    }

    fn opt_lowest_cycle_head_index(&self) -> Option<StackDepth> {
        self.heads.first_key_value().map(|(k, _)| *k)
    }

    fn remove_highest_cycle_head(&mut self) -> CycleHead {
        let last = self.heads.pop_last();
        last.unwrap().1
    }

    fn insert(
        &mut self,
        head_index: StackDepth,
        path_from_entry: impl Into<PathsToNested> + Copy,
        usages: HeadUsages,
    ) {
        match self.heads.entry(head_index) {
            btree_map::Entry::Vacant(entry) => {
                entry.insert(CycleHead { paths_to_head: path_from_entry.into(), usages });
            }
            btree_map::Entry::Occupied(entry) => {
                let head = entry.into_mut();
                head.paths_to_head |= path_from_entry.into();
                head.usages.add_usages_from_nested(usages);
            }
        }
    }

    fn ignore_usages(&mut self, head_index: StackDepth, usages: HeadUsages) {
        self.heads.get_mut(&head_index).unwrap().usages.ignore_usages(usages)
    }

    fn iter(&self) -> impl Iterator<Item = (StackDepth, CycleHead)> + '_ {
        self.heads.iter().map(|(k, v)| (*k, *v))
    }
}

bitflags::bitflags! {
    /// Tracks how nested goals have been accessed. This is necessary to disable
    /// global cache entries if computing them would otherwise result in a cycle or
    /// access a provisional cache entry.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct PathsToNested: u8 {
        /// The initial value when adding a goal to its own nested goals.
        const EMPTY                      = 1 << 0;
        const INDUCTIVE                  = 1 << 1;
        const UNKNOWN                    = 1 << 2;
        const COINDUCTIVE                = 1 << 3;
        const FORCED_AMBIGUITY           = 1 << 4;
    }
}
impl From<PathKind> for PathsToNested {
    fn from(path: PathKind) -> PathsToNested {
        match path {
            PathKind::Inductive => PathsToNested::INDUCTIVE,
            PathKind::Unknown => PathsToNested::UNKNOWN,
            PathKind::Coinductive => PathsToNested::COINDUCTIVE,
            PathKind::ForcedAmbiguity => PathsToNested::FORCED_AMBIGUITY,
        }
    }
}
impl PathsToNested {
    /// The implementation of this function is kind of ugly. We check whether
    /// there currently exist 'weaker' paths in the set, if so we upgrade these
    /// paths to at least `path`.
    #[must_use]
    fn extend_with(mut self, path: PathKind) -> Self {
        match path {
            PathKind::Inductive => {
                if self.intersects(PathsToNested::EMPTY) {
                    self.remove(PathsToNested::EMPTY);
                    self.insert(PathsToNested::INDUCTIVE);
                }
            }
            PathKind::Unknown => {
                if self.intersects(PathsToNested::EMPTY | PathsToNested::INDUCTIVE) {
                    self.remove(PathsToNested::EMPTY | PathsToNested::INDUCTIVE);
                    self.insert(PathsToNested::UNKNOWN);
                }
            }
            PathKind::Coinductive => {
                if self.intersects(
                    PathsToNested::EMPTY | PathsToNested::INDUCTIVE | PathsToNested::UNKNOWN,
                ) {
                    self.remove(
                        PathsToNested::EMPTY | PathsToNested::INDUCTIVE | PathsToNested::UNKNOWN,
                    );
                    self.insert(PathsToNested::COINDUCTIVE);
                }
            }
            PathKind::ForcedAmbiguity => {
                if self.intersects(
                    PathsToNested::EMPTY
                        | PathsToNested::INDUCTIVE
                        | PathsToNested::UNKNOWN
                        | PathsToNested::COINDUCTIVE,
                ) {
                    self.remove(
                        PathsToNested::EMPTY
                            | PathsToNested::INDUCTIVE
                            | PathsToNested::UNKNOWN
                            | PathsToNested::COINDUCTIVE,
                    );
                    self.insert(PathsToNested::FORCED_AMBIGUITY);
                }
            }
        }

        self
    }

    #[must_use]
    fn extend_with_paths(self, path: PathsToNested) -> Self {
        let mut new = PathsToNested::empty();
        for p in path.iter_paths() {
            new |= self.extend_with(p);
        }
        new
    }

    fn iter_paths(self) -> impl Iterator<Item = PathKind> {
        let (PathKind::Inductive
        | PathKind::Unknown
        | PathKind::Coinductive
        | PathKind::ForcedAmbiguity);
        [PathKind::Inductive, PathKind::Unknown, PathKind::Coinductive, PathKind::ForcedAmbiguity]
            .into_iter()
            .filter(move |&p| self.contains(p.into()))
    }
}

/// The nested goals of each stack entry and the path from the
/// stack entry to that nested goal.
///
/// They are used when checking whether reevaluating a global cache
/// would encounter a cycle or use a provisional cache entry given the
/// current search graph state. We need to disable the global cache
/// in this case as it could otherwise result in behavioral differences.
/// Cycles can impact behavior. The cycle ABA may have different final
/// results from a the cycle BAB depending on the cycle root.
///
/// We only start tracking nested goals once we've either encountered
/// overflow or a solver cycle. This is a performance optimization to
/// avoid tracking nested goals on the happy path.
#[derive_where(Debug, Default, Clone; X: Cx)]
struct NestedGoals<X: Cx> {
    nested_goals: HashMap<X::Input, PathsToNested>,
}
impl<X: Cx> NestedGoals<X> {
    fn is_empty(&self) -> bool {
        self.nested_goals.is_empty()
    }

    fn insert(&mut self, input: X::Input, paths_to_nested: PathsToNested) {
        match self.nested_goals.entry(input) {
            Entry::Occupied(mut entry) => *entry.get_mut() |= paths_to_nested,
            Entry::Vacant(entry) => drop(entry.insert(paths_to_nested)),
        }
    }

    /// Adds the nested goals of a nested goal, given that the path `step_kind` from this goal
    /// to the parent goal.
    ///
    /// If the path from this goal to the nested goal is inductive, the paths from this goal
    /// to all nested goals of that nested goal are also inductive. Otherwise the paths are
    /// the same as for the child.
    fn extend_from_child(&mut self, step_kind: PathKind, nested_goals: &NestedGoals<X>) {
        #[allow(rustc::potential_query_instability)]
        for (input, paths_to_nested) in nested_goals.iter() {
            let paths_to_nested = paths_to_nested.extend_with(step_kind);
            self.insert(input, paths_to_nested);
        }
    }

    #[cfg_attr(feature = "nightly", rustc_lint_query_instability)]
    #[allow(rustc::potential_query_instability)]
    fn iter(&self) -> impl Iterator<Item = (X::Input, PathsToNested)> + '_ {
        self.nested_goals.iter().map(|(i, p)| (*i, *p))
    }

    fn contains(&self, input: X::Input) -> bool {
        self.nested_goals.contains_key(&input)
    }
}

/// A provisional result of an already computed goals which depends on other
/// goals still on the stack.
#[derive_where(Debug; X: Cx)]
struct ProvisionalCacheEntry<X: Cx> {
    /// Whether evaluating the goal encountered overflow. This is used to
    /// disable the cache entry except if the last goal on the stack is
    /// already involved in this cycle.
    encountered_overflow: bool,
    /// All cycle heads this cache entry depends on.
    heads: CycleHeads,
    /// The path from the highest cycle head to this goal. This differs from
    /// `heads` which tracks the path to the cycle head *from* this goal.
    path_from_head: PathKind,
    result: X::Result,
}

/// The final result of evaluating a goal.
///
/// We reset `encountered_overflow` when reevaluating a goal,
/// but need to track whether we've hit the recursion limit at
/// all for correctness.
///
/// We've previously simply returned the final `StackEntry` but this
/// made it easy to accidentally drop information from the previous
/// evaluation.
#[derive_where(Debug; X: Cx)]
struct EvaluationResult<X: Cx> {
    encountered_overflow: bool,
    required_depth: usize,
    heads: CycleHeads,
    nested_goals: NestedGoals<X>,
    result: X::Result,
}

impl<X: Cx> EvaluationResult<X> {
    fn finalize(
        final_entry: StackEntry<X>,
        encountered_overflow: bool,
        result: X::Result,
    ) -> EvaluationResult<X> {
        EvaluationResult {
            encountered_overflow,
            // Unlike `encountered_overflow`, we share `heads`, `required_depth`,
            // and `nested_goals` between evaluations.
            required_depth: final_entry.required_depth,
            heads: final_entry.heads,
            nested_goals: final_entry.nested_goals,
            // We only care about the final result.
            result,
        }
    }
}

pub struct SearchGraph<D: Delegate<Cx = X>, X: Cx = <D as Delegate>::Cx> {
    root_depth: AvailableDepth,
    stack: Stack<X>,
    /// The provisional cache contains entries for already computed goals which
    /// still depend on goals higher-up in the stack. We don't move them to the
    /// global cache and track them locally instead. A provisional cache entry
    /// is only valid until the result of one of its cycle heads changes.
    provisional_cache: HashMap<X::Input, Vec<ProvisionalCacheEntry<X>>>,

    _marker: PhantomData<D>,
}

/// While [`SearchGraph::update_parent_goal`] can be mostly shared between
/// ordinary nested goals/global cache hits and provisional cache hits,
/// using the provisional cache should not add any nested goals.
///
/// `nested_goals` are only used when checking whether global cache entries
/// are applicable. This only cares about whether a goal is actually accessed.
/// Given that the usage of the provisional cache is fully deterministic, we
/// don't need to track the nested goals used while computing a provisional
/// cache entry.
enum UpdateParentGoalCtxt<'a, X: Cx> {
    Ordinary(&'a NestedGoals<X>),
    CycleOnStack(X::Input),
    ProvisionalCacheHit,
}

impl<D: Delegate<Cx = X>, X: Cx> SearchGraph<D> {
    pub fn new(root_depth: usize) -> SearchGraph<D> {
        Self {
            root_depth: AvailableDepth(root_depth),
            stack: Default::default(),
            provisional_cache: Default::default(),
            _marker: PhantomData,
        }
    }

    /// Lazily update the stack entry for the parent goal.
    /// This behavior is shared between actually evaluating goals
    /// and using existing global cache entries to make sure they
    /// have the same impact on the remaining evaluation.
    fn update_parent_goal(
        stack: &mut Stack<X>,
        step_kind_from_parent: PathKind,
        required_depth_for_nested: usize,
        heads: impl Iterator<Item = (StackDepth, CycleHead)>,
        encountered_overflow: bool,
        context: UpdateParentGoalCtxt<'_, X>,
    ) {
        if let Some((parent_index, parent)) = stack.last_mut_with_index() {
            parent.required_depth = parent.required_depth.max(required_depth_for_nested + 1);
            parent.encountered_overflow |= encountered_overflow;

            for (head_index, head) in heads {
                if let Some(candidate_usages) = &mut parent.candidate_usages {
                    candidate_usages
                        .usages
                        .get_or_insert_default()
                        .entry(head_index)
                        .or_default()
                        .add_usages_from_nested(head.usages);
                }
                match head_index.cmp(&parent_index) {
                    Ordering::Less => parent.heads.insert(
                        head_index,
                        head.paths_to_head.extend_with(step_kind_from_parent),
                        head.usages,
                    ),
                    Ordering::Equal => {
                        parent.usages.get_or_insert_default().add_usages_from_nested(head.usages);
                    }
                    Ordering::Greater => unreachable!(),
                }
            }
            let parent_depends_on_cycle = match context {
                UpdateParentGoalCtxt::Ordinary(nested_goals) => {
                    parent.nested_goals.extend_from_child(step_kind_from_parent, nested_goals);
                    !nested_goals.is_empty()
                }
                UpdateParentGoalCtxt::CycleOnStack(head) => {
                    // We lookup provisional cache entries before detecting cycles.
                    // We therefore can't use a global cache entry if it contains a cycle
                    // whose head is in the provisional cache.
                    parent.nested_goals.insert(head, step_kind_from_parent.into());
                    true
                }
                UpdateParentGoalCtxt::ProvisionalCacheHit => true,
            };
            // Once we've got goals which encountered overflow or a cycle,
            // we track all goals whose behavior may depend depend on these
            // goals as this change may cause them to now depend on additional
            // goals, resulting in new cycles. See the dev-guide for examples.
            if parent_depends_on_cycle {
                parent.nested_goals.insert(parent.input, PathsToNested::EMPTY);
            }
        }
    }

    pub fn is_empty(&self) -> bool {
        if self.stack.is_empty() {
            debug_assert!(self.provisional_cache.is_empty());
            true
        } else {
            false
        }
    }

    /// The number of goals currently in the search graph. This should only be
    /// used for debugging purposes.
    pub fn debug_current_depth(&self) -> usize {
        self.stack.len()
    }

    /// Whether the path from `head` to the current stack entry is inductive or coinductive.
    ///
    /// The `step_kind_to_head` is used to add a single additional path segment to the path on
    /// the stack which completes the cycle. This given an inductive step AB which then cycles
    /// coinductively with A, we need to treat this cycle as coinductive.
    fn cycle_path_kind(
        stack: &Stack<X>,
        step_kind_to_head: PathKind,
        head: StackDepth,
    ) -> PathKind {
        stack.cycle_step_kinds(head).fold(step_kind_to_head, |curr, step| curr.extend(step))
    }

    pub fn enter_single_candidate(&mut self) {
        let prev = self.stack.last_mut().unwrap().candidate_usages.replace(Default::default());
        debug_assert!(prev.is_none(), "existing candidate_usages: {prev:?}");
    }

    pub fn finish_single_candidate(&mut self) -> CandidateHeadUsages {
        self.stack.last_mut().unwrap().candidate_usages.take().unwrap()
    }

    pub fn ignore_candidate_head_usages(&mut self, usages: CandidateHeadUsages) {
        if let Some(usages) = usages.usages {
            let (entry_index, entry) = self.stack.last_mut_with_index().unwrap();
            #[allow(rustc::potential_query_instability)]
            for (head_index, usages) in usages.into_iter() {
                if head_index == entry_index {
                    entry.usages.unwrap().ignore_usages(usages);
                } else {
                    entry.heads.ignore_usages(head_index, usages);
                }
            }
        }
    }

    pub fn evaluate_root_goal_for_proof_tree(
        cx: X,
        root_depth: usize,
        input: X::Input,
        inspect: &mut D::ProofTreeBuilder,
    ) -> X::Result {
        let mut this = SearchGraph::<D>::new(root_depth);
        let available_depth = AvailableDepth(root_depth);
        let step_kind_from_parent = PathKind::Inductive; // is never used
        this.stack.push(StackEntry {
            input,
            step_kind_from_parent,
            available_depth,
            provisional_result: None,
            required_depth: 0,
            heads: Default::default(),
            encountered_overflow: false,
            usages: None,
            candidate_usages: None,
            nested_goals: Default::default(),
        });
        let evaluation_result = this.evaluate_goal_in_task(cx, input, inspect);
        evaluation_result.result
    }

    /// Probably the most involved method of the whole solver.
    ///
    /// While goals get computed via `D::compute_goal`, this function handles
    /// caching, overflow, and cycles.
    #[instrument(level = "debug", skip(self, cx, inspect), ret)]
    pub fn evaluate_goal(
        &mut self,
        cx: X,
        input: X::Input,
        step_kind_from_parent: PathKind,
        inspect: &mut D::ProofTreeBuilder,
    ) -> X::Result {
        let Some(available_depth) =
            AvailableDepth::allowed_depth_for_nested::<D>(self.root_depth, &self.stack)
        else {
            return self.handle_overflow(cx, input);
        };

        // We check the provisional cache before checking the global cache. This simplifies
        // the implementation as we can avoid worrying about cases where both the global and
        // provisional cache may apply, e.g. consider the following example
        //
        // - xxBA overflow
        // - A
        //     - BA cycle
        //     - CB :x:
        if let Some(result) = self.lookup_provisional_cache(input, step_kind_from_parent) {
            return result;
        }

        // Lookup the global cache unless we're building proof trees or are currently
        // fuzzing.
        let validate_cache = if !D::inspect_is_noop(inspect) {
            None
        } else if let Some(scope) = D::enter_validation_scope(cx, input) {
            // When validating the global cache we need to track the goals for which the
            // global cache has been disabled as it may otherwise change the result for
            // cyclic goals. We don't care about goals which are not on the current stack
            // so it's fine to drop their scope eagerly.
            self.lookup_global_cache_untracked(cx, input, step_kind_from_parent, available_depth)
                .inspect(|expected| debug!(?expected, "validate cache entry"))
                .map(|r| (scope, r))
        } else if let Some(result) =
            self.lookup_global_cache(cx, input, step_kind_from_parent, available_depth)
        {
            return result;
        } else {
            None
        };

        // Detect cycles on the stack. We do this after the global cache lookup to
        // avoid iterating over the stack in case a goal has already been computed.
        // This may not have an actual performance impact and we could reorder them
        // as it may reduce the number of `nested_goals` we need to track.
        if let Some(result) = self.check_cycle_on_stack(cx, input, step_kind_from_parent) {
            debug_assert!(validate_cache.is_none(), "global cache and cycle on stack: {input:?}");
            return result;
        }

        // Unfortunate, it looks like we actually have to compute this goal.
        self.stack.push(StackEntry {
            input,
            step_kind_from_parent,
            available_depth,
            provisional_result: None,
            required_depth: 0,
            heads: Default::default(),
            encountered_overflow: false,
            usages: None,
            candidate_usages: None,
            nested_goals: Default::default(),
        });

        // This is for global caching, so we properly track query dependencies.
        // Everything that affects the `result` should be performed within this
        // `with_cached_task` closure. If computing this goal depends on something
        // not tracked by the cache key and from outside of this anon task, it
        // must not be added to the global cache. Notably, this is the case for
        // trait solver cycles participants.
        let (evaluation_result, dep_node) =
            cx.with_cached_task(|| self.evaluate_goal_in_task(cx, input, inspect));

        // We've finished computing the goal and have popped it from the stack,
        // lazily update its parent goal.
        Self::update_parent_goal(
            &mut self.stack,
            step_kind_from_parent,
            evaluation_result.required_depth,
            evaluation_result.heads.iter(),
            evaluation_result.encountered_overflow,
            UpdateParentGoalCtxt::Ordinary(&evaluation_result.nested_goals),
        );
        let result = evaluation_result.result;

        // We're now done with this goal. We only add the root of cycles to the global cache.
        // In case this goal is involved in a larger cycle add it to the provisional cache.
        if evaluation_result.heads.is_empty() {
            if let Some((_scope, expected)) = validate_cache {
                // Do not try to move a goal into the cache again if we're testing
                // the global cache.
                assert_eq!(expected, evaluation_result.result, "input={input:?}");
            } else if D::inspect_is_noop(inspect) {
                self.insert_global_cache(cx, input, evaluation_result, dep_node)
            }
        } else if D::ENABLE_PROVISIONAL_CACHE {
            debug_assert!(validate_cache.is_none(), "unexpected non-root: {input:?}");
            let entry = self.provisional_cache.entry(input).or_default();
            let EvaluationResult {
                encountered_overflow,
                required_depth: _,
                heads,
                nested_goals: _,
                result,
            } = evaluation_result;
            let path_from_head = Self::cycle_path_kind(
                &self.stack,
                step_kind_from_parent,
                heads.highest_cycle_head_index(),
            );
            let provisional_cache_entry =
                ProvisionalCacheEntry { encountered_overflow, heads, path_from_head, result };
            debug!(?provisional_cache_entry);
            entry.push(provisional_cache_entry);
        } else {
            debug_assert!(validate_cache.is_none(), "unexpected non-root: {input:?}");
        }

        result
    }

    fn handle_overflow(&mut self, cx: X, input: X::Input) -> X::Result {
        if let Some(last) = self.stack.last_mut() {
            last.encountered_overflow = true;
            // If computing a goal `B` depends on another goal `A` and
            // `A` has a nested goal which overflows, then computing `B`
            // at the same depth, but with `A` already on the stack,
            // would encounter a solver cycle instead, potentially
            // changing the result.
            //
            // We must therefore not use the global cache entry for `B` in that case.
            // See tests/ui/traits/next-solver/cycles/hidden-by-overflow.rs
            last.nested_goals.insert(last.input, PathsToNested::EMPTY);
        }

        debug!("encountered stack overflow");
        D::stack_overflow_result(cx, input)
    }

    /// When reevaluating a goal with a changed provisional result, all provisional cache entry
    /// which depend on this goal get invalidated.
    ///
    /// Note that we keep provisional cache entries which accessed this goal as a cycle head, but
    /// don't depend on its value.
    fn clear_dependent_provisional_results_for_rerun(&mut self) {
        let rerun_index = self.stack.next_index();
        #[allow(rustc::potential_query_instability)]
        self.provisional_cache.retain(|_, entries| {
            entries.retain(|entry| {
                let (head_index, head) = entry.heads.highest_cycle_head();
                head_index != rerun_index || head.usages.is_empty()
            });
            !entries.is_empty()
        });
    }
}

/// We need to rebase provisional cache entries when popping one of their cycle
/// heads from the stack. This may not necessarily mean that we've actually
/// reached a fixpoint for that cycle head, which impacts the way we rebase
/// provisional cache entries.
enum RebaseReason {
    NoCycleUsages,
    Ambiguity,
    Overflow,
    /// We've actually reached a fixpoint.
    ///
    /// This either happens in the first evaluation step for the cycle head.
    /// In this case the used provisional result depends on the cycle `PathKind`.
    /// We store this path kind to check whether the the provisional cache entry
    /// we're rebasing relied on the same cycles.
    ///
    /// In later iterations cycles always return `stack_entry.provisional_result`
    /// so we no longer depend on the `PathKind`. We store `None` in that case.
    ReachedFixpoint(Option<PathKind>),
}

impl<D: Delegate<Cx = X>, X: Cx> SearchGraph<D, X> {
    /// A necessary optimization to handle complex solver cycles. A provisional cache entry
    /// relies on a set of cycle heads and the path towards these heads. When popping a cycle
    /// head from the stack after we've finished computing it, we can't be sure that the
    /// provisional cache entry is still applicable. We need to keep the cache entries to
    /// prevent hangs.
    ///
    /// This can be thought of as pretending to reevaluate the popped head as nested goals
    /// of this provisional result. For this to be correct, all cycles encountered while
    /// we'd reevaluate the cycle head as a nested goal must keep the same cycle kind.
    /// [rustc-dev-guide chapter](https://rustc-dev-guide.rust-lang.org/solve/caching.html).
    ///
    /// In case the popped cycle head failed to reach a fixpoint anything which depends on
    /// its provisional result is invalid. Actually discarding provisional cache entries in
    /// this case would cause hangs, so we instead change the result of dependant provisional
    /// cache entries to also be ambiguous. This causes some undesirable ambiguity for nested
    /// goals whose result doesn't actually depend on this cycle head, but that's acceptable
    /// to me.
    fn rebase_provisional_cache_entries(
        &mut self,
        cx: X,
        stack_entry: &StackEntry<X>,
        rebase_reason: RebaseReason,
    ) {
        let popped_head_index = self.stack.next_index();
        #[allow(rustc::potential_query_instability)]
        self.provisional_cache.retain(|&input, entries| {
            entries.retain_mut(|entry| {
                let ProvisionalCacheEntry {
                    encountered_overflow: _,
                    heads,
                    path_from_head,
                    result,
                } = entry;
                let popped_head = if heads.highest_cycle_head_index() == popped_head_index {
                    heads.remove_highest_cycle_head()
                } else {
                    return true;
                };

                let Some(new_highest_head_index) = heads.opt_highest_cycle_head_index() else {
                    return false;
                };

                // We're rebasing an entry `e` over a head `p`. This head
                // has a number of own heads `h` it depends on.
                //
                // This causes our provisional result to depend on the heads
                // of `p` to avoid moving any goal which uses this cache entry to
                // the global cache.
                if popped_head.usages.is_empty() {
                    // The result of `e` does not depend on the value of `p`. This we can
                    // keep using the result of this provisional cache entry even if evaluating
                    // `p` as a nested goal of `e` would have a different result.
                    for (head_index, _) in stack_entry.heads.iter() {
                        heads.insert(head_index, PathsToNested::EMPTY, HeadUsages::default());
                    }
                } else {
                    // The entry `e` actually depends on the value of `p`. We need
                    // to make sure that the value of `p` wouldn't change even if we
                    // were to reevaluate it as a nested goal of `e` instead. For this
                    // we check that the path kind of all paths `hph` remain the
                    // same after rebasing.
                    //
                    // After rebasing the cycles `hph` will go through `e`. We need to make
                    // sure that forall possible paths `hep`, `heph` is equal to `hph.`
                    let ep = popped_head.paths_to_head;
                    for (head_index, head) in stack_entry.heads.iter() {
                        let ph = head.paths_to_head;
                        let hp = Self::cycle_path_kind(
                            &self.stack,
                            stack_entry.step_kind_from_parent,
                            head_index,
                        );
                        // We first validate that all cycles while computing `p` would stay
                        // the same if we were to recompute it as a nested goal of `e`.
                        let he = hp.extend(*path_from_head);
                        for ph in ph.iter_paths() {
                            let hph = hp.extend(ph);
                            for ep in ep.iter_paths() {
                                let hep = ep.extend(he);
                                let heph = hep.extend(ph);
                                if hph != heph {
                                    return false;
                                }
                            }
                        }

                        // If so, all paths reached while computing `p` have to get added
                        // the heads of `e` to make sure that rebasing `e` again also considers
                        // them.
                        let eph = ep.extend_with_paths(ph);
                        heads.insert(head_index, eph, head.usages);
                    }

                    // The provisional cache entry does depend on the provisional result
                    // of the popped cycle head. We need to mutate the result of our
                    // provisional cache entry in case we did not reach a fixpoint.
                    match rebase_reason {
                        // If the cycle head does not actually depend on itself, then
                        // the provisional result used by the provisional cache entry
                        // is not actually equal to the final provisional result. We
                        // need to discard the provisional cache entry in this case.
                        RebaseReason::NoCycleUsages => return false,
                        RebaseReason::Ambiguity => {
                            *result = D::propagate_ambiguity(cx, input, *result);
                        }
                        RebaseReason::Overflow => *result = D::fixpoint_overflow_result(cx, input),
                        RebaseReason::ReachedFixpoint(None) => {}
                        RebaseReason::ReachedFixpoint(Some(path_kind)) => {
                            if !popped_head.usages.is_single(path_kind) {
                                return false;
                            }
                        }
                    };
                }

                // We now care about the path from the next highest cycle head to the
                // provisional cache entry.
                *path_from_head = path_from_head.extend(Self::cycle_path_kind(
                    &self.stack,
                    stack_entry.step_kind_from_parent,
                    new_highest_head_index,
                ));

                true
            });
            !entries.is_empty()
        });
    }

    fn lookup_provisional_cache(
        &mut self,
        input: X::Input,
        step_kind_from_parent: PathKind,
    ) -> Option<X::Result> {
        if !D::ENABLE_PROVISIONAL_CACHE {
            return None;
        }

        let entries = self.provisional_cache.get(&input)?;
        for &ProvisionalCacheEntry { encountered_overflow, ref heads, path_from_head, result } in
            entries
        {
            let head_index = heads.highest_cycle_head_index();
            if encountered_overflow {
                // This check is overly strict and very subtle. We need to make sure that if
                // a global cache entry depends on some goal without adding it to its
                // `nested_goals`, that goal must never have an applicable provisional
                // cache entry to avoid incorrectly applying the cache entry.
                //
                // As we'd have to otherwise track literally all nested goals, we only
                // apply provisional cache entries which encountered overflow once the
                // current goal is already part of the same cycle. This check could be
                // improved but seems to be good enough for now.
                let last = self.stack.last().unwrap();
                if last.heads.opt_lowest_cycle_head_index().is_none_or(|lowest| lowest > head_index)
                {
                    continue;
                }
            }

            // A provisional cache entry is only valid if the current path from its
            // highest cycle head to the goal is the same.
            if path_from_head
                == Self::cycle_path_kind(&self.stack, step_kind_from_parent, head_index)
            {
                Self::update_parent_goal(
                    &mut self.stack,
                    step_kind_from_parent,
                    0,
                    heads.iter(),
                    encountered_overflow,
                    UpdateParentGoalCtxt::ProvisionalCacheHit,
                );
                debug!(?head_index, ?path_from_head, "provisional cache hit");
                return Some(result);
            }
        }

        None
    }

    /// Even if there is a global cache entry for a given goal, we need to make sure
    /// evaluating this entry would not have ended up depending on either a goal
    /// already on the stack or a provisional cache entry.
    fn candidate_is_applicable(
        &self,
        step_kind_from_parent: PathKind,
        nested_goals: &NestedGoals<X>,
    ) -> bool {
        // If the global cache entry didn't depend on any nested goals, it always
        // applies.
        if nested_goals.is_empty() {
            return true;
        }

        // If a nested goal of the global cache entry is on the stack, we would
        // definitely encounter a cycle.
        if self.stack.iter().any(|e| nested_goals.contains(e.input)) {
            debug!("cache entry not applicable due to stack");
            return false;
        }

        // The global cache entry is also invalid if there's a provisional cache entry
        // would apply for any of its nested goals.
        #[allow(rustc::potential_query_instability)]
        for (input, path_from_global_entry) in nested_goals.iter() {
            let Some(entries) = self.provisional_cache.get(&input) else {
                continue;
            };

            debug!(?input, ?path_from_global_entry, ?entries, "candidate_is_applicable");
            // A provisional cache entry is applicable if the path to
            // its highest cycle head is equal to the expected path.
            for &ProvisionalCacheEntry {
                encountered_overflow,
                ref heads,
                path_from_head: head_to_provisional,
                result: _,
            } in entries.iter()
            {
                // We don't have to worry about provisional cache entries which encountered
                // overflow, see the relevant comment in `lookup_provisional_cache`.
                if encountered_overflow {
                    continue;
                }

                // A provisional cache entry only applies if the path from its highest head
                // matches the path when encountering the goal.
                //
                // We check if any of the paths taken while computing the global goal
                // would end up with an applicable provisional cache entry.
                let head_index = heads.highest_cycle_head_index();
                let head_to_curr =
                    Self::cycle_path_kind(&self.stack, step_kind_from_parent, head_index);
                let full_paths = path_from_global_entry.extend_with(head_to_curr);
                if full_paths.contains(head_to_provisional.into()) {
                    debug!(
                        ?full_paths,
                        ?head_to_provisional,
                        "cache entry not applicable due to matching paths"
                    );
                    return false;
                }
            }
        }

        true
    }

    /// Used when fuzzing the global cache. Accesses the global cache without
    /// updating the state of the search graph.
    fn lookup_global_cache_untracked(
        &self,
        cx: X,
        input: X::Input,
        step_kind_from_parent: PathKind,
        available_depth: AvailableDepth,
    ) -> Option<X::Result> {
        cx.with_global_cache(|cache| {
            cache
                .get(cx, input, available_depth, |nested_goals| {
                    self.candidate_is_applicable(step_kind_from_parent, nested_goals)
                })
                .map(|c| c.result)
        })
    }

    /// Try to fetch a previously computed result from the global cache,
    /// making sure to only do so if it would match the result of reevaluating
    /// this goal.
    fn lookup_global_cache(
        &mut self,
        cx: X,
        input: X::Input,
        step_kind_from_parent: PathKind,
        available_depth: AvailableDepth,
    ) -> Option<X::Result> {
        cx.with_global_cache(|cache| {
            let CacheData { result, required_depth, encountered_overflow, nested_goals } = cache
                .get(cx, input, available_depth, |nested_goals| {
                    self.candidate_is_applicable(step_kind_from_parent, nested_goals)
                })?;

            // We don't move cycle participants to the global cache, so the
            // cycle heads are always empty.
            let heads = iter::empty();
            Self::update_parent_goal(
                &mut self.stack,
                step_kind_from_parent,
                required_depth,
                heads,
                encountered_overflow,
                UpdateParentGoalCtxt::Ordinary(nested_goals),
            );

            debug!(?required_depth, "global cache hit");
            Some(result)
        })
    }

    fn check_cycle_on_stack(
        &mut self,
        cx: X,
        input: X::Input,
        step_kind_from_parent: PathKind,
    ) -> Option<X::Result> {
        let head_index = self.stack.find(input)?;
        // We have a nested goal which directly relies on a goal deeper in the stack.
        //
        // We start by tagging all cycle participants, as that's necessary for caching.
        //
        // Finally we can return either the provisional response or the initial response
        // in case we're in the first fixpoint iteration for this goal.
        let path_kind = Self::cycle_path_kind(&self.stack, step_kind_from_parent, head_index);
        debug!(?path_kind, "encountered cycle with depth {head_index:?}");
        let mut usages = HeadUsages::default();
        usages.add_usage(path_kind);
        let head = CycleHead { paths_to_head: step_kind_from_parent.into(), usages };
        Self::update_parent_goal(
            &mut self.stack,
            step_kind_from_parent,
            0,
            iter::once((head_index, head)),
            false,
            UpdateParentGoalCtxt::CycleOnStack(input),
        );

        // Return the provisional result or, if we're in the first iteration,
        // start with no constraints.
        if let Some(result) = self.stack[head_index].provisional_result {
            Some(result)
        } else {
            Some(D::initial_provisional_result(cx, path_kind, input))
        }
    }

    /// Whether we've reached a fixpoint when evaluating a cycle head.
    fn reached_fixpoint(
        &mut self,
        stack_entry: &StackEntry<X>,
        usages: HeadUsages,
        result: X::Result,
    ) -> Result<Option<PathKind>, ()> {
        let provisional_result = stack_entry.provisional_result;
        if let Some(provisional_result) = provisional_result {
            if provisional_result == result { Ok(None) } else { Err(()) }
        } else if let Some(path_kind) = D::is_initial_provisional_result(result)
            .filter(|&path_kind| usages.is_single(path_kind))
        {
            Ok(Some(path_kind))
        } else {
            Err(())
        }
    }

    /// When we encounter a coinductive cycle, we have to fetch the
    /// result of that cycle while we are still computing it. Because
    /// of this we continuously recompute the cycle until the result
    /// of the previous iteration is equal to the final result, at which
    /// point we are done.
    fn evaluate_goal_in_task(
        &mut self,
        cx: X,
        input: X::Input,
        inspect: &mut D::ProofTreeBuilder,
    ) -> EvaluationResult<X> {
        // We reset `encountered_overflow` each time we rerun this goal
        // but need to make sure we currently propagate it to the global
        // cache even if only some of the evaluations actually reach the
        // recursion limit.
        let mut encountered_overflow = false;
        let mut i = 0;
        loop {
            let result = D::compute_goal(self, cx, input, inspect);
            let stack_entry = self.stack.pop();
            encountered_overflow |= stack_entry.encountered_overflow;
            debug_assert_eq!(stack_entry.input, input);

            // If the current goal is not a cycle head, we are done.
            //
            // There are no provisional cache entries which depend on this goal.
            let Some(usages) = stack_entry.usages else {
                return EvaluationResult::finalize(stack_entry, encountered_overflow, result);
            };

            // If it is a cycle head, we have to keep trying to prove it until
            // we reach a fixpoint. We need to do so for all cycle heads,
            // not only for the root.
            //
            // See tests/ui/traits/next-solver/cycles/fixpoint-rerun-all-cycle-heads.rs
            // for an example.
            //
            // Check whether we reached a fixpoint, either because the final result
            // is equal to the provisional result of the previous iteration, or because
            // this was only the head of either coinductive or inductive cycles, and the
            // final result is equal to the initial response for that case.
            if let Ok(fixpoint) = self.reached_fixpoint(&stack_entry, usages, result) {
                self.rebase_provisional_cache_entries(
                    cx,
                    &stack_entry,
                    RebaseReason::ReachedFixpoint(fixpoint),
                );
                return EvaluationResult::finalize(stack_entry, encountered_overflow, result);
            } else if usages.is_empty() {
                self.rebase_provisional_cache_entries(
                    cx,
                    &stack_entry,
                    RebaseReason::NoCycleUsages,
                );
                return EvaluationResult::finalize(stack_entry, encountered_overflow, result);
            }

            // If computing this goal results in ambiguity with no constraints,
            // we do not rerun it. It's incredibly difficult to get a different
            // response in the next iteration in this case. These changes would
            // likely either be caused by incompleteness or can change the maybe
            // cause from ambiguity to overflow. Returning ambiguity always
            // preserves soundness and completeness even if the goal is be known
            // to succeed or fail.
            //
            // This prevents exponential blowup affecting multiple major crates.
            // As we only get to this branch if we haven't yet reached a fixpoint,
            // we also taint all provisional cache entries which depend on the
            // current goal.
            if D::is_ambiguous_result(result) {
                self.rebase_provisional_cache_entries(cx, &stack_entry, RebaseReason::Ambiguity);
                return EvaluationResult::finalize(stack_entry, encountered_overflow, result);
            };

            // If we've reached the fixpoint step limit, we bail with overflow and taint all
            // provisional cache entries which depend on the current goal.
            i += 1;
            if i >= D::FIXPOINT_STEP_LIMIT {
                debug!("canonical cycle overflow");
                let result = D::fixpoint_overflow_result(cx, input);
                self.rebase_provisional_cache_entries(cx, &stack_entry, RebaseReason::Overflow);
                return EvaluationResult::finalize(stack_entry, encountered_overflow, result);
            }

            // Clear all provisional cache entries which depend on a previous provisional
            // result of this goal and rerun.
            self.clear_dependent_provisional_results_for_rerun();

            debug!(?result, "fixpoint changed provisional results");
            self.stack.push(StackEntry {
                input,
                step_kind_from_parent: stack_entry.step_kind_from_parent,
                available_depth: stack_entry.available_depth,
                provisional_result: Some(result),
                // We can keep these goals from previous iterations as they are only
                // ever read after finalizing this evaluation.
                required_depth: stack_entry.required_depth,
                heads: stack_entry.heads,
                nested_goals: stack_entry.nested_goals,
                // We reset these two fields when rerunning this goal. We could
                // keep `encountered_overflow` as it's only used as a performance
                // optimization. However, given that the proof tree will likely look
                // similar to the previous iterations when reevaluating, it's better
                // for caching if the reevaluation also starts out with `false`.
                encountered_overflow: false,
                usages: None,
                candidate_usages: None,
            });
        }
    }

    /// When encountering a cycle, both inductive and coinductive, we only
    /// move the root into the global cache. We also store all other cycle
    /// participants involved.
    ///
    /// We must not use the global cache entry of a root goal if a cycle
    /// participant is on the stack. This is necessary to prevent unstable
    /// results. See the comment of `StackEntry::nested_goals` for
    /// more details.
    fn insert_global_cache(
        &mut self,
        cx: X,
        input: X::Input,
        evaluation_result: EvaluationResult<X>,
        dep_node: X::DepNodeIndex,
    ) {
        debug!(?evaluation_result, "insert global cache");
        cx.with_global_cache(|cache| cache.insert(cx, input, evaluation_result, dep_node))
    }
}
