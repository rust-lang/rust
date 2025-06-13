/// The search graph is responsible for caching and cycle detection in the trait
/// solver. Making sure that caching doesn't result in soundness bugs or unstable
/// query results is very challenging and makes this one of the most-involved
/// self-contained components of the compiler.
///
/// We added fuzzing support to test its correctness. The fuzzers used to verify
/// the current implementation can be found in https://github.com/lcnr/search_graph_fuzz.
///
/// This is just a quick overview of the general design, please check out the relevant
/// [rustc-dev-guide chapter](https://rustc-dev-guide.rust-lang.org/solve/caching.html) for
/// more details. Caching is split between a global cache and the per-cycle `provisional_cache`.
/// The global cache has to be completely unobservable, while the per-cycle cache may impact
/// behavior as long as the resulting behavior is still correct.
use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::collections::hash_map::Entry;
use std::fmt::Debug;
use std::hash::Hash;
use std::marker::PhantomData;

use derive_where::derive_where;
#[cfg(feature = "nightly")]
use rustc_macros::{Decodable_NoContext, Encodable_NoContext, HashStable_NoContext};
use tracing::debug;

use crate::data_structures::HashMap;

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

    fn evaluation_is_concurrent(&self) -> bool;
}

pub trait Delegate {
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
    fn is_initial_provisional_result(
        cx: Self::Cx,
        kind: PathKind,
        input: <Self::Cx as Cx>::Input,
        result: <Self::Cx as Cx>::Result,
    ) -> bool;
    fn on_stack_overflow(
        cx: Self::Cx,
        inspect: &mut Self::ProofTreeBuilder,
        input: <Self::Cx as Cx>::Input,
    ) -> <Self::Cx as Cx>::Result;
    fn on_fixpoint_overflow(
        cx: Self::Cx,
        input: <Self::Cx as Cx>::Input,
    ) -> <Self::Cx as Cx>::Result;

    fn is_ambiguous_result(result: <Self::Cx as Cx>::Result) -> bool;
    fn propagate_ambiguity(
        cx: Self::Cx,
        for_input: <Self::Cx as Cx>::Input,
        from_result: <Self::Cx as Cx>::Result,
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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UsageKind {
    Single(PathKind),
    Mixed,
}
impl From<PathKind> for UsageKind {
    fn from(path: PathKind) -> UsageKind {
        UsageKind::Single(path)
    }
}
impl UsageKind {
    #[must_use]
    fn merge(self, other: impl Into<Self>) -> Self {
        match (self, other.into()) {
            (UsageKind::Mixed, _) | (_, UsageKind::Mixed) => UsageKind::Mixed,
            (UsageKind::Single(lhs), UsageKind::Single(rhs)) => {
                if lhs == rhs {
                    UsageKind::Single(lhs)
                } else {
                    UsageKind::Mixed
                }
            }
        }
    }
}

/// For each goal we track whether the paths from this goal
/// to its cycle heads are coinductive.
///
/// This is a necessary condition to rebase provisional cache
/// entries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AllPathsToHeadCoinductive {
    Yes,
    No,
}
impl From<PathKind> for AllPathsToHeadCoinductive {
    fn from(path: PathKind) -> AllPathsToHeadCoinductive {
        match path {
            PathKind::Coinductive => AllPathsToHeadCoinductive::Yes,
            _ => AllPathsToHeadCoinductive::No,
        }
    }
}
impl AllPathsToHeadCoinductive {
    #[must_use]
    fn merge(self, other: impl Into<Self>) -> Self {
        match (self, other.into()) {
            (AllPathsToHeadCoinductive::Yes, AllPathsToHeadCoinductive::Yes) => {
                AllPathsToHeadCoinductive::Yes
            }
            (AllPathsToHeadCoinductive::No, _) | (_, AllPathsToHeadCoinductive::No) => {
                AllPathsToHeadCoinductive::No
            }
        }
    }
    fn and_merge(&mut self, other: impl Into<Self>) {
        *self = self.merge(other);
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

/// All cycle heads a given goal depends on, ordered by their stack depth.
///
/// We also track all paths from this goal to that head. This is necessary
/// when rebasing provisional cache results.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
struct CycleHeads {
    heads: BTreeMap<StackDepth, AllPathsToHeadCoinductive>,
}

impl CycleHeads {
    fn is_empty(&self) -> bool {
        self.heads.is_empty()
    }

    fn highest_cycle_head(&self) -> StackDepth {
        self.opt_highest_cycle_head().unwrap()
    }

    fn opt_highest_cycle_head(&self) -> Option<StackDepth> {
        self.heads.last_key_value().map(|(k, _)| *k)
    }

    fn opt_lowest_cycle_head(&self) -> Option<StackDepth> {
        self.heads.first_key_value().map(|(k, _)| *k)
    }

    fn remove_highest_cycle_head(&mut self) {
        let last = self.heads.pop_last();
        debug_assert_ne!(last, None);
    }

    fn insert(
        &mut self,
        head: StackDepth,
        path_from_entry: impl Into<AllPathsToHeadCoinductive> + Copy,
    ) {
        self.heads.entry(head).or_insert(path_from_entry.into()).and_merge(path_from_entry);
    }

    fn merge(&mut self, heads: &CycleHeads) {
        for (&head, &path_from_entry) in heads.heads.iter() {
            self.insert(head, path_from_entry);
            debug_assert!(matches!(self.heads[&head], AllPathsToHeadCoinductive::Yes));
        }
    }

    fn iter(&self) -> impl Iterator<Item = (StackDepth, AllPathsToHeadCoinductive)> + '_ {
        self.heads.iter().map(|(k, v)| (*k, *v))
    }

    /// Update the cycle heads of a goal at depth `this` given the cycle heads
    /// of a nested goal. This merges the heads after filtering the parent goal
    /// itself.
    fn extend_from_child(&mut self, this: StackDepth, step_kind: PathKind, child: &CycleHeads) {
        for (&head, &path_from_entry) in child.heads.iter() {
            match head.cmp(&this) {
                Ordering::Less => {}
                Ordering::Equal => continue,
                Ordering::Greater => unreachable!(),
            }

            let path_from_entry = match step_kind {
                PathKind::Coinductive => AllPathsToHeadCoinductive::Yes,
                PathKind::Unknown | PathKind::Inductive => path_from_entry,
            };

            self.insert(head, path_from_entry);
        }
    }
}

bitflags::bitflags! {
    /// Tracks how nested goals have been accessed. This is necessary to disable
    /// global cache entries if computing them would otherwise result in a cycle or
    /// access a provisional cache entry.
    #[derive(Debug, Clone, Copy)]
    pub struct PathsToNested: u8 {
        /// The initial value when adding a goal to its own nested goals.
        const EMPTY                      = 1 << 0;
        const INDUCTIVE                  = 1 << 1;
        const UNKNOWN                    = 1 << 2;
        const COINDUCTIVE                = 1 << 3;
    }
}
impl From<PathKind> for PathsToNested {
    fn from(path: PathKind) -> PathsToNested {
        match path {
            PathKind::Inductive => PathsToNested::INDUCTIVE,
            PathKind::Unknown => PathsToNested::UNKNOWN,
            PathKind::Coinductive => PathsToNested::COINDUCTIVE,
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
        }

        self
    }
}

/// The nested goals of each stack entry and the path from the
/// stack entry to that nested goal.
///
/// We only start tracking nested goals once we've either encountered
/// overflow or a solver cycle. This is a performance optimization to
/// avoid tracking nested goals on the happy path.
///
/// We use nested goals for two reasons:
/// - when rebasing provisional cache entries
/// - when checking whether we have to ignore a global cache entry as reevaluating
///   it would encounter a cycle or use a provisional cache entry.
///
/// We need to disable the global cache if using it would hide a cycle, as
/// cycles can impact behavior. The cycle ABA may have different final
/// results from a the cycle BAB depending on the cycle root.
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

pub struct SearchGraph<D: Delegate<Cx = X>, X: Cx = <D as Delegate>::Cx> {
    root_depth: AvailableDepth,
    /// The stack of goals currently being computed.
    ///
    /// An element is *deeper* in the stack if its index is *lower*.
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
/// Given that the usage of the provisional cache is fully determinstic, we
/// don't need to track the nested goals used while computing a provisional
/// cache entry.
enum UpdateParentGoalCtxt<'a, X: Cx> {
    Ordinary(&'a NestedGoals<X>),
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
        heads: &CycleHeads,
        encountered_overflow: bool,
        context: UpdateParentGoalCtxt<'_, X>,
    ) {
        if let Some(parent_index) = stack.last_index() {
            let parent = &mut stack[parent_index];
            parent.required_depth = parent.required_depth.max(required_depth_for_nested + 1);
            parent.encountered_overflow |= encountered_overflow;

            parent.heads.extend_from_child(parent_index, step_kind_from_parent, heads);
            let parent_depends_on_cycle = match context {
                UpdateParentGoalCtxt::Ordinary(nested_goals) => {
                    parent.nested_goals.extend_from_child(step_kind_from_parent, nested_goals);
                    !nested_goals.is_empty()
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

    /// Probably the most involved method of the whole solver.
    ///
    /// Given some goal which is proven via the `prove_goal` closure, this
    /// handles caching, overflow, and coinductive cycles.
    pub fn with_new_goal(
        &mut self,
        cx: X,
        input: X::Input,
        step_kind_from_parent: PathKind,
        inspect: &mut D::ProofTreeBuilder,
        mut evaluate_goal: impl FnMut(&mut Self, &mut D::ProofTreeBuilder) -> X::Result,
    ) -> X::Result {
        let Some(available_depth) =
            AvailableDepth::allowed_depth_for_nested::<D>(self.root_depth, &self.stack)
        else {
            return self.handle_overflow(cx, input, inspect);
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
            required_depth: 0,
            heads: Default::default(),
            encountered_overflow: false,
            has_been_used: None,
            nested_goals: Default::default(),
            provisional_result: None,
        });

        // This is for global caching, so we properly track query dependencies.
        // Everything that affects the `result` should be performed within this
        // `with_cached_task` closure. If computing this goal depends on something
        // not tracked by the cache key and from outside of this anon task, it
        // must not be added to the global cache. Notably, this is the case for
        // trait solver cycles participants.
        let ((final_entry, result), dep_node) = cx.with_cached_task(|| {
            self.evaluate_goal_in_task(cx, input, inspect, &mut evaluate_goal)
        });

        // We've finished computing the goal and have popped it from the stack,
        // lazily update its parent goal.
        Self::update_parent_goal(
            &mut self.stack,
            final_entry.step_kind_from_parent,
            final_entry.required_depth,
            &final_entry.heads,
            final_entry.encountered_overflow,
            UpdateParentGoalCtxt::Ordinary(&final_entry.nested_goals),
        );

        // We're now done with this goal. We only add the root of cycles to the global cache.
        // In case this goal is involved in a larger cycle add it to the provisional cache.
        if final_entry.heads.is_empty() {
            if let Some((_scope, expected)) = validate_cache {
                // Do not try to move a goal into the cache again if we're testing
                // the global cache.
                assert_eq!(result, expected, "input={input:?}");
            } else if D::inspect_is_noop(inspect) {
                self.insert_global_cache(cx, final_entry, result, dep_node)
            }
        } else if D::ENABLE_PROVISIONAL_CACHE {
            debug_assert!(validate_cache.is_none(), "unexpected non-root: {input:?}");
            let entry = self.provisional_cache.entry(input).or_default();
            let StackEntry { heads, encountered_overflow, .. } = final_entry;
            let path_from_head = Self::cycle_path_kind(
                &self.stack,
                step_kind_from_parent,
                heads.highest_cycle_head(),
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

    fn handle_overflow(
        &mut self,
        cx: X,
        input: X::Input,
        inspect: &mut D::ProofTreeBuilder,
    ) -> X::Result {
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
        D::on_stack_overflow(cx, inspect, input)
    }

    /// When reevaluating a goal with a changed provisional result, all provisional cache entry
    /// which depend on this goal get invalidated.
    fn clear_dependent_provisional_results(&mut self) {
        let head = self.stack.next_index();
        #[allow(rustc::potential_query_instability)]
        self.provisional_cache.retain(|_, entries| {
            entries.retain(|entry| entry.heads.highest_cycle_head() != head);
            !entries.is_empty()
        });
    }

    /// A necessary optimization to handle complex solver cycles. A provisional cache entry
    /// relies on a set of cycle heads and the path towards these heads. When popping a cycle
    /// head from the stack after we've finished computing it, we can't be sure that the
    /// provisional cache entry is still applicable. We need to keep the cache entries to
    /// prevent hangs.
    ///
    /// What we therefore do is check whether the cycle kind of all cycles the goal of a
    /// provisional cache entry is involved in would stay the same when computing the
    /// goal without its cycle head on the stack. For more details, see the relevant
    /// [rustc-dev-guide chapter](https://rustc-dev-guide.rust-lang.org/solve/caching.html).
    ///
    /// This can be thought of rotating the sub-tree of this provisional result and changing
    /// its entry point while making sure that all paths through this sub-tree stay the same.
    ///
    /// In case the popped cycle head failed to reach a fixpoint anything which depends on
    /// its provisional result is invalid. Actually discarding provisional cache entries in
    /// this case would cause hangs, so we instead change the result of dependant provisional
    /// cache entries to also be ambiguous. This causes some undesirable ambiguity for nested
    /// goals whose result doesn't actually depend on this cycle head, but that's acceptable
    /// to me.
    fn rebase_provisional_cache_entries(
        &mut self,
        stack_entry: &StackEntry<X>,
        mut mutate_result: impl FnMut(X::Input, X::Result) -> X::Result,
    ) {
        let head = self.stack.next_index();
        #[allow(rustc::potential_query_instability)]
        self.provisional_cache.retain(|&input, entries| {
            entries.retain_mut(|entry| {
                let ProvisionalCacheEntry {
                    encountered_overflow: _,
                    heads,
                    path_from_head,
                    result,
                } = entry;
                if heads.highest_cycle_head() == head {
                    heads.remove_highest_cycle_head()
                } else {
                    return true;
                }

                // We only try to rebase if all paths from the cache entry
                // to its heads are coinductive. In this case these cycle
                // kinds won't change, no matter the goals between these
                // heads and the provisional cache entry.
                if heads.iter().any(|(_, p)| matches!(p, AllPathsToHeadCoinductive::No)) {
                    return false;
                }

                // The same for nested goals of the cycle head.
                if stack_entry.heads.iter().any(|(_, p)| matches!(p, AllPathsToHeadCoinductive::No))
                {
                    return false;
                }

                // Merge the cycle heads of the provisional cache entry and the
                // popped head. If the popped cycle head was a root, discard all
                // provisional cache entries which depend on it.
                heads.merge(&stack_entry.heads);
                let Some(head) = heads.opt_highest_cycle_head() else {
                    return false;
                };

                // We now care about the path from the next highest cycle head to the
                // provisional cache entry.
                *path_from_head = path_from_head.extend(Self::cycle_path_kind(
                    &self.stack,
                    stack_entry.step_kind_from_parent,
                    head,
                ));
                // Mutate the result of the provisional cache entry in case we did
                // not reach a fixpoint.
                *result = mutate_result(input, *result);
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
            let head = heads.highest_cycle_head();
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
                if last.heads.opt_lowest_cycle_head().is_none_or(|lowest| lowest > head) {
                    continue;
                }
            }

            // A provisional cache entry is only valid if the current path from its
            // highest cycle head to the goal is the same.
            if path_from_head == Self::cycle_path_kind(&self.stack, step_kind_from_parent, head) {
                Self::update_parent_goal(
                    &mut self.stack,
                    step_kind_from_parent,
                    0,
                    heads,
                    encountered_overflow,
                    UpdateParentGoalCtxt::ProvisionalCacheHit,
                );
                debug_assert!(self.stack[head].has_been_used.is_some());
                debug!(?head, ?path_from_head, "provisional cache hit");
                return Some(result);
            }
        }

        None
    }

    /// Even if there is a global cache entry for a given goal, we need to make sure
    /// evaluating this entry would not have ended up depending on either a goal
    /// already on the stack or a provisional cache entry.
    fn candidate_is_applicable(
        stack: &Stack<X>,
        step_kind_from_parent: PathKind,
        provisional_cache: &HashMap<X::Input, Vec<ProvisionalCacheEntry<X>>>,
        nested_goals: &NestedGoals<X>,
    ) -> bool {
        // If the global cache entry didn't depend on any nested goals, it always
        // applies.
        if nested_goals.is_empty() {
            return true;
        }

        // If a nested goal of the global cache entry is on the stack, we would
        // definitely encounter a cycle.
        if stack.iter().any(|e| nested_goals.contains(e.input)) {
            debug!("cache entry not applicable due to stack");
            return false;
        }

        // The global cache entry is also invalid if there's a provisional cache entry
        // would apply for any of its nested goals.
        #[allow(rustc::potential_query_instability)]
        for (input, path_from_global_entry) in nested_goals.iter() {
            let Some(entries) = provisional_cache.get(&input) else {
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
                let head = heads.highest_cycle_head();
                let head_to_curr = Self::cycle_path_kind(stack, step_kind_from_parent, head);
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
                    Self::candidate_is_applicable(
                        &self.stack,
                        step_kind_from_parent,
                        &self.provisional_cache,
                        nested_goals,
                    )
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
                    Self::candidate_is_applicable(
                        &self.stack,
                        step_kind_from_parent,
                        &self.provisional_cache,
                        nested_goals,
                    )
                })?;

            // We don't move cycle participants to the global cache, so the
            // cycle heads are always empty.
            let heads = Default::default();
            Self::update_parent_goal(
                &mut self.stack,
                step_kind_from_parent,
                required_depth,
                &heads,
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
        let head = self.stack.find(input)?;
        // We have a nested goal which directly relies on a goal deeper in the stack.
        //
        // We start by tagging all cycle participants, as that's necessary for caching.
        //
        // Finally we can return either the provisional response or the initial response
        // in case we're in the first fixpoint iteration for this goal.
        let path_kind = Self::cycle_path_kind(&self.stack, step_kind_from_parent, head);
        debug!(?path_kind, "encountered cycle with depth {head:?}");
        let usage_kind = UsageKind::Single(path_kind);
        self.stack[head].has_been_used =
            Some(self.stack[head].has_been_used.map_or(usage_kind, |prev| prev.merge(usage_kind)));

        // Subtle: when encountering a cyclic goal, we still first checked for overflow,
        // so we have to update the reached depth.
        let last_index = self.stack.last_index().unwrap();
        let last = &mut self.stack[last_index];
        last.required_depth = last.required_depth.max(1);

        last.nested_goals.insert(input, step_kind_from_parent.into());
        last.nested_goals.insert(last.input, PathsToNested::EMPTY);
        if last_index != head {
            last.heads.insert(head, step_kind_from_parent);
        }

        // Return the provisional result or, if we're in the first iteration,
        // start with no constraints.
        if let Some(result) = self.stack[head].provisional_result {
            Some(result)
        } else {
            Some(D::initial_provisional_result(cx, path_kind, input))
        }
    }

    /// Whether we've reached a fixpoint when evaluating a cycle head.
    fn reached_fixpoint(
        &mut self,
        cx: X,
        stack_entry: &StackEntry<X>,
        usage_kind: UsageKind,
        result: X::Result,
    ) -> bool {
        if let Some(prev) = stack_entry.provisional_result {
            prev == result
        } else if let UsageKind::Single(kind) = usage_kind {
            D::is_initial_provisional_result(cx, kind, stack_entry.input, result)
        } else {
            false
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
        mut evaluate_goal: impl FnMut(&mut Self, &mut D::ProofTreeBuilder) -> X::Result,
    ) -> (StackEntry<X>, X::Result) {
        let mut i = 0;
        loop {
            let result = evaluate_goal(self, inspect);
            let stack_entry = self.stack.pop();
            debug_assert_eq!(stack_entry.input, input);

            // If the current goal is not the root of a cycle, we are done.
            //
            // There are no provisional cache entries which depend on this goal.
            let Some(usage_kind) = stack_entry.has_been_used else {
                return (stack_entry, result);
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
            // this was only the root of either coinductive or inductive cycles, and the
            // final result is equal to the initial response for that case.
            if self.reached_fixpoint(cx, &stack_entry, usage_kind, result) {
                self.rebase_provisional_cache_entries(&stack_entry, |_, result| result);
                return (stack_entry, result);
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
                self.rebase_provisional_cache_entries(&stack_entry, |input, _| {
                    D::propagate_ambiguity(cx, input, result)
                });
                return (stack_entry, result);
            };

            // If we've reached the fixpoint step limit, we bail with overflow and taint all
            // provisional cache entries which depend on the current goal.
            i += 1;
            if i >= D::FIXPOINT_STEP_LIMIT {
                debug!("canonical cycle overflow");
                let result = D::on_fixpoint_overflow(cx, input);
                self.rebase_provisional_cache_entries(&stack_entry, |input, _| {
                    D::on_fixpoint_overflow(cx, input)
                });
                return (stack_entry, result);
            }

            // Clear all provisional cache entries which depend on a previous provisional
            // result of this goal and rerun.
            self.clear_dependent_provisional_results();

            debug!(?result, "fixpoint changed provisional results");
            self.stack.push(StackEntry {
                has_been_used: None,
                provisional_result: Some(result),
                ..stack_entry
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
        final_entry: StackEntry<X>,
        result: X::Result,
        dep_node: X::DepNodeIndex,
    ) {
        debug!(?final_entry, ?result, "insert global cache");
        cx.with_global_cache(|cache| {
            cache.insert(
                cx,
                final_entry.input,
                result,
                dep_node,
                final_entry.required_depth,
                final_entry.encountered_overflow,
                final_entry.nested_goals,
            )
        })
    }
}
