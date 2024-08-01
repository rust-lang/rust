use std::fmt::Debug;
use std::hash::Hash;
use std::marker::PhantomData;
use std::mem;

use derive_where::derive_where;
use rustc_index::{Idx, IndexVec};
use tracing::debug;

use crate::data_structures::{HashMap, HashSet};
use crate::solve::SolverMode;

mod global_cache;
use global_cache::CacheData;
pub use global_cache::GlobalCache;
mod validate;

/// The search graph does not simply use `Interner` directly
/// to enable its fuzzing without having to stub the rest of
/// the interner. We don't make this a super trait of `Interner`
/// as users of the shared type library shouldn't have to care
/// about `Input` and `Result` as they are implementation details
/// of the search graph.
pub trait Cx: Copy {
    type ProofTree: Debug + Copy;
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

    fn with_global_cache<R>(
        self,
        mode: SolverMode,
        f: impl FnOnce(&mut GlobalCache<Self>) -> R,
    ) -> R;
}

pub trait ProofTreeBuilder<X: Cx> {
    fn try_apply_proof_tree(&mut self, proof_tree: X::ProofTree) -> bool;
    fn on_provisional_cache_hit(&mut self);
    fn on_cycle_in_stack(&mut self);
    fn finalize_canonical_goal_evaluation(&mut self, cx: X) -> X::ProofTree;
}

pub trait Delegate {
    type Cx: Cx;
    const FIXPOINT_STEP_LIMIT: usize;
    type ProofTreeBuilder: ProofTreeBuilder<Self::Cx>;

    fn recursion_limit(cx: Self::Cx) -> usize;

    fn initial_provisional_result(
        cx: Self::Cx,
        kind: CycleKind,
        input: <Self::Cx as Cx>::Input,
    ) -> <Self::Cx as Cx>::Result;
    fn reached_fixpoint(
        cx: Self::Cx,
        kind: UsageKind,
        input: <Self::Cx as Cx>::Input,
        provisional_result: Option<<Self::Cx as Cx>::Result>,
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

    fn step_is_coinductive(cx: Self::Cx, input: <Self::Cx as Cx>::Input) -> bool;
}

/// In the initial iteration of a cycle, we do not yet have a provisional
/// result. In the case we return an initial provisional result depending
/// on the kind of cycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CycleKind {
    Coinductive,
    Inductive,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UsageKind {
    Single(CycleKind),
    Mixed,
}
impl UsageKind {
    fn merge(self, other: Self) -> Self {
        match (self, other) {
            (UsageKind::Single(lhs), UsageKind::Single(rhs)) => {
                if lhs == rhs {
                    UsageKind::Single(lhs)
                } else {
                    UsageKind::Mixed
                }
            }
            (UsageKind::Mixed, UsageKind::Mixed)
            | (UsageKind::Mixed, UsageKind::Single(_))
            | (UsageKind::Single(_), UsageKind::Mixed) => UsageKind::Mixed,
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
        cx: D::Cx,
        stack: &IndexVec<StackDepth, StackEntry<D::Cx>>,
    ) -> Option<AvailableDepth> {
        if let Some(last) = stack.raw.last() {
            if last.available_depth.0 == 0 {
                return None;
            }

            Some(if last.encountered_overflow {
                AvailableDepth(last.available_depth.0 / 2)
            } else {
                AvailableDepth(last.available_depth.0 - 1)
            })
        } else {
            Some(AvailableDepth(D::recursion_limit(cx)))
        }
    }

    /// Whether we're allowed to use a global cache entry which required
    /// the given depth.
    fn cache_entry_is_applicable(self, additional_depth: usize) -> bool {
        self.0 >= additional_depth
    }
}

rustc_index::newtype_index! {
    #[orderable]
    #[gate_rustc_only]
    pub struct StackDepth {}
}

#[derive_where(Debug; X: Cx)]
struct StackEntry<X: Cx> {
    input: X::Input,

    available_depth: AvailableDepth,

    /// The maximum depth reached by this stack entry, only up-to date
    /// for the top of the stack and lazily updated for the rest.
    reached_depth: StackDepth,

    /// Whether this entry is a non-root cycle participant.
    ///
    /// We must not move the result of non-root cycle participants to the
    /// global cache. We store the highest stack depth of a head of a cycle
    /// this goal is involved in. This necessary to soundly cache its
    /// provisional result.
    non_root_cycle_participant: Option<StackDepth>,

    encountered_overflow: bool,

    has_been_used: Option<UsageKind>,

    /// We put only the root goal of a coinductive cycle into the global cache.
    ///
    /// If we were to use that result when later trying to prove another cycle
    /// participant, we can end up with unstable query results.
    ///
    /// See tests/ui/next-solver/coinduction/incompleteness-unstable-result.rs for
    /// an example of where this is needed.
    ///
    /// There can  be multiple roots on the same stack, so we need to track
    /// cycle participants per root:
    /// ```plain
    /// A :- B
    /// B :- A, C
    /// C :- D
    /// D :- C
    /// ```
    nested_goals: HashSet<X::Input>,
    /// Starts out as `None` and gets set when rerunning this
    /// goal in case we encounter a cycle.
    provisional_result: Option<X::Result>,
}

/// The provisional result for a goal which is not on the stack.
#[derive(Debug)]
struct DetachedEntry<X: Cx> {
    /// The head of the smallest non-trivial cycle involving this entry.
    ///
    /// Given the following rules, when proving `A` the head for
    /// the provisional entry of `C` would be `B`.
    /// ```plain
    /// A :- B
    /// B :- C
    /// C :- A + B + C
    /// ```
    head: StackDepth,
    result: X::Result,
}

/// Stores the stack depth of a currently evaluated goal *and* already
/// computed results for goals which depend on other goals still on the stack.
///
/// The provisional result may depend on whether the stack above it is inductive
/// or coinductive. Because of this, we store separate provisional results for
/// each case. If an provisional entry is not applicable, it may be the case
/// that we already have provisional result while computing a goal. In this case
/// we prefer the provisional result to potentially avoid fixpoint iterations.
/// See tests/ui/traits/next-solver/cycles/mixed-cycles-2.rs for an example.
///
/// The provisional cache can theoretically result in changes to the observable behavior,
/// see tests/ui/traits/next-solver/cycles/provisional-cache-impacts-behavior.rs.
#[derive_where(Default; X: Cx)]
struct ProvisionalCacheEntry<X: Cx> {
    stack_depth: Option<StackDepth>,
    with_inductive_stack: Option<DetachedEntry<X>>,
    with_coinductive_stack: Option<DetachedEntry<X>>,
}

impl<X: Cx> ProvisionalCacheEntry<X> {
    fn is_empty(&self) -> bool {
        self.stack_depth.is_none()
            && self.with_inductive_stack.is_none()
            && self.with_coinductive_stack.is_none()
    }
}

pub struct SearchGraph<D: Delegate<Cx = X>, X: Cx = <D as Delegate>::Cx> {
    mode: SolverMode,
    /// The stack of goals currently being computed.
    ///
    /// An element is *deeper* in the stack if its index is *lower*.
    stack: IndexVec<StackDepth, StackEntry<X>>,
    provisional_cache: HashMap<X::Input, ProvisionalCacheEntry<X>>,

    _marker: PhantomData<D>,
}

impl<D: Delegate<Cx = X>, X: Cx> SearchGraph<D> {
    pub fn new(mode: SolverMode) -> SearchGraph<D> {
        Self {
            mode,
            stack: Default::default(),
            provisional_cache: Default::default(),
            _marker: PhantomData,
        }
    }

    pub fn solver_mode(&self) -> SolverMode {
        self.mode
    }

    fn update_parent_goal(&mut self, reached_depth: StackDepth, encountered_overflow: bool) {
        if let Some(parent) = self.stack.raw.last_mut() {
            parent.reached_depth = parent.reached_depth.max(reached_depth);
            parent.encountered_overflow |= encountered_overflow;
        }
    }

    pub fn is_empty(&self) -> bool {
        self.stack.is_empty()
    }

    fn stack_coinductive_from(
        cx: X,
        stack: &IndexVec<StackDepth, StackEntry<X>>,
        head: StackDepth,
    ) -> bool {
        stack.raw[head.index()..].iter().all(|entry| D::step_is_coinductive(cx, entry.input))
    }

    // When encountering a solver cycle, the result of the current goal
    // depends on goals lower on the stack.
    //
    // We have to therefore be careful when caching goals. Only the final result
    // of the cycle root, i.e. the lowest goal on the stack involved in this cycle,
    // is moved to the global cache while all others are stored in a provisional cache.
    //
    // We update both the head of this cycle to rerun its evaluation until
    // we reach a fixpoint and all other cycle participants to make sure that
    // their result does not get moved to the global cache.
    fn tag_cycle_participants(
        stack: &mut IndexVec<StackDepth, StackEntry<X>>,
        usage_kind: Option<UsageKind>,
        head: StackDepth,
    ) {
        if let Some(usage_kind) = usage_kind {
            stack[head].has_been_used =
                Some(stack[head].has_been_used.map_or(usage_kind, |prev| prev.merge(usage_kind)));
        }
        debug_assert!(stack[head].has_been_used.is_some());

        // The current root of these cycles. Note that this may not be the final
        // root in case a later goal depends on a goal higher up the stack.
        let mut current_root = head;
        while let Some(parent) = stack[current_root].non_root_cycle_participant {
            current_root = parent;
            debug_assert!(stack[current_root].has_been_used.is_some());
        }

        let (stack, cycle_participants) = stack.raw.split_at_mut(head.index() + 1);
        let current_cycle_root = &mut stack[current_root.as_usize()];
        for entry in cycle_participants {
            entry.non_root_cycle_participant = entry.non_root_cycle_participant.max(Some(head));
            current_cycle_root.nested_goals.insert(entry.input);
            current_cycle_root.nested_goals.extend(mem::take(&mut entry.nested_goals));
        }
    }

    fn clear_dependent_provisional_results(
        provisional_cache: &mut HashMap<X::Input, ProvisionalCacheEntry<X>>,
        head: StackDepth,
    ) {
        #[allow(rustc::potential_query_instability)]
        provisional_cache.retain(|_, entry| {
            if entry.with_coinductive_stack.as_ref().is_some_and(|p| p.head == head) {
                entry.with_coinductive_stack.take();
            }
            if entry.with_inductive_stack.as_ref().is_some_and(|p| p.head == head) {
                entry.with_inductive_stack.take();
            }
            !entry.is_empty()
        });
    }

    /// Probably the most involved method of the whole solver.
    ///
    /// Given some goal which is proven via the `prove_goal` closure, this
    /// handles caching, overflow, and coinductive cycles.
    pub fn with_new_goal(
        &mut self,
        cx: X,
        input: X::Input,
        inspect: &mut D::ProofTreeBuilder,
        mut prove_goal: impl FnMut(&mut Self, &mut D::ProofTreeBuilder) -> X::Result,
    ) -> X::Result {
        self.check_invariants();
        // Check for overflow.
        let Some(available_depth) = AvailableDepth::allowed_depth_for_nested::<D>(cx, &self.stack)
        else {
            if let Some(last) = self.stack.raw.last_mut() {
                last.encountered_overflow = true;
            }

            debug!("encountered stack overflow");
            return D::on_stack_overflow(cx, inspect, input);
        };

        if let Some(result) = self.lookup_global_cache(cx, input, available_depth, inspect) {
            return result;
        }

        // Check whether the goal is in the provisional cache.
        // The provisional result may rely on the path to its cycle roots,
        // so we have to check the path of the current goal matches that of
        // the cache entry.
        let cache_entry = self.provisional_cache.entry(input).or_default();
        if let Some(entry) = cache_entry
            .with_coinductive_stack
            .as_ref()
            .filter(|p| Self::stack_coinductive_from(cx, &self.stack, p.head))
            .or_else(|| {
                cache_entry
                    .with_inductive_stack
                    .as_ref()
                    .filter(|p| !Self::stack_coinductive_from(cx, &self.stack, p.head))
            })
        {
            debug!("provisional cache hit");
            // We have a nested goal which is already in the provisional cache, use
            // its result. We do not provide any usage kind as that should have been
            // already set correctly while computing the cache entry.
            inspect.on_provisional_cache_hit();
            Self::tag_cycle_participants(&mut self.stack, None, entry.head);
            return entry.result;
        } else if let Some(stack_depth) = cache_entry.stack_depth {
            debug!("encountered cycle with depth {stack_depth:?}");
            // We have a nested goal which directly relies on a goal deeper in the stack.
            //
            // We start by tagging all cycle participants, as that's necessary for caching.
            //
            // Finally we can return either the provisional response or the initial response
            // in case we're in the first fixpoint iteration for this goal.
            inspect.on_cycle_in_stack();

            let is_coinductive_cycle = Self::stack_coinductive_from(cx, &self.stack, stack_depth);
            let cycle_kind =
                if is_coinductive_cycle { CycleKind::Coinductive } else { CycleKind::Inductive };
            Self::tag_cycle_participants(
                &mut self.stack,
                Some(UsageKind::Single(cycle_kind)),
                stack_depth,
            );

            // Return the provisional result or, if we're in the first iteration,
            // start with no constraints.
            return if let Some(result) = self.stack[stack_depth].provisional_result {
                result
            } else {
                D::initial_provisional_result(cx, cycle_kind, input)
            };
        } else {
            // No entry, we push this goal on the stack and try to prove it.
            let depth = self.stack.next_index();
            let entry = StackEntry {
                input,
                available_depth,
                reached_depth: depth,
                non_root_cycle_participant: None,
                encountered_overflow: false,
                has_been_used: None,
                nested_goals: Default::default(),
                provisional_result: None,
            };
            assert_eq!(self.stack.push(entry), depth);
            cache_entry.stack_depth = Some(depth);
        };

        // This is for global caching, so we properly track query dependencies.
        // Everything that affects the `result` should be performed within this
        // `with_anon_task` closure. If computing this goal depends on something
        // not tracked by the cache key and from outside of this anon task, it
        // must not be added to the global cache. Notably, this is the case for
        // trait solver cycles participants.
        let ((final_entry, result), dep_node) = cx.with_cached_task(|| {
            for _ in 0..D::FIXPOINT_STEP_LIMIT {
                match self.fixpoint_step_in_task(cx, input, inspect, &mut prove_goal) {
                    StepResult::Done(final_entry, result) => return (final_entry, result),
                    StepResult::HasChanged => debug!("fixpoint changed provisional results"),
                }
            }

            debug!("canonical cycle overflow");
            let current_entry = self.stack.pop().unwrap();
            debug_assert!(current_entry.has_been_used.is_none());
            let result = D::on_fixpoint_overflow(cx, input);
            (current_entry, result)
        });

        let proof_tree = inspect.finalize_canonical_goal_evaluation(cx);

        self.update_parent_goal(final_entry.reached_depth, final_entry.encountered_overflow);

        // We're now done with this goal. In case this goal is involved in a larger cycle
        // do not remove it from the provisional cache and update its provisional result.
        // We only add the root of cycles to the global cache.
        if let Some(head) = final_entry.non_root_cycle_participant {
            let coinductive_stack = Self::stack_coinductive_from(cx, &self.stack, head);

            let entry = self.provisional_cache.get_mut(&input).unwrap();
            entry.stack_depth = None;
            if coinductive_stack {
                entry.with_coinductive_stack = Some(DetachedEntry { head, result });
            } else {
                entry.with_inductive_stack = Some(DetachedEntry { head, result });
            }
        } else {
            // When encountering a cycle, both inductive and coinductive, we only
            // move the root into the global cache. We also store all other cycle
            // participants involved.
            //
            // We must not use the global cache entry of a root goal if a cycle
            // participant is on the stack. This is necessary to prevent unstable
            // results. See the comment of `StackEntry::nested_goals` for
            // more details.
            self.provisional_cache.remove(&input);
            let additional_depth = final_entry.reached_depth.as_usize() - self.stack.len();
            cx.with_global_cache(self.mode, |cache| {
                cache.insert(
                    cx,
                    input,
                    result,
                    proof_tree,
                    dep_node,
                    additional_depth,
                    final_entry.encountered_overflow,
                    &final_entry.nested_goals,
                )
            })
        }

        self.check_invariants();

        result
    }

    /// Try to fetch a previously computed result from the global cache,
    /// making sure to only do so if it would match the result of reevaluating
    /// this goal.
    fn lookup_global_cache(
        &mut self,
        cx: X,
        input: X::Input,
        available_depth: AvailableDepth,
        inspect: &mut D::ProofTreeBuilder,
    ) -> Option<X::Result> {
        cx.with_global_cache(self.mode, |cache| {
            let CacheData {
                result,
                proof_tree,
                additional_depth,
                encountered_overflow,
                nested_goals: _, // FIXME: consider nested goals here.
            } = cache.get(cx, input, &self.stack, available_depth)?;

            // If we're building a proof tree and the current cache entry does not
            // contain a proof tree, we do not use the entry but instead recompute
            // the goal. We simply overwrite the existing entry once we're done,
            // caching the proof tree.
            if !inspect.try_apply_proof_tree(proof_tree) {
                return None;
            }

            // Update the reached depth of the current goal to make sure
            // its state is the same regardless of whether we've used the
            // global cache or not.
            let reached_depth = self.stack.next_index().plus(additional_depth);
            self.update_parent_goal(reached_depth, encountered_overflow);

            debug!("global cache hit");
            Some(result)
        })
    }
}

enum StepResult<X: Cx> {
    Done(StackEntry<X>, X::Result),
    HasChanged,
}

impl<D: Delegate<Cx = X>, X: Cx> SearchGraph<D> {
    /// When we encounter a coinductive cycle, we have to fetch the
    /// result of that cycle while we are still computing it. Because
    /// of this we continuously recompute the cycle until the result
    /// of the previous iteration is equal to the final result, at which
    /// point we are done.
    fn fixpoint_step_in_task<F>(
        &mut self,
        cx: X,
        input: X::Input,
        inspect: &mut D::ProofTreeBuilder,
        prove_goal: &mut F,
    ) -> StepResult<X>
    where
        F: FnMut(&mut Self, &mut D::ProofTreeBuilder) -> X::Result,
    {
        let result = prove_goal(self, inspect);
        let stack_entry = self.stack.pop().unwrap();
        debug_assert_eq!(stack_entry.input, input);

        // If the current goal is not the root of a cycle, we are done.
        let Some(usage_kind) = stack_entry.has_been_used else {
            return StepResult::Done(stack_entry, result);
        };

        // If it is a cycle head, we have to keep trying to prove it until
        // we reach a fixpoint. We need to do so for all cycle heads,
        // not only for the root.
        //
        // See tests/ui/traits/next-solver/cycles/fixpoint-rerun-all-cycle-heads.rs
        // for an example.

        // Start by clearing all provisional cache entries which depend on this
        // the current goal.
        Self::clear_dependent_provisional_results(
            &mut self.provisional_cache,
            self.stack.next_index(),
        );

        // Check whether we reached a fixpoint, either because the final result
        // is equal to the provisional result of the previous iteration, or because
        // this was only the root of either coinductive or inductive cycles, and the
        // final result is equal to the initial response for that case.
        //
        // If we did not reach a fixpoint, update the provisional result and reevaluate.
        if D::reached_fixpoint(cx, usage_kind, input, stack_entry.provisional_result, result) {
            StepResult::Done(stack_entry, result)
        } else {
            let depth = self.stack.push(StackEntry {
                has_been_used: None,
                provisional_result: Some(result),
                ..stack_entry
            });
            debug_assert_eq!(self.provisional_cache[&input].stack_depth, Some(depth));
            StepResult::HasChanged
        }
    }
}
