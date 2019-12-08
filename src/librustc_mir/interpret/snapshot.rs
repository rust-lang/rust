//! Capture and compare snapshots of the compile-time interpreter state to detect when a program
//! will loop infinitely.
//!
//! This lives in `interpret` because it needs access to all sots of private state.  However,
//! it is not used by the general miri engine, just by CTFE.
//!
//! # Comparing interpreter state
//!
//! If we observe the same interpreter state at two different points in time during const-eval, we
//! can be certain that the program is in an infinite loop. For `CompileTimeInterpreter`, the
//! relevant bits of interpreter state are the stack and the heap. We refer to a copy of this state
//! as a "snapshot".
//!
//! While comparing stacks between snapshots is straightforward, comparing heaps between snapshots
//! is not. This is because `AllocId`s, which determine the target of a pointer, are frequently
//! thrown away and recreated, even if the actual value in memory did not change. Consider the
//! following code:
//!
//! ```rust,ignore(const-loop)
//! const fn inf_loop() {
//!     // Function call to prevent promotion.
//!     const fn zeros() -> [isize; 4] {
//!         [0; 4]
//!     }
//!
//!     loop {
//!         let arr = &zeros();
//!         if false {
//!             break;
//!         }
//!     }
//! }
//! ```
//!
//! Although this program will loop indefinitely, a new `AllocId` will be created for the value
//! pointed to by `arr` at each iteration of the loop. A naive method for comparing snapshots, one
//! that considered the numeric value of `AllocId`s, would cause this class of infinite loops to be
//! missed. See [#52475](https://github.com/rust-lang/rust/issues/52475).
//!
//! Instead, we assign a new, linear index, one that is used only while comparing snapshots, to
//! each allocation. This index, hereby referred to as the "DFS index" is the order that each
//! allocation would be encountered during a depth-first search of all allocations reachable from
//! the stack via a pointer. See below for an example of this numbering scheme.
//!
//! ```text
//! Stack pointers:    x  x    (DFS starts with leftmost pointer on top of stack)
//!                    |  |
//! Heap allocs:     1 o  o 3
//!                    | /|
//!                    |/ |
//!                  2 o  o 4
//! ```
//!
//! Instead of comparing `AllocId`s between snapshots, we first map `AllocId`s in each snapshot to
//! their DFS index, then compare those instead.

use std::convert::TryFrom;
use std::hash::{Hash, Hasher};

use rustc::mir;
use rustc::mir::interpret::{Allocation, InterpResult};
use rustc::ty::layout::Size;
use rustc_data_structures::fx::{FxHasher, FxHashMap, FxHashSet};

use super::{memory, AllocId, Frame, Immediate, MemPlace, Operand};
use crate::const_eval::CompileTimeInterpreter;
use crate::interpret::{LocalState, LocalValue};

type CtfeMemory<'mir, 'tcx> = Memory<'mir, 'tcx, CompileTimeInterpreter<'mir, 'tcx>>;

#[derive(Default)]
pub(crate) struct InfiniteLoopDetector<'mir, 'tcx> {
    /// The set of all `InterpSnapshot` *hashes* observed by this detector.
    ///
    /// When a collision occurs in this table, we store the full snapshot in
    /// `snapshots`.
    hashes: FxHashSet<u64>,

    /// The set of all `InterpSnapshot`s observed by this detector.
    ///
    /// An `InterpSnapshot` will only be fully cloned once it has caused a
    /// collision in `hashes`. As a result, the detector must observe at least
    /// *two* full cycles of an infinite loop before it triggers.
    snapshots: FxHashSet<InterpSnapshot<'mir, 'tcx>>,
}

impl<'mir, 'tcx> InfiniteLoopDetector<'mir, 'tcx> {
    pub fn observe_and_analyze(
        &mut self,
        stack: &[Frame<'mir, 'tcx>],
        memory: &CtfeMemory<'mir, 'tcx>,
    ) -> InterpResult<'tcx, ()> {
        let snapshot = InterpRef::new(stack, memory);

        let mut hasher = FxHasher::default();
        snapshot.hash(&mut hasher);
        let hash = hasher.finish();

        if self.hashes.insert(hash) {
            // No collision
            return Ok(())
        }

        // We need to make a full copy. NOW things start to get really expensive.
        if self.snapshots.insert(InterpSnapshot::capture(stack, memory)) {
            // Spurious collision or first cycle
            return Ok(())
        }

        // Second cycle
        throw_exhaust!(InfiniteLoop)
    }

    /// Returns `true` if the `InfiniteLoopDetector` has not yet been invoked.
    pub fn is_empty(&self) -> bool {
        self.hashes.is_empty()
    }
}

/// A copy of the virtual machine state at a certain point in time during const-evaluation.
struct InterpSnapshot<'mir, 'tcx> {
    stack: Vec<Frame<'mir, 'tcx>>,
    memory: CtfeMemory<'mir, 'tcx>,
}

impl InterpSnapshot<'mir, 'tcx> {
    fn capture(stack: &'a [Frame<'mir, 'tcx>], memory: &'a CtfeMemory<'mir, 'tcx>) -> Self {
        info!("snapshotting the state of the interpreter");

        // Copy all reachable allocations that exist in `memory.alloc_map` into a new `alloc_map`.
        // There's no need to copy allocations in `tcx`, since these are never mutated during
        // execution.
        let mut heap_snapshot = CtfeMemory::new(memory.tcx, ());
        for (id, _) in InterpRef::new(stack, memory).live_allocs() {
            if let Some(alloc) = memory.alloc_map.get(&id) {
                heap_snapshot.alloc_map.insert(id, alloc.clone());
                continue;
            }
        }

        InterpSnapshot {
            stack: stack.into(),
            memory: heap_snapshot,
        }
    }

    fn as_ref(&self) -> InterpRef<'_, 'mir, 'tcx> {
        InterpRef {
            stack: &self.stack,
            memory: &self.memory,
        }
    }
}

impl Eq for InterpSnapshot<'mir, 'tcx> {}
impl PartialEq for InterpSnapshot<'mir, 'tcx> {
    fn eq(&self, other: &Self) -> bool {
        self.as_ref() == other.as_ref()
    }
}

impl Hash for InterpSnapshot<'mir, 'tcx> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.as_ref().hash(state)
    }
}

/// A reference to the subset of the interpreter state that can change during execution.
///
/// Having a separate "reference" data type allows us to compare snapshots to the current
/// interpreter state without actually creating a snapshot of the current state.
//
// FIXME(ecstaticmorse): We could do away with this if we refactored the mutable parts of
// `InterpCx` into a single struct, e.g., an `InterpVm` inside of `InterpCx`. `InterpRef` would
// then become `IgnoreAllocId<&InterpVm>` and `InterpSnapshot` would be `IgnoreAllocId<InterpVm>`.
struct InterpRef<'a, 'mir, 'tcx> {
    stack: &'a [Frame<'mir, 'tcx>],
    memory: &'a CtfeMemory<'mir, 'tcx>
}

impl InterpRef<'a, 'mir, 'tcx> {
    fn new(stack: &'a [Frame<'mir, 'tcx>], memory: &'a CtfeMemory<'mir, 'tcx>) -> Self {
        InterpRef {
            stack,
            memory,
        }
    }

    /// Returns an iterator over all memory allocations reachable from anywhere on the stack.
    ///
    /// The order of iteration is predictable: Calling this method twice on a single
    /// `InterpRef` will yield the same `Allocation`s in the same order.
    fn live_allocs(&self)
        -> memory::DepthFirstSearch<'a, 'mir, 'tcx, CompileTimeInterpreter<'mir, 'tcx>>
    {
        memory::DepthFirstSearch::with_roots(self.memory, stack_roots(self.stack).into_iter())
    }
}

impl PartialEq for InterpRef<'_, 'mir, 'tcx> {
    fn eq(&self, other: &Self) -> bool {
        let is_stack_eq = self.stack.iter().map(IgnoreAllocId)
            .eq(other.stack.iter().map(IgnoreAllocId));

        if !is_stack_eq {
            return false;
        }

        // Map each `AllocId` to its DFS index as we traverse the allocation graph.
        let mut dfs_index = DfsIndexForAllocId::from_stack_roots(self.stack);
        let mut other_dfs_index = DfsIndexForAllocId::from_stack_roots(other.stack);

        for (a, b) in self.live_allocs().zip(other.live_allocs()) {
            let (a, b) = match (a.1, b.1) {
                (Ok(a), Ok(b)) => (a, b),

                // All pointers that cannot be dereferenced are considered equal. We don't
                // differentiate between null, dangling, etc.
                (Err(_), Err(_)) => continue,

                _ => return false,
            };

            // Compare the size, value, and number of pointers in both allocations, but not the
            // `AllocId`s.
            if IgnoreAllocId(a) != IgnoreAllocId(b) {
                return false;
            }

            // Check to see if each pointer in allocation `a` points to the allocation with the same
            // DFS index as the corresponding pointer in allocation `b`.
            let is_isomorphic = a.relocations().values()
                .zip(b.relocations().values())
                .all(|(&(_, pa), &(_, pb))| {
                    dfs_index.mark_visited(pa);
                    other_dfs_index.mark_visited(pb);

                    dfs_index.get(pa).unwrap() == other_dfs_index.get(pb).unwrap()
                });

            if !is_isomorphic {
                return false
            }
        }

        true
    }
}

impl Hash for InterpRef<'_, 'mir, 'tcx> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        for frame in self.stack {
            IgnoreAllocId(frame).hash(state);
        }

        // FIXME(ecstaticmorse): Incorporate the DFS index for each pointer into the hash?
        for (_, alloc) in self.live_allocs() {
            alloc.ok().map(IgnoreAllocId).hash(state);
        }
    }
}

/// A wrapper type for the various interpreter data structures that compares them without
/// considering the numeric value of `AllocId`s.
struct IgnoreAllocId<'a, T>(&'a T);

impl PartialEq for IgnoreAllocId<'_, Frame<'mir, 'tcx>> {
    fn eq(&self, IgnoreAllocId(other): &Self) -> bool {
        // This *must* remain exhaustive to ensure that all fields are considered for equality.
        let super::Frame {
            body,
            instance,
            locals,
            block,
            stmt,
            span, // Not really necessary, but cheap to compare.

            // No need to check these, since when comparing snapshots we have to check whether
            // we are at the same statement in the caller's frame.
            return_to_block: _,
            return_place: _,

            extra: (),
        } = self.0;

        std::ptr::eq::<mir::Body<'tcx>>(*body, other.body)
            && instance == &other.instance
            && block == &other.block
            && stmt == &other.stmt
            && span == &other.span
            && locals.iter().map(IgnoreAllocId)
                .eq(other.locals.iter().map(IgnoreAllocId))
    }
}

impl<'mir, 'tcx> Hash for IgnoreAllocId<'_, Frame<'mir, 'tcx>> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // See `PartialEq` impl above for description of ignored fields.
        let super::Frame {
            body,
            instance,
            locals,
            block,
            stmt,
            span,
            return_to_block: _,
            return_place: _,
            extra: (),
        } = &self.0;

        std::ptr::hash::<mir::Body<'tcx>, _>(*body, state);
        instance.hash(state);
        block.hash(state);
        stmt.hash(state);
        span.hash(state);

        for local in locals {
            IgnoreAllocId(local).hash(state);
        }
    }
}

impl PartialEq for IgnoreAllocId<'_, LocalState<'tcx>> {
    fn eq(&self, IgnoreAllocId(other): &Self) -> bool {
        // This *must* remain exhaustive to ensure that all fields are considered for equality.
        let super::LocalState {
            value,
            layout: _, // Memoized result of the layout query.
        } = self.0;

        match (value, &other.value) {
            // Compare both `Operand`s with any `AllocId`s erased.
            (LocalValue::Live(op), LocalValue::Live(other_op))
                => op.erase_alloc_id() == other_op.erase_alloc_id(),

            // Otherwise we can use the `PartialEq` impl of `LocalValue` directly.
            _ => value == &other.value,
        }
    }
}

impl Hash for IgnoreAllocId<'_, LocalState<'tcx>> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // See `PartialEq` impl above for description of ignored fields.
        let super::LocalState {
            value,
            layout: _,
        } = self.0;

        match value {
            LocalValue::Live(op) => op.erase_alloc_id().hash(state),
            LocalValue::Dead | LocalValue::Uninitialized => value.hash(state),
        }
    }
}

impl PartialEq for IgnoreAllocId<'_, Allocation> {
    fn eq(&self, IgnoreAllocId(other): &Self) -> bool {
        // `Allocation` has private fields, so we cannot exhaustively match on it. If `Allocation`
        // is updated, make sure to update the `IgnoreAllocId` impls for `PartialEq` and `Hash`
        // as well.
        let super::Allocation {
            size,
            align,
            mutability,
            extra: (),
            ..
        } = *self.0;

        let undef_mask = self.0.undef_mask();
        let relocations = self.0.relocations();
        let is_eq = align == other.align
            && size == other.size
            && mutability == other.mutability
            && undef_mask == other.undef_mask()
            && relocations.keys().eq(other.relocations().keys());

        if !is_eq {
            return false;
        }

        // We now know that the two allocations have identical sizes and undef masks. We rely
        // on this below.
        debug_assert_eq!(undef_mask, other.undef_mask());
        debug_assert_eq!(size, other.size);

        let len = self.0.len();
        let bytes = self.0.inspect_with_undef_and_ptr_outside_interpreter(0..len);
        let other_bytes = other.inspect_with_undef_and_ptr_outside_interpreter(0..len);

        // In the common case, neither allocation will have undefined bytes, and we can compare
        // the bytes directly for equality.
        if undef_mask.is_range_defined(Size::ZERO, size).is_ok() {
            return bytes == other_bytes;
        }

        // Otherwise, we only compare the bytes that are defined.
        //
        // FIXME(ecstaticmorse): This is slow. Add a method to `UndefMask` to iterate over each
        // range of defined bytes.
        (0..len)
            .filter(|&i| undef_mask.get(Size::from_bytes(u64::try_from(i).unwrap())))
            .all(|i| bytes[i] == other_bytes[i])
    }
}

impl Hash for IgnoreAllocId<'_, Allocation> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // `Allocation` has private fields, so we cannot exhaustively match on it. If `Allocation`
        // is updated, make sure to update the `IgnoreAllocId` impls for `PartialEq` and `Hash`
        // as well.
        let super::Allocation {
            size,
            align,
            mutability,
            extra: (),
            ..
        } = *self.0;

        let undef_mask = self.0.undef_mask();
        let relocations = self.0.relocations();

        size.hash(state);
        align.hash(state);
        mutability.hash(state);
        undef_mask.hash(state);

        // Hash the number of pointers in the `Allocation` as well as the offsets of those pointers
        // within the allocation.
        relocations.len().hash(state);
        relocations.keys().for_each(|pos| pos.hash(state));

        let len = self.0.len();
        let bytes = self.0.inspect_with_undef_and_ptr_outside_interpreter(0..len);

        // See `PartialEq` impl above.
        if undef_mask.is_range_defined(Size::ZERO, size).is_ok() {
            bytes.hash(state);
            return;
        }

        for i in 0..len {
            if undef_mask.get(Size::from_bytes(u64::try_from(i).unwrap())) {
                bytes[i].hash(state)
            }
        }
    }
}

/// Returns a `Vec` containing each `AllocId` that appears in a `Local` on the stack.
///
/// A single `AllocId` may appear more than once.
fn stack_roots(stack: &[Frame<'_, '_>]) -> Vec<AllocId> {
    let operands = stack
        .iter()
        .flat_map(|frame| frame.locals.iter())
        .filter_map(|local| match local.value {
            LocalValue::Live(op) => Some(op),
            LocalValue::Dead | LocalValue::Uninitialized => None,
        });

    let mut stack_roots = vec![];
    for op in operands {
        let scalars = match op {
            Operand::Immediate(Immediate::Scalar(a)) => [a.not_undef().ok(), None],
            Operand::Immediate(Immediate::ScalarPair(a, b))
                => [a.not_undef().ok(), b.not_undef().ok()],

            Operand::Indirect(MemPlace { ptr, meta, align: _ }) => [Some(ptr), meta],
        };

        for scalar in scalars.iter() {
            if let Some(ptr) = scalar.and_then(|s| s.to_ptr().ok()) {
                stack_roots.push(ptr.alloc_id);
            }
        }
    }

    stack_roots
}

/// A mapping from `AllocId`s to the order they were encountered in the DFS.
#[derive(Default)]
struct DfsIndexForAllocId {
    map: FxHashMap<AllocId, u64>,
    next_idx: u64,
}

impl DfsIndexForAllocId {
    /// Returns a mapping for all `AllocId`s present on a given stack.
    fn from_stack_roots(stack: &[Frame<'_, '_>]) -> Self {
        let mut ret = Self::default();

        for id in stack_roots(stack) {
            ret.mark_visited(id);
        }

        ret
    }

    fn mark_visited(&mut self, id: AllocId) {
        if self.map.insert(id, self.next_idx).is_none() {
            self.next_idx += 1;
        }
    }

    fn get(&self, id: AllocId) -> Option<u64> {
        self.map.get(&id).copied()
    }
}
