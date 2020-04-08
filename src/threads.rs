//! Implements threads.

use std::cell::RefCell;
use std::collections::hash_map::Entry;
use std::convert::TryFrom;

use log::trace;

use rustc_data_structures::fx::FxHashMap;
use rustc_index::vec::{Idx, IndexVec};
use rustc_middle::mir;
use rustc_middle::ty;

use crate::*;

/// A thread identifier.
#[derive(Clone, Copy, Debug, PartialOrd, Ord, PartialEq, Eq, Hash)]
pub struct ThreadId(usize);

impl Idx for ThreadId {
    fn new(idx: usize) -> Self {
        ThreadId(idx)
    }
    fn index(self) -> usize {
        self.0
    }
}

impl From<u64> for ThreadId {
    fn from(id: u64) -> Self {
        Self(id as usize)
    }
}

impl From<u32> for ThreadId {
    fn from(id: u32) -> Self {
        Self(id as usize)
    }
}

impl ThreadId {
    pub fn to_u32_scalar<'tcx>(&self) -> Scalar<Tag> {
        Scalar::from_u32(u32::try_from(self.0).unwrap())
    }
}

/// An identifier of a set of blocked threads.
///
/// Note: 0 is not a valid identifier.
#[derive(Clone, Copy, Debug, PartialOrd, Ord, PartialEq, Eq, Hash)]
pub struct BlockSetId(u32);

impl From<u32> for BlockSetId {
    fn from(id: u32) -> Self {
        assert_ne!(id, 0, "0 is not a valid blockset id");
        Self(id)
    }
}

impl BlockSetId {
    pub fn to_u32_scalar<'tcx>(&self) -> Scalar<Tag> {
        Scalar::from_u32(self.0)
    }
}

/// The state of a thread.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum ThreadState {
    /// The thread is enabled and can be executed.
    Enabled,
    /// The thread tried to join the specified thread and is blocked until that
    /// thread terminates.
    BlockedOnJoin(ThreadId),
    /// The thread is blocked and belongs to the given blockset..
    Blocked(BlockSetId),
    /// The thread has terminated its execution (we do not delete terminated
    /// threads.)
    Terminated,
}

/// A thread.
pub struct Thread<'mir, 'tcx> {
    state: ThreadState,
    /// Name of the thread.
    thread_name: Option<Vec<u8>>,
    /// The virtual call stack.
    stack: Vec<Frame<'mir, 'tcx, Tag, FrameData<'tcx>>>,
    /// Is the thread detached?
    ///
    /// A thread is detached if its join handle was destroyed and no other
    /// thread can join it.
    detached: bool,
}

impl<'mir, 'tcx> Thread<'mir, 'tcx> {
    /// Check if the thread terminated. If yes, change the state to terminated
    /// and return `true`.
    fn check_terminated(&mut self) -> bool {
        if self.state == ThreadState::Enabled {
            if self.stack.is_empty() {
                self.state = ThreadState::Terminated;
                return true;
            }
        }
        false
    }
}

impl<'mir, 'tcx> std::fmt::Debug for Thread<'mir, 'tcx> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.state)
    }
}

impl<'mir, 'tcx> Default for Thread<'mir, 'tcx> {
    fn default() -> Self {
        Self { state: ThreadState::Enabled, thread_name: None, stack: Vec::new(), detached: false }
    }
}

/// A set of threads.
#[derive(Debug)]
pub struct ThreadSet<'mir, 'tcx> {
    /// Identifier of the currently active thread.
    active_thread: ThreadId,
    /// Threads used in the program.
    ///
    /// Note that this vector also contains terminated threads.
    threads: IndexVec<ThreadId, Thread<'mir, 'tcx>>,
    /// A counter used to generate unique identifiers for blocksets.
    blockset_counter: u32,
}

impl<'mir, 'tcx> Default for ThreadSet<'mir, 'tcx> {
    fn default() -> Self {
        let mut threads = IndexVec::new();
        threads.push(Default::default());
        Self { active_thread: ThreadId::new(0), threads: threads, blockset_counter: 0 }
    }
}

impl<'mir, 'tcx: 'mir> ThreadSet<'mir, 'tcx> {
    /// Borrow the stack of the active thread.
    fn active_thread_stack(&self) -> &[Frame<'mir, 'tcx, Tag, FrameData<'tcx>>] {
        &self.threads[self.active_thread].stack
    }
    /// Mutably borrow the stack of the active thread.
    fn active_thread_stack_mut(&mut self) -> &mut Vec<Frame<'mir, 'tcx, Tag, FrameData<'tcx>>> {
        &mut self.threads[self.active_thread].stack
    }
    /// Create a new thread and returns its id.
    fn create_thread(&mut self) -> ThreadId {
        let new_thread_id = ThreadId::new(self.threads.len());
        self.threads.push(Default::default());
        new_thread_id
    }
    /// Set an active thread and return the id of the thread that was active before.
    fn set_active_thread_id(&mut self, id: ThreadId) -> ThreadId {
        let active_thread_id = self.active_thread;
        self.active_thread = id;
        assert!(self.active_thread.index() < self.threads.len());
        active_thread_id
    }
    /// Get the id of the currently active thread.
    fn get_active_thread_id(&self) -> ThreadId {
        self.active_thread
    }
    /// Get the borrow of the currently active thread.
    fn active_thread_mut(&mut self) -> &mut Thread<'mir, 'tcx> {
        &mut self.threads[self.active_thread]
    }
    /// Mark the thread as detached, which means that no other thread will try
    /// to join it and the thread is responsible for cleaning up.
    fn detach_thread(&mut self, id: ThreadId) {
        self.threads[id].detached = true;
    }
    /// Mark that the active thread tries to join the thread with `joined_thread_id`.
    fn join_thread(&mut self, joined_thread_id: ThreadId) {
        assert!(!self.threads[joined_thread_id].detached, "Bug: trying to join a detached thread.");
        assert_ne!(joined_thread_id, self.active_thread, "Bug: trying to join itself");
        assert!(
            self.threads
                .iter()
                .all(|thread| thread.state != ThreadState::BlockedOnJoin(joined_thread_id)),
            "Bug: multiple threads try to join the same thread."
        );
        if self.threads[joined_thread_id].state != ThreadState::Terminated {
            // The joined thread is still running, we need to wait for it.
            self.active_thread_mut().state = ThreadState::BlockedOnJoin(joined_thread_id);
            trace!(
                "{:?} blocked on {:?} when trying to join",
                self.active_thread,
                joined_thread_id
            );
        }
    }
    /// Set the name of the active thread.
    fn set_thread_name(&mut self, new_thread_name: Vec<u8>) {
        self.active_thread_mut().thread_name = Some(new_thread_name);
    }
    /// Get ids and states of all threads ever allocated.
    fn get_all_thread_ids_with_states(&self) -> Vec<(ThreadId, ThreadState)> {
        self.threads.iter_enumerated().map(|(id, thread)| (id, thread.state)).collect()
    }
    fn create_blockset(&mut self) -> BlockSetId {
        self.blockset_counter = self.blockset_counter.checked_add(1).unwrap();
        self.blockset_counter.into()
    }
    fn block_active_thread(&mut self, set: BlockSetId) {
        let state = &mut self.active_thread_mut().state;
        assert_eq!(*state, ThreadState::Enabled);
        *state = ThreadState::Blocked(set);
    }
    fn unblock_random_thread(&mut self, set: BlockSetId) -> Option<ThreadId> {
        for (id, thread) in self.threads.iter_enumerated_mut() {
            if thread.state == ThreadState::Blocked(set) {
                trace!("unblocking {:?} in blockset {:?}", id, set);
                thread.state = ThreadState::Enabled;
                return Some(id);
            }
        }
        None
    }
    /// Decide which thread to run next.
    ///
    /// Returns `false` if all threads terminated.
    fn schedule(&mut self) -> InterpResult<'tcx, bool> {
        if self.threads[self.active_thread].check_terminated() {
            // Check if we need to unblock any threads.
            for (i, thread) in self.threads.iter_enumerated_mut() {
                if thread.state == ThreadState::BlockedOnJoin(self.active_thread) {
                    trace!("unblocking {:?} because {:?} terminated", i, self.active_thread);
                    thread.state = ThreadState::Enabled;
                }
            }
        }
        if self.threads[self.active_thread].state == ThreadState::Enabled {
            return Ok(true);
        }
        if let Some(enabled_thread) =
            self.threads.iter().position(|thread| thread.state == ThreadState::Enabled)
        {
            self.active_thread = ThreadId::new(enabled_thread);
            return Ok(true);
        }
        if self.threads.iter().all(|thread| thread.state == ThreadState::Terminated) {
            Ok(false)
        } else {
            throw_machine_stop!(TerminationInfo::Deadlock);
        }
    }
}

/// In Rust, a thread local variable is just a specially marked static. To
/// ensure a property that each memory allocation has a globally unique
/// allocation identifier, we create a fresh allocation id for each thread. This
/// data structure keeps the track of the created allocation identifiers and
/// their relation to the original static allocations.
#[derive(Clone, Debug, Default)]
pub struct ThreadLocalStorage {
    /// A map from a thread local allocation identifier to the static from which
    /// it was created.
    thread_local_origin: RefCell<FxHashMap<AllocId, AllocId>>,
    /// A map from a thread local static and thread id to the unique thread
    /// local allocation.
    thread_local_allocations: RefCell<FxHashMap<(AllocId, ThreadId), AllocId>>,
    /// The currently active thread.
    active_thread: Option<ThreadId>,
}

impl ThreadLocalStorage {
    /// For static allocation identifier `original_id` get a thread local
    /// allocation identifier. If it is not allocated yet, allocate.
    pub fn get_or_register_allocation(&self, tcx: ty::TyCtxt<'_>, original_id: AllocId) -> AllocId {
        match self
            .thread_local_allocations
            .borrow_mut()
            .entry((original_id, self.active_thread.unwrap()))
        {
            Entry::Occupied(entry) => *entry.get(),
            Entry::Vacant(entry) => {
                let fresh_id = tcx.alloc_map.lock().reserve();
                entry.insert(fresh_id);
                self.thread_local_origin.borrow_mut().insert(fresh_id, original_id);
                trace!(
                    "get_or_register_allocation(original_id={:?}) -> {:?}",
                    original_id,
                    fresh_id
                );
                fresh_id
            }
        }
    }
    /// For thread local allocation identifier `alloc_id`, retrieve the original
    /// static allocation identifier from which it was created.
    pub fn resolve_allocation<'tcx>(
        &self,
        tcx: ty::TyCtxt<'tcx>,
        alloc_id: AllocId,
    ) -> Option<mir::interpret::GlobalAlloc<'tcx>> {
        trace!("resolve_allocation(alloc_id: {:?})", alloc_id);
        if let Some(original_id) = self.thread_local_origin.borrow().get(&alloc_id) {
            trace!("resolve_allocation(alloc_id: {:?}) -> {:?}", alloc_id, original_id);
            tcx.alloc_map.lock().get(*original_id)
        } else {
            tcx.alloc_map.lock().get(alloc_id)
        }
    }
    /// Set which thread is currently active.
    fn set_active_thread(&mut self, active_thread: ThreadId) {
        self.active_thread = Some(active_thread);
    }
}

impl<'mir, 'tcx: 'mir> EvalContextExt<'mir, 'tcx> for crate::MiriEvalContext<'mir, 'tcx> {}
pub trait EvalContextExt<'mir, 'tcx: 'mir>: crate::MiriEvalContextExt<'mir, 'tcx> {
    fn create_thread(&mut self) -> InterpResult<'tcx, ThreadId> {
        let this = self.eval_context_mut();
        Ok(this.machine.threads.create_thread())
    }
    fn detach_thread(&mut self, thread_id: ThreadId) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();
        this.machine.threads.detach_thread(thread_id);
        Ok(())
    }
    fn join_thread(&mut self, joined_thread_id: ThreadId) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();
        this.machine.threads.join_thread(joined_thread_id);
        Ok(())
    }
    fn set_active_thread(&mut self, thread_id: ThreadId) -> InterpResult<'tcx, ThreadId> {
        let this = self.eval_context_mut();
        this.memory.extra.tls.set_active_thread(thread_id);
        Ok(this.machine.threads.set_active_thread_id(thread_id))
    }
    fn get_active_thread(&self) -> InterpResult<'tcx, ThreadId> {
        let this = self.eval_context_ref();
        Ok(this.machine.threads.get_active_thread_id())
    }
    fn active_thread_stack(&self) -> &[Frame<'mir, 'tcx, Tag, FrameData<'tcx>>] {
        let this = self.eval_context_ref();
        this.machine.threads.active_thread_stack()
    }
    fn active_thread_stack_mut(&mut self) -> &mut Vec<Frame<'mir, 'tcx, Tag, FrameData<'tcx>>> {
        let this = self.eval_context_mut();
        this.machine.threads.active_thread_stack_mut()
    }
    fn set_active_thread_name(&mut self, new_thread_name: Vec<u8>) -> InterpResult<'tcx, ()> {
        let this = self.eval_context_mut();
        Ok(this.machine.threads.set_thread_name(new_thread_name))
    }
    fn get_all_thread_ids_with_states(&mut self) -> Vec<(ThreadId, ThreadState)> {
        let this = self.eval_context_mut();
        this.machine.threads.get_all_thread_ids_with_states()
    }
    fn create_blockset(&mut self) -> InterpResult<'tcx, BlockSetId> {
        let this = self.eval_context_mut();
        Ok(this.machine.threads.create_blockset())
    }
    fn block_active_thread(&mut self, set: BlockSetId) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();
        Ok(this.machine.threads.block_active_thread(set))
    }
    fn unblock_random_thread(&mut self, set: BlockSetId) -> InterpResult<'tcx, Option<ThreadId>> {
        let this = self.eval_context_mut();
        Ok(this.machine.threads.unblock_random_thread(set))
    }
    /// Decide which thread to run next.
    ///
    /// Returns `false` if all threads terminated.
    fn schedule(&mut self) -> InterpResult<'tcx, bool> {
        let this = self.eval_context_mut();
        // Find the next thread to run.
        if this.machine.threads.schedule()? {
            let active_thread = this.machine.threads.get_active_thread_id();
            this.memory.extra.tls.set_active_thread(active_thread);
            Ok(true)
        } else {
            Ok(false)
        }
    }
}
