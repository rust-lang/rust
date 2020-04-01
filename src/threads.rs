//! Implements threads.

use std::cell::RefCell;
use std::collections::hash_map::Entry;

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

/// The state of a thread.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum ThreadState {
    /// The thread is enabled and can be executed.
    Enabled,
    /// The thread tried to join the specified thread and is blocked until that
    /// thread terminates.
    Blocked(ThreadId),
    /// The thread has terminated its execution (we do not delete terminated
    /// threads.)
    Terminated,
}

/// A thread.
pub struct Thread<'mir, 'tcx> {
    state: ThreadState,
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
        Self { state: ThreadState::Enabled, stack: Vec::new(), detached: false }
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

    /// List of threads that just terminated. TODO: Cleanup.
    terminated_threads: Vec<ThreadId>,
}

impl<'mir, 'tcx> Default for ThreadSet<'mir, 'tcx> {
    fn default() -> Self {
        let mut threads = IndexVec::new();
        threads.push(Default::default());
        Self {
            active_thread: ThreadId::new(0),
            threads: threads,
            terminated_threads: Default::default(),
        }
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
    fn set_active_thread(&mut self, id: ThreadId) -> ThreadId {
        let active_thread_id = self.active_thread;
        self.active_thread = id;
        assert!(self.active_thread.index() < self.threads.len());
        active_thread_id
    }
    /// Get the id of the currently active thread.
    fn get_active_thread(&self) -> ThreadId {
        self.active_thread
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
                .all(|thread| thread.state != ThreadState::Blocked(joined_thread_id)),
            "Bug: multiple threads try to join the same thread."
        );
        if self.threads[joined_thread_id].state != ThreadState::Terminated {
            // The joined thread is still running, we need to wait for it.
            self.threads[self.active_thread].state = ThreadState::Blocked(joined_thread_id);
            trace!(
                "{:?} blocked on {:?} when trying to join",
                self.active_thread,
                joined_thread_id
            );
        }
    }
    /// Get ids of all threads ever allocated.
    fn get_all_thread_ids(&mut self) -> Vec<ThreadId> {
        (0..self.threads.len()).map(ThreadId::new).collect()
    }
    /// Decide which thread to run next.
    ///
    /// Returns `false` if all threads terminated.
    fn schedule(&mut self) -> InterpResult<'tcx, bool> {
        if self.threads[self.active_thread].check_terminated() {
            // Check if we need to unblock any threads.
            for (i, thread) in self.threads.iter_enumerated_mut() {
                if thread.state == ThreadState::Blocked(self.active_thread) {
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
            throw_machine_stop!(TerminationInfo::Abort(Some(format!("execution deadlocked"))))
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
        Ok(this.machine.threads.set_active_thread(thread_id))
    }
    fn get_active_thread(&self) -> InterpResult<'tcx, ThreadId> {
        let this = self.eval_context_ref();
        Ok(this.machine.threads.get_active_thread())
    }
    fn active_thread_stack(&self) -> &[Frame<'mir, 'tcx, Tag, FrameData<'tcx>>] {
        let this = self.eval_context_ref();
        this.machine.threads.active_thread_stack()
    }
    fn active_thread_stack_mut(&mut self) -> &mut Vec<Frame<'mir, 'tcx, Tag, FrameData<'tcx>>> {
        let this = self.eval_context_mut();
        this.machine.threads.active_thread_stack_mut()
    }
    fn get_all_thread_ids(&mut self) -> Vec<ThreadId> {
        let this = self.eval_context_mut();
        this.machine.threads.get_all_thread_ids()
    }
    /// Decide which thread to run next.
    ///
    /// Returns `false` if all threads terminated.
    fn schedule(&mut self) -> InterpResult<'tcx, bool> {
        let this = self.eval_context_mut();
        // Find the next thread to run.
        if this.machine.threads.schedule()? {
            let active_thread = this.machine.threads.get_active_thread();
            this.memory.extra.tls.set_active_thread(active_thread);
            Ok(true)
        } else {
            Ok(false)
        }
    }
}
