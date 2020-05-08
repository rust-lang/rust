//! Implements threads.

use std::cell::RefCell;
use std::convert::TryFrom;
use std::num::{NonZeroU32, TryFromIntError};

use log::trace;

use rustc_data_structures::fx::FxHashMap;
use rustc_hir::def_id::DefId;
use rustc_index::vec::{Idx, IndexVec};
use rustc_middle::{
    middle::codegen_fn_attrs::CodegenFnAttrFlags,
    mir,
    ty::{self, Instance},
};

use crate::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SchedulingAction {
    /// Execute step on the active thread.
    ExecuteStep,
    /// Execute destructors of the active thread.
    ExecuteDtors,
    /// Stop the program.
    Stop,
}

/// A thread identifier.
#[derive(Clone, Copy, Debug, PartialOrd, Ord, PartialEq, Eq, Hash)]
pub struct ThreadId(u32);

/// The main thread. When it terminates, the whole application terminates.
const MAIN_THREAD: ThreadId = ThreadId(0);

impl ThreadId {
    pub fn to_u32(self) -> u32 {
        self.0
    }
}

impl Idx for ThreadId {
    fn new(idx: usize) -> Self {
        ThreadId(u32::try_from(idx).unwrap())
    }

    fn index(self) -> usize {
        usize::try_from(self.0).unwrap()
    }
}

impl TryFrom<u64> for ThreadId {
    type Error = TryFromIntError;
    fn try_from(id: u64) -> Result<Self, Self::Error> {
        u32::try_from(id).map(|id_u32| Self(id_u32))
    }
}

impl From<u32> for ThreadId {
    fn from(id: u32) -> Self {
        Self(id)
    }
}

impl ThreadId {
    pub fn to_u32_scalar<'tcx>(&self) -> Scalar<Tag> {
        Scalar::from_u32(u32::try_from(self.0).unwrap())
    }
}

/// An identifier of a set of blocked threads. 0 is used to indicate the absence
/// of a blockset identifier and, therefore, is not a valid identifier.
#[derive(Clone, Copy, Debug, PartialOrd, Ord, PartialEq, Eq, Hash)]
pub struct BlockSetId(NonZeroU32);

impl BlockSetId {
    /// Panics if `id` is 0.
    pub fn new(id: u32) -> Self {
        Self(NonZeroU32::new(id).expect("0 is not a valid blockset id"))
    }
    pub fn to_u32_scalar<'tcx>(&self) -> Scalar<Tag> {
        Scalar::from_u32(self.0.get())
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
    /// The thread is blocked and belongs to the given blockset.
    Blocked(BlockSetId),
    /// The thread has terminated its execution (we do not delete terminated
    /// threads).
    Terminated,
}

/// The join status of a thread.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
enum ThreadJoinStatus {
    /// The thread can be joined.
    Joinable,
    /// A thread is detached if its join handle was destroyed and no other
    /// thread can join it.
    Detached,
    /// The thread was already joined by some thread and cannot be joined again.
    Joined,
}

/// A thread.
pub struct Thread<'mir, 'tcx> {
    state: ThreadState,
    /// Name of the thread.
    thread_name: Option<Vec<u8>>,
    /// The virtual call stack.
    stack: Vec<Frame<'mir, 'tcx, Tag, FrameData<'tcx>>>,
    /// The join status.
    join_status: ThreadJoinStatus,
}

impl<'mir, 'tcx> Thread<'mir, 'tcx> {
    /// Check if the thread is done executing (no more stack frames). If yes,
    /// change the state to terminated and return `true`.
    fn check_terminated(&mut self) -> bool {
        if self.state == ThreadState::Enabled {
            if self.stack.is_empty() {
                self.state = ThreadState::Terminated;
                return true;
            }
        }
        false
    }

    /// Get the name of the current thread, or `<unnamed>` if it was not set.
    fn thread_name(&self) -> &[u8] {
        if let Some(ref thread_name) = self.thread_name {
            thread_name
        } else {
            b"<unnamed>"
        }
    }
}

impl<'mir, 'tcx> std::fmt::Debug for Thread<'mir, 'tcx> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}({:?}, {:?})", String::from_utf8_lossy(self.thread_name()), self.state, self.join_status)
    }
}

impl<'mir, 'tcx> Default for Thread<'mir, 'tcx> {
    fn default() -> Self {
        Self {
            state: ThreadState::Enabled,
            thread_name: None,
            stack: Vec::new(),
            join_status: ThreadJoinStatus::Joinable,
        }
    }
}

/// A set of threads.
#[derive(Debug)]
pub struct ThreadManager<'mir, 'tcx> {
    /// Identifier of the currently active thread.
    active_thread: ThreadId,
    /// Threads used in the program.
    ///
    /// Note that this vector also contains terminated threads.
    threads: IndexVec<ThreadId, Thread<'mir, 'tcx>>,
    /// A counter used to generate unique identifiers for blocksets.
    blockset_counter: u32,
    /// A mapping from a thread-local static to an allocation id of a thread
    /// specific allocation.
    thread_local_alloc_ids: RefCell<FxHashMap<(DefId, ThreadId), AllocId>>,
    /// A flag that indicates that we should change the active thread.
    yield_active_thread: bool,
}

impl<'mir, 'tcx> Default for ThreadManager<'mir, 'tcx> {
    fn default() -> Self {
        let mut threads = IndexVec::new();
        // Create the main thread and add it to the list of threads.
        let mut main_thread = Thread::default();
        // The main thread can *not* be joined on.
        main_thread.join_status = ThreadJoinStatus::Detached;
        threads.push(main_thread);
        Self {
            active_thread: ThreadId::new(0),
            threads: threads,
            blockset_counter: 0,
            thread_local_alloc_ids: Default::default(),
            yield_active_thread: false,
        }
    }
}

impl<'mir, 'tcx: 'mir> ThreadManager<'mir, 'tcx> {
    /// Check if we have an allocation for the given thread local static for the
    /// active thread.
    fn get_thread_local_alloc_id(&self, def_id: DefId) -> Option<AllocId> {
        self.thread_local_alloc_ids.borrow().get(&(def_id, self.active_thread)).cloned()
    }

    /// Set the allocation id as the allocation id of the given thread local
    /// static for the active thread.
    ///
    /// Panics if a thread local is initialized twice for the same thread.
    fn set_thread_local_alloc_id(&self, def_id: DefId, new_alloc_id: AllocId) {
        self.thread_local_alloc_ids
            .borrow_mut()
            .insert((def_id, self.active_thread), new_alloc_id)
            .unwrap_none();
    }

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

    /// Get the total number of threads that were ever spawn by this program.
    fn get_total_thread_count(&self) -> usize {
        self.threads.len()
    }

    /// Has the given thread terminated?
    fn has_terminated(&self, thread_id: ThreadId) -> bool {
        self.threads[thread_id].state == ThreadState::Terminated
    }

    /// Enable the thread for execution. The thread must be terminated.
    fn enable_thread(&mut self, thread_id: ThreadId) {
        assert!(self.has_terminated(thread_id));
        self.threads[thread_id].state = ThreadState::Enabled;
    }

    /// Get a mutable borrow of the currently active thread.
    fn active_thread_mut(&mut self) -> &mut Thread<'mir, 'tcx> {
        &mut self.threads[self.active_thread]
    }

    /// Get a shared borrow of the currently active thread.
    fn active_thread_ref(&self) -> &Thread<'mir, 'tcx> {
        &self.threads[self.active_thread]
    }

    /// Mark the thread as detached, which means that no other thread will try
    /// to join it and the thread is responsible for cleaning up.
    fn detach_thread(&mut self, id: ThreadId) -> InterpResult<'tcx> {
        if self.threads[id].join_status != ThreadJoinStatus::Joinable {
            throw_ub_format!("trying to detach thread that was already detached or joined");
        }
        self.threads[id].join_status = ThreadJoinStatus::Detached;
        Ok(())
    }

    /// Mark that the active thread tries to join the thread with `joined_thread_id`.
    fn join_thread(&mut self, joined_thread_id: ThreadId) -> InterpResult<'tcx> {
        if self.threads[joined_thread_id].join_status != ThreadJoinStatus::Joinable {
            throw_ub_format!("trying to join a detached or already joined thread");
        }
        if joined_thread_id == self.active_thread {
            throw_ub_format!("trying to join itself");
        }
        assert!(
            self.threads
                .iter()
                .all(|thread| thread.state != ThreadState::BlockedOnJoin(joined_thread_id)),
            "a joinable thread already has threads waiting for its termination"
        );
        // Mark the joined thread as being joined so that we detect if other
        // threads try to join it.
        self.threads[joined_thread_id].join_status = ThreadJoinStatus::Joined;
        if self.threads[joined_thread_id].state != ThreadState::Terminated {
            // The joined thread is still running, we need to wait for it.
            self.active_thread_mut().state = ThreadState::BlockedOnJoin(joined_thread_id);
            trace!(
                "{:?} blocked on {:?} when trying to join",
                self.active_thread,
                joined_thread_id
            );
        }
        Ok(())
    }

    /// Set the name of the active thread.
    fn set_thread_name(&mut self, new_thread_name: Vec<u8>) {
        self.active_thread_mut().thread_name = Some(new_thread_name);
    }

    /// Get the name of the active thread.
    fn get_thread_name(&self) -> &[u8] {
        self.active_thread_ref().thread_name()
    }

    /// Allocate a new blockset id.
    fn create_blockset(&mut self) -> BlockSetId {
        self.blockset_counter = self.blockset_counter.checked_add(1).unwrap();
        BlockSetId::new(self.blockset_counter)
    }

    /// Block the currently active thread and put it into the given blockset.
    fn block_active_thread(&mut self, set: BlockSetId) {
        let state = &mut self.active_thread_mut().state;
        assert_eq!(*state, ThreadState::Enabled);
        *state = ThreadState::Blocked(set);
    }

    /// Unblock any one thread from the given blockset if it contains at least
    /// one. Return the id of the unblocked thread.
    fn unblock_some_thread(&mut self, set: BlockSetId) -> Option<ThreadId> {
        for (id, thread) in self.threads.iter_enumerated_mut() {
            if thread.state == ThreadState::Blocked(set) {
                trace!("unblocking {:?} in blockset {:?}", id, set);
                thread.state = ThreadState::Enabled;
                return Some(id);
            }
        }
        None
    }

    /// Change the active thread to some enabled thread.
    fn yield_active_thread(&mut self) {
        self.yield_active_thread = true;
    }

    /// Decide which action to take next and on which thread.
    ///
    /// The currently implemented scheduling policy is the one that is commonly
    /// used in stateless model checkers such as Loom: run the active thread as
    /// long as we can and switch only when we have to (the active thread was
    /// blocked, terminated, or has explicitly asked to be preempted).
    fn schedule(&mut self) -> InterpResult<'tcx, SchedulingAction> {
        // Check whether the thread has **just** terminated (`check_terminated`
        // checks whether the thread has popped all its stack and if yes, sets
        // the thread state to terminated).
        if self.threads[self.active_thread].check_terminated() {
            // Check if we need to unblock any threads.
            for (i, thread) in self.threads.iter_enumerated_mut() {
                if thread.state == ThreadState::BlockedOnJoin(self.active_thread) {
                    trace!("unblocking {:?} because {:?} terminated", i, self.active_thread);
                    thread.state = ThreadState::Enabled;
                }
            }
            return Ok(SchedulingAction::ExecuteDtors);
        }
        if self.threads[MAIN_THREAD].state == ThreadState::Terminated {
            // The main thread terminated; stop the program.
            if self.threads.iter().any(|thread| thread.state != ThreadState::Terminated) {
                // FIXME: This check should be either configurable or just emit
                // a warning. For example, it seems normal for a program to
                // terminate without waiting for its detached threads to
                // terminate. However, this case is not trivial to support
                // because we also probably do not want to consider the memory
                // owned by these threads as leaked.
                throw_unsup_format!("the main thread terminated without waiting for other threads");
            }
            return Ok(SchedulingAction::Stop);
        }
        if self.threads[self.active_thread].state == ThreadState::Enabled
            && !self.yield_active_thread
        {
            // The currently active thread is still enabled, just continue with it.
            return Ok(SchedulingAction::ExecuteStep);
        }
        // We need to pick a new thread for execution.
        for (id, thread) in self.threads.iter_enumerated() {
            if thread.state == ThreadState::Enabled {
                if !self.yield_active_thread || id != self.active_thread {
                    self.active_thread = id;
                    break;
                }
            }
        }
        self.yield_active_thread = false;
        if self.threads[self.active_thread].state == ThreadState::Enabled {
            return Ok(SchedulingAction::ExecuteStep);
        }
        // We have not found a thread to execute.
        if self.threads.iter().all(|thread| thread.state == ThreadState::Terminated) {
            unreachable!();
        } else {
            throw_machine_stop!(TerminationInfo::Deadlock);
        }
    }
}

// Public interface to thread management.
impl<'mir, 'tcx: 'mir> EvalContextExt<'mir, 'tcx> for crate::MiriEvalContext<'mir, 'tcx> {}
pub trait EvalContextExt<'mir, 'tcx: 'mir>: crate::MiriEvalContextExt<'mir, 'tcx> {
    /// A workaround for thread-local statics until
    /// https://github.com/rust-lang/rust/issues/70685 is fixed: change the
    /// thread-local allocation id with a freshly generated allocation id for
    /// the currently active thread.
    fn remap_thread_local_alloc_ids(
        &self,
        val: &mut mir::interpret::ConstValue<'tcx>,
    ) -> InterpResult<'tcx> {
        let this = self.eval_context_ref();
        match *val {
            mir::interpret::ConstValue::Scalar(Scalar::Ptr(ref mut ptr)) => {
                let alloc_id = ptr.alloc_id;
                let alloc = this.tcx.get_global_alloc(alloc_id);
                let tcx = this.tcx;
                let is_thread_local = |def_id| {
                    tcx.codegen_fn_attrs(def_id).flags.contains(CodegenFnAttrFlags::THREAD_LOCAL)
                };
                match alloc {
                    Some(GlobalAlloc::Static(def_id)) if is_thread_local(def_id) => {
                        ptr.alloc_id = this.get_or_create_thread_local_alloc_id(def_id)?;
                    }
                    _ => {}
                }
            }
            _ => {
                // FIXME: Handling only `Scalar` seems to work for now, but at
                // least in principle thread-locals could be in any constant, so
                // we should also consider other cases. However, once
                // https://github.com/rust-lang/rust/issues/70685 gets fixed,
                // this code will have to be rewritten anyway.
            }
        }
        Ok(())
    }

    /// Get a thread-specific allocation id for the given thread-local static.
    /// If needed, allocate a new one.
    ///
    /// FIXME: This method should be replaced as soon as
    /// https://github.com/rust-lang/rust/issues/70685 gets fixed.
    fn get_or_create_thread_local_alloc_id(&self, def_id: DefId) -> InterpResult<'tcx, AllocId> {
        let this = self.eval_context_ref();
        let tcx = this.tcx;
        if let Some(new_alloc_id) = this.machine.threads.get_thread_local_alloc_id(def_id) {
            // We already have a thread-specific allocation id for this
            // thread-local static.
            Ok(new_alloc_id)
        } else {
            // We need to allocate a thread-specific allocation id for this
            // thread-local static.
            //
            // At first, we invoke the `const_eval_raw` query and extract the
            // allocation from it. Unfortunately, we have to duplicate the code
            // from `Memory::get_global_alloc` that does this.
            //
            // Then we store the retrieved allocation back into the `alloc_map`
            // to get a fresh allocation id, which we can use as a
            // thread-specific allocation id for the thread-local static.
            if tcx.is_foreign_item(def_id) {
                throw_unsup_format!("foreign thread-local statics are not supported");
            }
            // Invoke the `const_eval_raw` query.
            let instance = Instance::mono(tcx.tcx, def_id);
            let gid = GlobalId { instance, promoted: None };
            let raw_const =
                tcx.const_eval_raw(ty::ParamEnv::reveal_all().and(gid)).map_err(|err| {
                    // no need to report anything, the const_eval call takes care of that
                    // for statics
                    assert!(tcx.is_static(def_id));
                    err
                })?;
            let id = raw_const.alloc_id;
            // Extract the allocation from the query result.
            let allocation = tcx.global_alloc(id).unwrap_memory();
            // Create a new allocation id for the same allocation in this hacky
            // way. Internally, `alloc_map` deduplicates allocations, but this
            // is fine because Miri will make a copy before a first mutable
            // access.
            let new_alloc_id = tcx.create_memory_alloc(allocation);
            this.machine.threads.set_thread_local_alloc_id(def_id, new_alloc_id);
            Ok(new_alloc_id)
        }
    }

    #[inline]
    fn create_thread(&mut self) -> InterpResult<'tcx, ThreadId> {
        let this = self.eval_context_mut();
        Ok(this.machine.threads.create_thread())
    }

    #[inline]
    fn detach_thread(&mut self, thread_id: ThreadId) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();
        this.machine.threads.detach_thread(thread_id)
    }

    #[inline]
    fn join_thread(&mut self, joined_thread_id: ThreadId) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();
        this.machine.threads.join_thread(joined_thread_id)
    }

    #[inline]
    fn set_active_thread(&mut self, thread_id: ThreadId) -> InterpResult<'tcx, ThreadId> {
        let this = self.eval_context_mut();
        Ok(this.machine.threads.set_active_thread_id(thread_id))
    }

    #[inline]
    fn get_active_thread(&self) -> InterpResult<'tcx, ThreadId> {
        let this = self.eval_context_ref();
        Ok(this.machine.threads.get_active_thread_id())
    }

    #[inline]
    fn get_total_thread_count(&self) -> InterpResult<'tcx, usize> {
        let this = self.eval_context_ref();
        Ok(this.machine.threads.get_total_thread_count())
    }

    #[inline]
    fn has_terminated(&self, thread_id: ThreadId) -> InterpResult<'tcx, bool> {
        let this = self.eval_context_ref();
        Ok(this.machine.threads.has_terminated(thread_id))
    }

    #[inline]
    fn enable_thread(&mut self, thread_id: ThreadId) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();
        this.machine.threads.enable_thread(thread_id);
        Ok(())
    }

    #[inline]
    fn active_thread_stack(&self) -> &[Frame<'mir, 'tcx, Tag, FrameData<'tcx>>] {
        let this = self.eval_context_ref();
        this.machine.threads.active_thread_stack()
    }

    #[inline]
    fn active_thread_stack_mut(&mut self) -> &mut Vec<Frame<'mir, 'tcx, Tag, FrameData<'tcx>>> {
        let this = self.eval_context_mut();
        this.machine.threads.active_thread_stack_mut()
    }

    #[inline]
    fn set_active_thread_name(&mut self, new_thread_name: Vec<u8>) -> InterpResult<'tcx, ()> {
        let this = self.eval_context_mut();
        Ok(this.machine.threads.set_thread_name(new_thread_name))
    }

    #[inline]
    fn get_active_thread_name<'c>(&'c self) -> InterpResult<'tcx, &'c [u8]>
    where
        'mir: 'c,
    {
        let this = self.eval_context_ref();
        Ok(this.machine.threads.get_thread_name())
    }

    #[inline]
    fn create_blockset(&mut self) -> InterpResult<'tcx, BlockSetId> {
        let this = self.eval_context_mut();
        Ok(this.machine.threads.create_blockset())
    }

    #[inline]
    fn block_active_thread(&mut self, set: BlockSetId) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();
        Ok(this.machine.threads.block_active_thread(set))
    }

    #[inline]
    fn unblock_some_thread(&mut self, set: BlockSetId) -> InterpResult<'tcx, Option<ThreadId>> {
        let this = self.eval_context_mut();
        Ok(this.machine.threads.unblock_some_thread(set))
    }

    #[inline]
    fn yield_active_thread(&mut self) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();
        this.machine.threads.yield_active_thread();
        Ok(())
    }

    /// Decide which action to take next and on which thread.
    #[inline]
    fn schedule(&mut self) -> InterpResult<'tcx, SchedulingAction> {
        let this = self.eval_context_mut();
        this.machine.threads.schedule()
    }
}
