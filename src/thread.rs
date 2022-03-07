//! Implements threads.

use std::cell::RefCell;
use std::collections::hash_map::Entry;
use std::convert::TryFrom;
use std::num::TryFromIntError;
use std::time::{Duration, Instant, SystemTime};

use log::trace;

use rustc_data_structures::fx::FxHashMap;
use rustc_hir::def_id::DefId;
use rustc_index::vec::{Idx, IndexVec};

use crate::sync::SynchronizationState;
use crate::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SchedulingAction {
    /// Execute step on the active thread.
    ExecuteStep,
    /// Execute a timeout callback.
    ExecuteTimeoutCallback,
    /// Execute destructors of the active thread.
    ExecuteDtors,
    /// Stop the program.
    Stop,
}

/// Timeout callbacks can be created by synchronization primitives to tell the
/// scheduler that they should be called once some period of time passes.
type TimeoutCallback<'mir, 'tcx> =
    Box<dyn FnOnce(&mut InterpCx<'mir, 'tcx, Evaluator<'mir, 'tcx>>) -> InterpResult<'tcx> + 'tcx>;

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

/// The state of a thread.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum ThreadState {
    /// The thread is enabled and can be executed.
    Enabled,
    /// The thread tried to join the specified thread and is blocked until that
    /// thread terminates.
    BlockedOnJoin(ThreadId),
    /// The thread is blocked on some synchronization primitive. It is the
    /// responsibility of the synchronization primitives to track threads that
    /// are blocked by them.
    BlockedOnSync,
    /// The thread has terminated its execution. We do not delete terminated
    /// threads (FIXME: why?).
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

    /// The temporary used for storing the argument of
    /// the call to `miri_start_panic` (the panic payload) when unwinding.
    /// This is pointer-sized, and matches the `Payload` type in `src/libpanic_unwind/miri.rs`.
    pub(crate) panic_payload: Option<Scalar<Tag>>,

    /// Last OS error location in memory. It is a 32-bit integer.
    pub(crate) last_error: Option<MPlaceTy<'tcx, Tag>>,
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
        if let Some(ref thread_name) = self.thread_name { thread_name } else { b"<unnamed>" }
    }
}

impl<'mir, 'tcx> std::fmt::Debug for Thread<'mir, 'tcx> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}({:?}, {:?})",
            String::from_utf8_lossy(self.thread_name()),
            self.state,
            self.join_status
        )
    }
}

impl<'mir, 'tcx> Default for Thread<'mir, 'tcx> {
    fn default() -> Self {
        Self {
            state: ThreadState::Enabled,
            thread_name: None,
            stack: Vec::new(),
            join_status: ThreadJoinStatus::Joinable,
            panic_payload: None,
            last_error: None,
        }
    }
}

/// A specific moment in time.
#[derive(Debug)]
pub enum Time {
    Monotonic(Instant),
    RealTime(SystemTime),
}

impl Time {
    /// How long do we have to wait from now until the specified time?
    fn get_wait_time(&self) -> Duration {
        match self {
            Time::Monotonic(instant) => instant.saturating_duration_since(Instant::now()),
            Time::RealTime(time) =>
                time.duration_since(SystemTime::now()).unwrap_or(Duration::new(0, 0)),
        }
    }
}

/// Callbacks are used to implement timeouts. For example, waiting on a
/// conditional variable with a timeout creates a callback that is called after
/// the specified time and unblocks the thread. If another thread signals on the
/// conditional variable, the signal handler deletes the callback.
struct TimeoutCallbackInfo<'mir, 'tcx> {
    /// The callback should be called no earlier than this time.
    call_time: Time,
    /// The called function.
    callback: TimeoutCallback<'mir, 'tcx>,
}

impl<'mir, 'tcx> std::fmt::Debug for TimeoutCallbackInfo<'mir, 'tcx> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "TimeoutCallback({:?})", self.call_time)
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
    /// This field is pub(crate) because the synchronization primitives
    /// (`crate::sync`) need a way to access it.
    pub(crate) sync: SynchronizationState,
    /// A mapping from a thread-local static to an allocation id of a thread
    /// specific allocation.
    thread_local_alloc_ids: RefCell<FxHashMap<(DefId, ThreadId), Pointer<Tag>>>,
    /// A flag that indicates that we should change the active thread.
    yield_active_thread: bool,
    /// Callbacks that are called once the specified time passes.
    timeout_callbacks: FxHashMap<ThreadId, TimeoutCallbackInfo<'mir, 'tcx>>,
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
            sync: SynchronizationState::default(),
            thread_local_alloc_ids: Default::default(),
            yield_active_thread: false,
            timeout_callbacks: FxHashMap::default(),
        }
    }
}

impl<'mir, 'tcx: 'mir> ThreadManager<'mir, 'tcx> {
    /// Check if we have an allocation for the given thread local static for the
    /// active thread.
    fn get_thread_local_alloc_id(&self, def_id: DefId) -> Option<Pointer<Tag>> {
        self.thread_local_alloc_ids.borrow().get(&(def_id, self.active_thread)).cloned()
    }

    /// Set the pointer for the allocation of the given thread local
    /// static for the active thread.
    ///
    /// Panics if a thread local is initialized twice for the same thread.
    fn set_thread_local_alloc(&self, def_id: DefId, ptr: Pointer<Tag>) {
        self.thread_local_alloc_ids
            .borrow_mut()
            .try_insert((def_id, self.active_thread), ptr)
            .unwrap();
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

    /// Have all threads terminated?
    fn have_all_terminated(&self) -> bool {
        self.threads.iter().all(|thread| thread.state == ThreadState::Terminated)
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
    fn join_thread(
        &mut self,
        joined_thread_id: ThreadId,
        data_race: Option<&mut data_race::GlobalState>,
    ) -> InterpResult<'tcx> {
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
        } else {
            // The thread has already terminated - mark join happens-before
            if let Some(data_race) = data_race {
                data_race.thread_joined(self.active_thread, joined_thread_id);
            }
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

    /// Put the thread into the blocked state.
    fn block_thread(&mut self, thread: ThreadId) {
        let state = &mut self.threads[thread].state;
        assert_eq!(*state, ThreadState::Enabled);
        *state = ThreadState::BlockedOnSync;
    }

    /// Put the blocked thread into the enabled state.
    fn unblock_thread(&mut self, thread: ThreadId) {
        let state = &mut self.threads[thread].state;
        assert_eq!(*state, ThreadState::BlockedOnSync);
        *state = ThreadState::Enabled;
    }

    /// Change the active thread to some enabled thread.
    fn yield_active_thread(&mut self) {
        // We do not yield immediately, as swapping out the current stack while executing a MIR statement
        // could lead to all sorts of confusion.
        // We should only switch stacks between steps.
        self.yield_active_thread = true;
    }

    /// Register the given `callback` to be called once the `call_time` passes.
    ///
    /// The callback will be called with `thread` being the active thread, and
    /// the callback may not change the active thread.
    fn register_timeout_callback(
        &mut self,
        thread: ThreadId,
        call_time: Time,
        callback: TimeoutCallback<'mir, 'tcx>,
    ) {
        self.timeout_callbacks
            .try_insert(thread, TimeoutCallbackInfo { call_time, callback })
            .unwrap();
    }

    /// Unregister the callback for the `thread`.
    fn unregister_timeout_callback_if_exists(&mut self, thread: ThreadId) {
        self.timeout_callbacks.remove(&thread);
    }

    /// Get a callback that is ready to be called.
    fn get_ready_callback(&mut self) -> Option<(ThreadId, TimeoutCallback<'mir, 'tcx>)> {
        // We iterate over all threads in the order of their indices because
        // this allows us to have a deterministic scheduler.
        for thread in self.threads.indices() {
            match self.timeout_callbacks.entry(thread) {
                Entry::Occupied(entry) =>
                    if entry.get().call_time.get_wait_time() == Duration::new(0, 0) {
                        return Some((thread, entry.remove().callback));
                    },
                Entry::Vacant(_) => {}
            }
        }
        None
    }

    /// Wakes up threads joining on the active one and deallocates thread-local statics.
    /// The `AllocId` that can now be freed are returned.
    fn thread_terminated(
        &mut self,
        mut data_race: Option<&mut data_race::GlobalState>,
    ) -> Vec<Pointer<Tag>> {
        let mut free_tls_statics = Vec::new();
        {
            let mut thread_local_statics = self.thread_local_alloc_ids.borrow_mut();
            thread_local_statics.retain(|&(_def_id, thread), &mut alloc_id| {
                if thread != self.active_thread {
                    // Keep this static around.
                    return true;
                }
                // Delete this static from the map and from memory.
                // We cannot free directly here as we cannot use `?` in this context.
                free_tls_statics.push(alloc_id);
                return false;
            });
        }
        // Set the thread into a terminated state in the data-race detector
        if let Some(ref mut data_race) = data_race {
            data_race.thread_terminated();
        }
        // Check if we need to unblock any threads.
        for (i, thread) in self.threads.iter_enumerated_mut() {
            if thread.state == ThreadState::BlockedOnJoin(self.active_thread) {
                // The thread has terminated, mark happens-before edge to joining thread
                if let Some(ref mut data_race) = data_race {
                    data_race.thread_joined(i, self.active_thread);
                }
                trace!("unblocking {:?} because {:?} terminated", i, self.active_thread);
                thread.state = ThreadState::Enabled;
            }
        }
        return free_tls_statics;
    }

    /// Decide which action to take next and on which thread.
    ///
    /// The currently implemented scheduling policy is the one that is commonly
    /// used in stateless model checkers such as Loom: run the active thread as
    /// long as we can and switch only when we have to (the active thread was
    /// blocked, terminated, or has explicitly asked to be preempted).
    fn schedule(
        &mut self,
        data_race: &Option<data_race::GlobalState>,
    ) -> InterpResult<'tcx, SchedulingAction> {
        // Check whether the thread has **just** terminated (`check_terminated`
        // checks whether the thread has popped all its stack and if yes, sets
        // the thread state to terminated).
        if self.threads[self.active_thread].check_terminated() {
            return Ok(SchedulingAction::ExecuteDtors);
        }
        // If we get here again and the thread is *still* terminated, there are no more dtors to run.
        if self.threads[MAIN_THREAD].state == ThreadState::Terminated {
            // The main thread terminated; stop the program.
            // We do *not* run TLS dtors of remaining threads, which seems to match rustc behavior.
            return Ok(SchedulingAction::Stop);
        }
        // This thread and the program can keep going.
        if self.threads[self.active_thread].state == ThreadState::Enabled
            && !self.yield_active_thread
        {
            // The currently active thread is still enabled, just continue with it.
            return Ok(SchedulingAction::ExecuteStep);
        }
        // The active thread yielded. Let's see if there are any timeouts to take care of. We do
        // this *before* running any other thread, to ensure that timeouts "in the past" fire before
        // any other thread can take an action. This ensures that for `pthread_cond_timedwait`, "an
        // error is returned if [...] the absolute time specified by abstime has already been passed
        // at the time of the call".
        // <https://pubs.opengroup.org/onlinepubs/9699919799/functions/pthread_cond_timedwait.html>
        let potential_sleep_time =
            self.timeout_callbacks.values().map(|info| info.call_time.get_wait_time()).min();
        if potential_sleep_time == Some(Duration::new(0, 0)) {
            return Ok(SchedulingAction::ExecuteTimeoutCallback);
        }
        // No callbacks scheduled, pick a regular thread to execute.
        // We need to pick a new thread for execution.
        for (id, thread) in self.threads.iter_enumerated() {
            if thread.state == ThreadState::Enabled {
                if !self.yield_active_thread || id != self.active_thread {
                    self.active_thread = id;
                    if let Some(data_race) = data_race {
                        data_race.thread_set_active(self.active_thread);
                    }
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
            unreachable!("all threads terminated without the main thread terminating?!");
        } else if let Some(sleep_time) = potential_sleep_time {
            // All threads are currently blocked, but we have unexecuted
            // timeout_callbacks, which may unblock some of the threads. Hence,
            // sleep until the first callback.
            std::thread::sleep(sleep_time);
            Ok(SchedulingAction::ExecuteTimeoutCallback)
        } else {
            throw_machine_stop!(TerminationInfo::Deadlock);
        }
    }
}

// Public interface to thread management.
impl<'mir, 'tcx: 'mir> EvalContextExt<'mir, 'tcx> for crate::MiriEvalContext<'mir, 'tcx> {}
pub trait EvalContextExt<'mir, 'tcx: 'mir>: crate::MiriEvalContextExt<'mir, 'tcx> {
    /// Get a thread-specific allocation id for the given thread-local static.
    /// If needed, allocate a new one.
    fn get_or_create_thread_local_alloc(
        &mut self,
        def_id: DefId,
    ) -> InterpResult<'tcx, Pointer<Tag>> {
        let this = self.eval_context_mut();
        let tcx = this.tcx;
        if let Some(old_alloc) = this.machine.threads.get_thread_local_alloc_id(def_id) {
            // We already have a thread-specific allocation id for this
            // thread-local static.
            Ok(old_alloc)
        } else {
            // We need to allocate a thread-specific allocation id for this
            // thread-local static.
            // First, we compute the initial value for this static.
            if tcx.is_foreign_item(def_id) {
                throw_unsup_format!("foreign thread-local statics are not supported");
            }
            let allocation = tcx.eval_static_initializer(def_id)?;
            // Create a fresh allocation with this content.
            let new_alloc =
                this.memory.allocate_with(allocation.inner().clone(), MiriMemoryKind::Tls.into());
            this.machine.threads.set_thread_local_alloc(def_id, new_alloc);
            Ok(new_alloc)
        }
    }

    #[inline]
    fn create_thread(&mut self) -> ThreadId {
        let this = self.eval_context_mut();
        let id = this.machine.threads.create_thread();
        if let Some(data_race) = &mut this.memory.extra.data_race {
            data_race.thread_created(id);
        }
        id
    }

    #[inline]
    fn detach_thread(&mut self, thread_id: ThreadId) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();
        this.machine.threads.detach_thread(thread_id)
    }

    #[inline]
    fn join_thread(&mut self, joined_thread_id: ThreadId) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();
        this.machine.threads.join_thread(joined_thread_id, this.memory.extra.data_race.as_mut())?;
        Ok(())
    }

    #[inline]
    fn set_active_thread(&mut self, thread_id: ThreadId) -> ThreadId {
        let this = self.eval_context_mut();
        if let Some(data_race) = &this.memory.extra.data_race {
            data_race.thread_set_active(thread_id);
        }
        this.machine.threads.set_active_thread_id(thread_id)
    }

    #[inline]
    fn get_active_thread(&self) -> ThreadId {
        let this = self.eval_context_ref();
        this.machine.threads.get_active_thread_id()
    }

    #[inline]
    fn active_thread_mut(&mut self) -> &mut Thread<'mir, 'tcx> {
        let this = self.eval_context_mut();
        this.machine.threads.active_thread_mut()
    }

    #[inline]
    fn active_thread_ref(&self) -> &Thread<'mir, 'tcx> {
        let this = self.eval_context_ref();
        this.machine.threads.active_thread_ref()
    }

    #[inline]
    fn get_total_thread_count(&self) -> usize {
        let this = self.eval_context_ref();
        this.machine.threads.get_total_thread_count()
    }

    #[inline]
    fn has_terminated(&self, thread_id: ThreadId) -> bool {
        let this = self.eval_context_ref();
        this.machine.threads.has_terminated(thread_id)
    }

    #[inline]
    fn have_all_terminated(&self) -> bool {
        let this = self.eval_context_ref();
        this.machine.threads.have_all_terminated()
    }

    #[inline]
    fn enable_thread(&mut self, thread_id: ThreadId) {
        let this = self.eval_context_mut();
        this.machine.threads.enable_thread(thread_id);
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
    fn set_active_thread_name(&mut self, new_thread_name: Vec<u8>) {
        let this = self.eval_context_mut();
        if let Some(data_race) = &mut this.memory.extra.data_race {
            if let Ok(string) = String::from_utf8(new_thread_name.clone()) {
                data_race.thread_set_name(this.machine.threads.active_thread, string);
            }
        }
        this.machine.threads.set_thread_name(new_thread_name);
    }

    #[inline]
    fn get_active_thread_name<'c>(&'c self) -> &'c [u8]
    where
        'mir: 'c,
    {
        let this = self.eval_context_ref();
        this.machine.threads.get_thread_name()
    }

    #[inline]
    fn block_thread(&mut self, thread: ThreadId) {
        let this = self.eval_context_mut();
        this.machine.threads.block_thread(thread);
    }

    #[inline]
    fn unblock_thread(&mut self, thread: ThreadId) {
        let this = self.eval_context_mut();
        this.machine.threads.unblock_thread(thread);
    }

    #[inline]
    fn yield_active_thread(&mut self) {
        let this = self.eval_context_mut();
        this.machine.threads.yield_active_thread();
    }

    #[inline]
    fn register_timeout_callback(
        &mut self,
        thread: ThreadId,
        call_time: Time,
        callback: TimeoutCallback<'mir, 'tcx>,
    ) {
        let this = self.eval_context_mut();
        this.machine.threads.register_timeout_callback(thread, call_time, callback);
    }

    #[inline]
    fn unregister_timeout_callback_if_exists(&mut self, thread: ThreadId) {
        let this = self.eval_context_mut();
        this.machine.threads.unregister_timeout_callback_if_exists(thread);
    }

    /// Execute a timeout callback on the callback's thread.
    #[inline]
    fn run_timeout_callback(&mut self) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();
        let (thread, callback) =
            if let Some((thread, callback)) = this.machine.threads.get_ready_callback() {
                (thread, callback)
            } else {
                // get_ready_callback can return None if the computer's clock
                // was shifted after calling the scheduler and before the call
                // to get_ready_callback (see issue
                // https://github.com/rust-lang/miri/issues/1763). In this case,
                // just do nothing, which effectively just returns to the
                // scheduler.
                return Ok(());
            };
        // This back-and-forth with `set_active_thread` is here because of two
        // design decisions:
        // 1. Make the caller and not the callback responsible for changing
        //    thread.
        // 2. Make the scheduler the only place that can change the active
        //    thread.
        let old_thread = this.set_active_thread(thread);
        callback(this)?;
        this.set_active_thread(old_thread);
        Ok(())
    }

    /// Decide which action to take next and on which thread.
    #[inline]
    fn schedule(&mut self) -> InterpResult<'tcx, SchedulingAction> {
        let this = self.eval_context_mut();
        let data_race = &this.memory.extra.data_race;
        this.machine.threads.schedule(data_race)
    }

    /// Handles thread termination of the active thread: wakes up threads joining on this one,
    /// and deallocated thread-local statics.
    ///
    /// This is called from `tls.rs` after handling the TLS dtors.
    #[inline]
    fn thread_terminated(&mut self) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();
        for ptr in this.machine.threads.thread_terminated(this.memory.extra.data_race.as_mut()) {
            this.memory.deallocate(ptr.into(), None, MiriMemoryKind::Tls.into())?;
        }
        Ok(())
    }
}
