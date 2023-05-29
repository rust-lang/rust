//! Implements threads.

use std::cell::RefCell;
use std::collections::hash_map::Entry;
use std::num::TryFromIntError;
use std::sync::atomic::{AtomicBool, Ordering::Relaxed};
use std::task::Poll;
use std::time::{Duration, SystemTime};

use log::trace;

use rustc_data_structures::fx::FxHashMap;
use rustc_hir::def_id::DefId;
use rustc_index::{Idx, IndexVec};
use rustc_middle::mir::Mutability;
use rustc_middle::ty::layout::TyAndLayout;
use rustc_span::Span;
use rustc_target::spec::abi::Abi;

use crate::concurrency::data_race;
use crate::concurrency::sync::SynchronizationState;
use crate::shims::tls;
use crate::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SchedulingAction {
    /// Execute step on the active thread.
    ExecuteStep,
    /// Execute a timeout callback.
    ExecuteTimeoutCallback,
    /// Wait for a bit, until there is a timeout to be called.
    Sleep(Duration),
}

/// Trait for callbacks that can be executed when some event happens, such as after a timeout.
pub trait MachineCallback<'mir, 'tcx>: VisitTags {
    fn call(&self, ecx: &mut InterpCx<'mir, 'tcx, MiriMachine<'mir, 'tcx>>) -> InterpResult<'tcx>;
}

type TimeoutCallback<'mir, 'tcx> = Box<dyn MachineCallback<'mir, 'tcx> + 'tcx>;

/// A thread identifier.
#[derive(Clone, Copy, Debug, PartialOrd, Ord, PartialEq, Eq, Hash)]
pub struct ThreadId(u32);

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
        u32::try_from(id).map(Self)
    }
}

impl From<u32> for ThreadId {
    fn from(id: u32) -> Self {
        Self(id)
    }
}

impl From<ThreadId> for u64 {
    fn from(t: ThreadId) -> Self {
        t.0.into()
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
    stack: Vec<Frame<'mir, 'tcx, Provenance, FrameExtra<'tcx>>>,

    /// The function to call when the stack ran empty, to figure out what to do next.
    /// Conceptually, this is the interpreter implementation of the things that happen 'after' the
    /// Rust language entry point for this thread returns (usually implemented by the C or OS runtime).
    /// (`None` is an error, it means the callback has not been set up yet or is actively running.)
    pub(crate) on_stack_empty: Option<StackEmptyCallback<'mir, 'tcx>>,

    /// The index of the topmost user-relevant frame in `stack`. This field must contain
    /// the value produced by `get_top_user_relevant_frame`.
    /// The `None` state here represents
    /// This field is a cache to reduce how often we call that method. The cache is manually
    /// maintained inside `MiriMachine::after_stack_push` and `MiriMachine::after_stack_pop`.
    top_user_relevant_frame: Option<usize>,

    /// The join status.
    join_status: ThreadJoinStatus,

    /// Stack of active panic payloads for the current thread. Used for storing
    /// the argument of the call to `miri_start_panic` (the panic payload) when unwinding.
    /// This is pointer-sized, and matches the `Payload` type in `src/libpanic_unwind/miri.rs`.
    ///
    /// In real unwinding, the payload gets passed as an argument to the landing pad,
    /// which then forwards it to 'Resume'. However this argument is implicit in MIR,
    /// so we have to store it out-of-band. When there are multiple active unwinds,
    /// the innermost one is always caught first, so we can store them as a stack.
    pub(crate) panic_payloads: Vec<Scalar<Provenance>>,

    /// Last OS error location in memory. It is a 32-bit integer.
    pub(crate) last_error: Option<MPlaceTy<'tcx, Provenance>>,
}

pub type StackEmptyCallback<'mir, 'tcx> =
    Box<dyn FnMut(&mut MiriInterpCx<'mir, 'tcx>) -> InterpResult<'tcx, Poll<()>>>;

impl<'mir, 'tcx> Thread<'mir, 'tcx> {
    /// Get the name of the current thread, or `<unnamed>` if it was not set.
    fn thread_name(&self) -> &[u8] {
        if let Some(ref thread_name) = self.thread_name { thread_name } else { b"<unnamed>" }
    }

    /// Return the top user-relevant frame, if there is one.
    /// Note that the choice to return `None` here when there is no user-relevant frame is part of
    /// justifying the optimization that only pushes of user-relevant frames require updating the
    /// `top_user_relevant_frame` field.
    fn compute_top_user_relevant_frame(&self) -> Option<usize> {
        self.stack
            .iter()
            .enumerate()
            .rev()
            .find_map(|(idx, frame)| if frame.extra.is_user_relevant { Some(idx) } else { None })
    }

    /// Re-compute the top user-relevant frame from scratch.
    pub fn recompute_top_user_relevant_frame(&mut self) {
        self.top_user_relevant_frame = self.compute_top_user_relevant_frame();
    }

    /// Set the top user-relevant frame to the given value. Must be equal to what
    /// `get_top_user_relevant_frame` would return!
    pub fn set_top_user_relevant_frame(&mut self, frame_idx: usize) {
        debug_assert_eq!(Some(frame_idx), self.compute_top_user_relevant_frame());
        self.top_user_relevant_frame = Some(frame_idx);
    }

    /// Returns the topmost frame that is considered user-relevant, or the
    /// top of the stack if there is no such frame, or `None` if the stack is empty.
    pub fn top_user_relevant_frame(&self) -> Option<usize> {
        debug_assert_eq!(self.top_user_relevant_frame, self.compute_top_user_relevant_frame());
        // This can be called upon creation of an allocation. We create allocations while setting up
        // parts of the Rust runtime when we do not have any stack frames yet, so we need to handle
        // empty stacks.
        self.top_user_relevant_frame.or_else(|| self.stack.len().checked_sub(1))
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

impl<'mir, 'tcx> Thread<'mir, 'tcx> {
    fn new(name: Option<&str>, on_stack_empty: Option<StackEmptyCallback<'mir, 'tcx>>) -> Self {
        Self {
            state: ThreadState::Enabled,
            thread_name: name.map(|name| Vec::from(name.as_bytes())),
            stack: Vec::new(),
            top_user_relevant_frame: None,
            join_status: ThreadJoinStatus::Joinable,
            panic_payloads: Vec::new(),
            last_error: None,
            on_stack_empty,
        }
    }
}

impl VisitTags for Thread<'_, '_> {
    fn visit_tags(&self, visit: &mut dyn FnMut(BorTag)) {
        let Thread {
            panic_payloads: panic_payload,
            last_error,
            stack,
            top_user_relevant_frame: _,
            state: _,
            thread_name: _,
            join_status: _,
            on_stack_empty: _, // we assume the closure captures no GC-relevant state
        } = self;

        for payload in panic_payload {
            payload.visit_tags(visit);
        }
        last_error.visit_tags(visit);
        for frame in stack {
            frame.visit_tags(visit)
        }
    }
}

impl VisitTags for Frame<'_, '_, Provenance, FrameExtra<'_>> {
    fn visit_tags(&self, visit: &mut dyn FnMut(BorTag)) {
        let Frame {
            return_place,
            locals,
            extra,
            body: _,
            instance: _,
            return_to_block: _,
            loc: _,
            // There are some private fields we cannot access; they contain no tags.
            ..
        } = self;

        // Return place.
        return_place.visit_tags(visit);
        // Locals.
        for local in locals.iter() {
            if let LocalValue::Live(value) = &local.value {
                value.visit_tags(visit);
            }
        }

        extra.visit_tags(visit);
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
    fn get_wait_time(&self, clock: &Clock) -> Duration {
        match self {
            Time::Monotonic(instant) => instant.duration_since(clock.now()),
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
    pub(crate) sync: SynchronizationState<'mir, 'tcx>,
    /// A mapping from a thread-local static to an allocation id of a thread
    /// specific allocation.
    thread_local_alloc_ids: RefCell<FxHashMap<(DefId, ThreadId), Pointer<Provenance>>>,
    /// A flag that indicates that we should change the active thread.
    yield_active_thread: bool,
    /// Callbacks that are called once the specified time passes.
    timeout_callbacks: FxHashMap<ThreadId, TimeoutCallbackInfo<'mir, 'tcx>>,
}

impl VisitTags for ThreadManager<'_, '_> {
    fn visit_tags(&self, visit: &mut dyn FnMut(BorTag)) {
        let ThreadManager {
            threads,
            thread_local_alloc_ids,
            timeout_callbacks,
            active_thread: _,
            yield_active_thread: _,
            sync,
        } = self;

        for thread in threads {
            thread.visit_tags(visit);
        }
        for ptr in thread_local_alloc_ids.borrow().values() {
            ptr.visit_tags(visit);
        }
        for callback in timeout_callbacks.values() {
            callback.callback.visit_tags(visit);
        }
        sync.visit_tags(visit);
    }
}

impl<'mir, 'tcx> Default for ThreadManager<'mir, 'tcx> {
    fn default() -> Self {
        let mut threads = IndexVec::new();
        // Create the main thread and add it to the list of threads.
        threads.push(Thread::new(Some("main"), None));
        Self {
            active_thread: ThreadId::new(0),
            threads,
            sync: SynchronizationState::default(),
            thread_local_alloc_ids: Default::default(),
            yield_active_thread: false,
            timeout_callbacks: FxHashMap::default(),
        }
    }
}

impl<'mir, 'tcx: 'mir> ThreadManager<'mir, 'tcx> {
    pub(crate) fn init(
        ecx: &mut MiriInterpCx<'mir, 'tcx>,
        on_main_stack_empty: StackEmptyCallback<'mir, 'tcx>,
    ) {
        ecx.machine.threads.threads[ThreadId::new(0)].on_stack_empty = Some(on_main_stack_empty);
        if ecx.tcx.sess.target.os.as_ref() != "windows" {
            // The main thread can *not* be joined on except on windows.
            ecx.machine.threads.threads[ThreadId::new(0)].join_status = ThreadJoinStatus::Detached;
        }
    }

    /// Check if we have an allocation for the given thread local static for the
    /// active thread.
    fn get_thread_local_alloc_id(&self, def_id: DefId) -> Option<Pointer<Provenance>> {
        self.thread_local_alloc_ids.borrow().get(&(def_id, self.active_thread)).cloned()
    }

    /// Set the pointer for the allocation of the given thread local
    /// static for the active thread.
    ///
    /// Panics if a thread local is initialized twice for the same thread.
    fn set_thread_local_alloc(&self, def_id: DefId, ptr: Pointer<Provenance>) {
        self.thread_local_alloc_ids
            .borrow_mut()
            .try_insert((def_id, self.active_thread), ptr)
            .unwrap();
    }

    /// Borrow the stack of the active thread.
    pub fn active_thread_stack(&self) -> &[Frame<'mir, 'tcx, Provenance, FrameExtra<'tcx>>] {
        &self.threads[self.active_thread].stack
    }

    /// Mutably borrow the stack of the active thread.
    fn active_thread_stack_mut(
        &mut self,
    ) -> &mut Vec<Frame<'mir, 'tcx, Provenance, FrameExtra<'tcx>>> {
        &mut self.threads[self.active_thread].stack
    }

    pub fn all_stacks(
        &self,
    ) -> impl Iterator<Item = &[Frame<'mir, 'tcx, Provenance, FrameExtra<'tcx>>]> {
        self.threads.iter().map(|t| &t.stack[..])
    }

    /// Create a new thread and returns its id.
    fn create_thread(&mut self, on_stack_empty: StackEmptyCallback<'mir, 'tcx>) -> ThreadId {
        let new_thread_id = ThreadId::new(self.threads.len());
        self.threads.push(Thread::new(None, Some(on_stack_empty)));
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
    pub fn get_active_thread_id(&self) -> ThreadId {
        self.active_thread
    }

    /// Get the total number of threads that were ever spawn by this program.
    pub fn get_total_thread_count(&self) -> usize {
        self.threads.len()
    }

    /// Get the total of threads that are currently live, i.e., not yet terminated.
    /// (They might be blocked.)
    pub fn get_live_thread_count(&self) -> usize {
        self.threads.iter().filter(|t| !matches!(t.state, ThreadState::Terminated)).count()
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
    pub fn active_thread_mut(&mut self) -> &mut Thread<'mir, 'tcx> {
        &mut self.threads[self.active_thread]
    }

    /// Get a shared borrow of the currently active thread.
    pub fn active_thread_ref(&self) -> &Thread<'mir, 'tcx> {
        &self.threads[self.active_thread]
    }

    /// Mark the thread as detached, which means that no other thread will try
    /// to join it and the thread is responsible for cleaning up.
    ///
    /// `allow_terminated_joined` allows detaching joined threads that have already terminated.
    /// This matches Windows's behavior for `CloseHandle`.
    ///
    /// See <https://docs.microsoft.com/en-us/windows/win32/procthread/thread-handles-and-identifiers>:
    /// > The handle is valid until closed, even after the thread it represents has been terminated.
    fn detach_thread(&mut self, id: ThreadId, allow_terminated_joined: bool) -> InterpResult<'tcx> {
        trace!("detaching {:?}", id);

        let is_ub = if allow_terminated_joined && self.threads[id].state == ThreadState::Terminated
        {
            // "Detached" in particular means "not yet joined". Redundant detaching is still UB.
            self.threads[id].join_status == ThreadJoinStatus::Detached
        } else {
            self.threads[id].join_status != ThreadJoinStatus::Joinable
        };
        if is_ub {
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
        if self.threads[joined_thread_id].join_status == ThreadJoinStatus::Detached {
            // On Windows this corresponds to joining on a closed handle.
            throw_ub_format!("trying to join a detached thread");
        }

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
                data_race.thread_joined(self, self.active_thread, joined_thread_id);
            }
        }
        Ok(())
    }

    /// Mark that the active thread tries to exclusively join the thread with `joined_thread_id`.
    /// If the thread is already joined by another thread, it will throw UB
    fn join_thread_exclusive(
        &mut self,
        joined_thread_id: ThreadId,
        data_race: Option<&mut data_race::GlobalState>,
    ) -> InterpResult<'tcx> {
        if self.threads[joined_thread_id].join_status == ThreadJoinStatus::Joined {
            throw_ub_format!("trying to join an already joined thread");
        }

        if joined_thread_id == self.active_thread {
            throw_ub_format!("trying to join itself");
        }

        assert!(
            self.threads
                .iter()
                .all(|thread| thread.state != ThreadState::BlockedOnJoin(joined_thread_id)),
            "this thread already has threads waiting for its termination"
        );

        self.join_thread(joined_thread_id, data_race)
    }

    /// Set the name of the given thread.
    pub fn set_thread_name(&mut self, thread: ThreadId, new_thread_name: Vec<u8>) {
        self.threads[thread].thread_name = Some(new_thread_name);
    }

    /// Get the name of the given thread.
    pub fn get_thread_name(&self, thread: ThreadId) -> &[u8] {
        self.threads[thread].thread_name()
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
    fn get_ready_callback(
        &mut self,
        clock: &Clock,
    ) -> Option<(ThreadId, TimeoutCallback<'mir, 'tcx>)> {
        // We iterate over all threads in the order of their indices because
        // this allows us to have a deterministic scheduler.
        for thread in self.threads.indices() {
            match self.timeout_callbacks.entry(thread) {
                Entry::Occupied(entry) => {
                    if entry.get().call_time.get_wait_time(clock) == Duration::new(0, 0) {
                        return Some((thread, entry.remove().callback));
                    }
                }
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
        current_span: Span,
    ) -> Vec<Pointer<Provenance>> {
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
                false
            });
        }
        // Set the thread into a terminated state in the data-race detector.
        if let Some(ref mut data_race) = data_race {
            data_race.thread_terminated(self, current_span);
        }
        // Check if we need to unblock any threads.
        let mut joined_threads = vec![]; // store which threads joined, we'll need it
        for (i, thread) in self.threads.iter_enumerated_mut() {
            if thread.state == ThreadState::BlockedOnJoin(self.active_thread) {
                // The thread has terminated, mark happens-before edge to joining thread
                if data_race.is_some() {
                    joined_threads.push(i);
                }
                trace!("unblocking {:?} because {:?} terminated", i, self.active_thread);
                thread.state = ThreadState::Enabled;
            }
        }
        for &i in &joined_threads {
            data_race.as_mut().unwrap().thread_joined(self, i, self.active_thread);
        }
        free_tls_statics
    }

    /// Decide which action to take next and on which thread.
    ///
    /// The currently implemented scheduling policy is the one that is commonly
    /// used in stateless model checkers such as Loom: run the active thread as
    /// long as we can and switch only when we have to (the active thread was
    /// blocked, terminated, or has explicitly asked to be preempted).
    fn schedule(&mut self, clock: &Clock) -> InterpResult<'tcx, SchedulingAction> {
        // This thread and the program can keep going.
        if self.threads[self.active_thread].state == ThreadState::Enabled
            && !self.yield_active_thread
        {
            // The currently active thread is still enabled, just continue with it.
            return Ok(SchedulingAction::ExecuteStep);
        }
        // The active thread yielded or got terminated. Let's see if there are any timeouts to take
        // care of. We do this *before* running any other thread, to ensure that timeouts "in the
        // past" fire before any other thread can take an action. This ensures that for
        // `pthread_cond_timedwait`, "an error is returned if [...] the absolute time specified by
        // abstime has already been passed at the time of the call".
        // <https://pubs.opengroup.org/onlinepubs/9699919799/functions/pthread_cond_timedwait.html>
        let potential_sleep_time =
            self.timeout_callbacks.values().map(|info| info.call_time.get_wait_time(clock)).min();
        if potential_sleep_time == Some(Duration::new(0, 0)) {
            return Ok(SchedulingAction::ExecuteTimeoutCallback);
        }
        // No callbacks immediately scheduled, pick a regular thread to execute.
        // The active thread blocked or yielded. So we go search for another enabled thread.
        // Crucially, we start searching at the current active thread ID, rather than at 0, since we
        // want to avoid always scheduling threads 0 and 1 without ever making progress in thread 2.
        //
        // `skip(N)` means we start iterating at thread N, so we skip 1 more to start just *after*
        // the active thread. Then after that we look at `take(N)`, i.e., the threads *before* the
        // active thread.
        let threads = self
            .threads
            .iter_enumerated()
            .skip(self.active_thread.index() + 1)
            .chain(self.threads.iter_enumerated().take(self.active_thread.index()));
        for (id, thread) in threads {
            debug_assert_ne!(self.active_thread, id);
            if thread.state == ThreadState::Enabled {
                self.active_thread = id;
                break;
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
            Ok(SchedulingAction::Sleep(sleep_time))
        } else {
            throw_machine_stop!(TerminationInfo::Deadlock);
        }
    }
}

impl<'mir, 'tcx: 'mir> EvalContextPrivExt<'mir, 'tcx> for MiriInterpCx<'mir, 'tcx> {}
trait EvalContextPrivExt<'mir, 'tcx: 'mir>: MiriInterpCxExt<'mir, 'tcx> {
    /// Execute a timeout callback on the callback's thread.
    #[inline]
    fn run_timeout_callback(&mut self) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();
        let (thread, callback) = if let Some((thread, callback)) =
            this.machine.threads.get_ready_callback(&this.machine.clock)
        {
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
        callback.call(this)?;
        this.set_active_thread(old_thread);
        Ok(())
    }

    #[inline]
    fn run_on_stack_empty(&mut self) -> InterpResult<'tcx, Poll<()>> {
        let this = self.eval_context_mut();
        let mut callback = this
            .active_thread_mut()
            .on_stack_empty
            .take()
            .expect("`on_stack_empty` not set up, or already running");
        let res = callback(this)?;
        this.active_thread_mut().on_stack_empty = Some(callback);
        Ok(res)
    }
}

// Public interface to thread management.
impl<'mir, 'tcx: 'mir> EvalContextExt<'mir, 'tcx> for crate::MiriInterpCx<'mir, 'tcx> {}
pub trait EvalContextExt<'mir, 'tcx: 'mir>: crate::MiriInterpCxExt<'mir, 'tcx> {
    /// Get a thread-specific allocation id for the given thread-local static.
    /// If needed, allocate a new one.
    fn get_or_create_thread_local_alloc(
        &mut self,
        def_id: DefId,
    ) -> InterpResult<'tcx, Pointer<Provenance>> {
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
            // We don't give a span -- statics don't need that, they cannot be generic or associated.
            let allocation = this.ctfe_query(None, |tcx| tcx.eval_static_initializer(def_id))?;
            let mut allocation = allocation.inner().clone();
            // This allocation will be deallocated when the thread dies, so it is not in read-only memory.
            allocation.mutability = Mutability::Mut;
            // Create a fresh allocation with this content.
            let new_alloc = this.allocate_raw_ptr(allocation, MiriMemoryKind::Tls.into())?;
            this.machine.threads.set_thread_local_alloc(def_id, new_alloc);
            Ok(new_alloc)
        }
    }

    /// Start a regular (non-main) thread.
    #[inline]
    fn start_regular_thread(
        &mut self,
        thread: Option<MPlaceTy<'tcx, Provenance>>,
        start_routine: Pointer<Option<Provenance>>,
        start_abi: Abi,
        func_arg: ImmTy<'tcx, Provenance>,
        ret_layout: TyAndLayout<'tcx>,
    ) -> InterpResult<'tcx, ThreadId> {
        let this = self.eval_context_mut();

        // Create the new thread
        let new_thread_id = this.machine.threads.create_thread({
            let mut state = tls::TlsDtorsState::default();
            Box::new(move |m| state.on_stack_empty(m))
        });
        let current_span = this.machine.current_span();
        if let Some(data_race) = &mut this.machine.data_race {
            data_race.thread_created(&this.machine.threads, new_thread_id, current_span);
        }

        // Write the current thread-id, switch to the next thread later
        // to treat this write operation as occurring on the current thread.
        if let Some(thread_info_place) = thread {
            this.write_scalar(
                Scalar::from_uint(new_thread_id.to_u32(), thread_info_place.layout.size),
                &thread_info_place.into(),
            )?;
        }

        // Finally switch to new thread so that we can push the first stackframe.
        // After this all accesses will be treated as occurring in the new thread.
        let old_thread_id = this.set_active_thread(new_thread_id);

        // Perform the function pointer load in the new thread frame.
        let instance = this.get_ptr_fn(start_routine)?.as_instance()?;

        // Note: the returned value is currently ignored (see the FIXME in
        // pthread_join in shims/unix/thread.rs) because the Rust standard library does not use
        // it.
        let ret_place = this.allocate(ret_layout, MiriMemoryKind::Machine.into())?;

        this.call_function(
            instance,
            start_abi,
            &[*func_arg],
            Some(&ret_place.into()),
            StackPopCleanup::Root { cleanup: true },
        )?;

        // Restore the old active thread frame.
        this.set_active_thread(old_thread_id);

        Ok(new_thread_id)
    }

    #[inline]
    fn detach_thread(
        &mut self,
        thread_id: ThreadId,
        allow_terminated_joined: bool,
    ) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();
        this.machine.threads.detach_thread(thread_id, allow_terminated_joined)
    }

    #[inline]
    fn join_thread(&mut self, joined_thread_id: ThreadId) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();
        this.machine.threads.join_thread(joined_thread_id, this.machine.data_race.as_mut())?;
        Ok(())
    }

    #[inline]
    fn join_thread_exclusive(&mut self, joined_thread_id: ThreadId) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();
        this.machine
            .threads
            .join_thread_exclusive(joined_thread_id, this.machine.data_race.as_mut())?;
        Ok(())
    }

    #[inline]
    fn set_active_thread(&mut self, thread_id: ThreadId) -> ThreadId {
        let this = self.eval_context_mut();
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
    fn active_thread_stack(&self) -> &[Frame<'mir, 'tcx, Provenance, FrameExtra<'tcx>>] {
        let this = self.eval_context_ref();
        this.machine.threads.active_thread_stack()
    }

    #[inline]
    fn active_thread_stack_mut(
        &mut self,
    ) -> &mut Vec<Frame<'mir, 'tcx, Provenance, FrameExtra<'tcx>>> {
        let this = self.eval_context_mut();
        this.machine.threads.active_thread_stack_mut()
    }

    /// Set the name of the current thread. The buffer must not include the null terminator.
    #[inline]
    fn set_thread_name(&mut self, thread: ThreadId, new_thread_name: Vec<u8>) {
        let this = self.eval_context_mut();
        this.machine.threads.set_thread_name(thread, new_thread_name);
    }

    #[inline]
    fn set_thread_name_wide(&mut self, thread: ThreadId, new_thread_name: &[u16]) {
        let this = self.eval_context_mut();

        // The Windows `GetThreadDescription` shim to get the thread name isn't implemented, so being lossy is okay.
        // This is only read by diagnostics, which already use `from_utf8_lossy`.
        this.machine
            .threads
            .set_thread_name(thread, String::from_utf16_lossy(new_thread_name).into_bytes());
    }

    #[inline]
    fn get_thread_name<'c>(&'c self, thread: ThreadId) -> &'c [u8]
    where
        'mir: 'c,
    {
        self.eval_context_ref().machine.threads.get_thread_name(thread)
    }

    #[inline]
    fn block_thread(&mut self, thread: ThreadId) {
        self.eval_context_mut().machine.threads.block_thread(thread);
    }

    #[inline]
    fn unblock_thread(&mut self, thread: ThreadId) {
        self.eval_context_mut().machine.threads.unblock_thread(thread);
    }

    #[inline]
    fn yield_active_thread(&mut self) {
        self.eval_context_mut().machine.threads.yield_active_thread();
    }

    #[inline]
    fn maybe_preempt_active_thread(&mut self) {
        use rand::Rng as _;

        let this = self.eval_context_mut();
        if this.machine.rng.get_mut().gen_bool(this.machine.preemption_rate) {
            this.yield_active_thread();
        }
    }

    #[inline]
    fn register_timeout_callback(
        &mut self,
        thread: ThreadId,
        call_time: Time,
        callback: TimeoutCallback<'mir, 'tcx>,
    ) {
        let this = self.eval_context_mut();
        if !this.machine.communicate() && matches!(call_time, Time::RealTime(..)) {
            panic!("cannot have `RealTime` callback with isolation enabled!")
        }
        this.machine.threads.register_timeout_callback(thread, call_time, callback);
    }

    #[inline]
    fn unregister_timeout_callback_if_exists(&mut self, thread: ThreadId) {
        let this = self.eval_context_mut();
        this.machine.threads.unregister_timeout_callback_if_exists(thread);
    }

    /// Run the core interpreter loop. Returns only when an interrupt occurs (an error or program
    /// termination).
    fn run_threads(&mut self) -> InterpResult<'tcx, !> {
        static SIGNALED: AtomicBool = AtomicBool::new(false);
        ctrlc::set_handler(move || {
            // Indicate that we have ben signaled to stop. If we were already signaled, exit
            // immediately. In our interpreter loop we try to consult this value often, but if for
            // whatever reason we don't get to that check or the cleanup we do upon finding that
            // this bool has become true takes a long time, the exit here will promptly exit the
            // process on the second Ctrl-C.
            if SIGNALED.swap(true, Relaxed) {
                std::process::exit(1);
            }
        })
        .unwrap();
        let this = self.eval_context_mut();
        loop {
            if SIGNALED.load(Relaxed) {
                this.machine.handle_abnormal_termination();
                std::process::exit(1);
            }
            match this.machine.threads.schedule(&this.machine.clock)? {
                SchedulingAction::ExecuteStep => {
                    if !this.step()? {
                        // See if this thread can do something else.
                        match this.run_on_stack_empty()? {
                            Poll::Pending => {} // keep going
                            Poll::Ready(()) => this.terminate_active_thread()?,
                        }
                    }
                }
                SchedulingAction::ExecuteTimeoutCallback => {
                    this.run_timeout_callback()?;
                }
                SchedulingAction::Sleep(duration) => {
                    this.machine.clock.sleep(duration);
                }
            }
        }
    }

    /// Handles thread termination of the active thread: wakes up threads joining on this one,
    /// and deallocated thread-local statics.
    ///
    /// This is called by the eval loop when a thread's on_stack_empty returns `Ready`.
    #[inline]
    fn terminate_active_thread(&mut self) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();
        let thread = this.active_thread_mut();
        assert!(thread.stack.is_empty(), "only threads with an empty stack can be terminated");
        thread.state = ThreadState::Terminated;

        let current_span = this.machine.current_span();
        for ptr in
            this.machine.threads.thread_terminated(this.machine.data_race.as_mut(), current_span)
        {
            this.deallocate_ptr(ptr.into(), None, MiriMemoryKind::Tls.into())?;
        }
        Ok(())
    }
}
