//! Implements threads.

use std::mem;
use std::sync::atomic::Ordering::Relaxed;
use std::task::Poll;
use std::time::{Duration, SystemTime};

use either::Either;
use rustc_abi::ExternAbi;
use rustc_const_eval::CTRL_C_RECEIVED;
use rustc_data_structures::fx::FxHashMap;
use rustc_hir::def_id::DefId;
use rustc_index::{Idx, IndexVec};
use rustc_middle::mir::Mutability;
use rustc_middle::ty::layout::TyAndLayout;
use rustc_span::Span;

use crate::concurrency::data_race;
use crate::shims::tls;
use crate::*;

#[derive(Clone, Copy, Debug, PartialEq)]
enum SchedulingAction {
    /// Execute step on the active thread.
    ExecuteStep,
    /// Execute a timeout callback.
    ExecuteTimeoutCallback,
    /// Wait for a bit, until there is a timeout to be called.
    Sleep(Duration),
}

/// What to do with TLS allocations from terminated threads
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TlsAllocAction {
    /// Deallocate backing memory of thread-local statics as usual
    Deallocate,
    /// Skip deallocating backing memory of thread-local statics and consider all memory reachable
    /// from them as "allowed to leak" (like global `static`s).
    Leak,
}

/// The argument type for the "unblock" callback, indicating why the thread got unblocked.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum UnblockKind {
    /// Operation completed successfully, thread continues normal execution.
    Ready,
    /// The operation did not complete within its specified duration.
    TimedOut,
}

/// Type alias for unblock callbacks, i.e. machine callbacks invoked when
/// a thread gets unblocked.
pub type DynUnblockCallback<'tcx> = DynMachineCallback<'tcx, UnblockKind>;

/// A thread identifier.
#[derive(Clone, Copy, Debug, PartialOrd, Ord, PartialEq, Eq, Hash)]
pub struct ThreadId(u32);

impl ThreadId {
    pub fn to_u32(self) -> u32 {
        self.0
    }

    /// Create a new thread id from a `u32` without checking if this thread exists.
    pub fn new_unchecked(id: u32) -> Self {
        Self(id)
    }

    pub const MAIN_THREAD: ThreadId = ThreadId(0);
}

impl Idx for ThreadId {
    fn new(idx: usize) -> Self {
        ThreadId(u32::try_from(idx).unwrap())
    }

    fn index(self) -> usize {
        usize::try_from(self.0).unwrap()
    }
}

impl From<ThreadId> for u64 {
    fn from(t: ThreadId) -> Self {
        t.0.into()
    }
}

/// Keeps track of what the thread is blocked on.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum BlockReason {
    /// The thread tried to join the specified thread and is blocked until that
    /// thread terminates.
    Join(ThreadId),
    /// Waiting for time to pass.
    Sleep,
    /// Blocked on a mutex.
    Mutex,
    /// Blocked on a condition variable.
    Condvar(CondvarId),
    /// Blocked on a reader-writer lock.
    RwLock(RwLockId),
    /// Blocked on a Futex variable.
    Futex,
    /// Blocked on an InitOnce.
    InitOnce(InitOnceId),
    /// Blocked on epoll.
    Epoll,
    /// Blocked on eventfd.
    Eventfd,
    /// Blocked on unnamed_socket.
    UnnamedSocket,
}

/// The state of a thread.
enum ThreadState<'tcx> {
    /// The thread is enabled and can be executed.
    Enabled,
    /// The thread is blocked on something.
    Blocked { reason: BlockReason, timeout: Option<Timeout>, callback: DynUnblockCallback<'tcx> },
    /// The thread has terminated its execution. We do not delete terminated
    /// threads (FIXME: why?).
    Terminated,
}

impl<'tcx> std::fmt::Debug for ThreadState<'tcx> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Enabled => write!(f, "Enabled"),
            Self::Blocked { reason, timeout, .. } =>
                f.debug_struct("Blocked").field("reason", reason).field("timeout", timeout).finish(),
            Self::Terminated => write!(f, "Terminated"),
        }
    }
}

impl<'tcx> ThreadState<'tcx> {
    fn is_enabled(&self) -> bool {
        matches!(self, ThreadState::Enabled)
    }

    fn is_terminated(&self) -> bool {
        matches!(self, ThreadState::Terminated)
    }

    fn is_blocked_on(&self, reason: BlockReason) -> bool {
        matches!(*self, ThreadState::Blocked { reason: actual_reason, .. } if actual_reason == reason)
    }
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
pub struct Thread<'tcx> {
    state: ThreadState<'tcx>,

    /// Name of the thread.
    thread_name: Option<Vec<u8>>,

    /// The virtual call stack.
    stack: Vec<Frame<'tcx, Provenance, FrameExtra<'tcx>>>,

    /// The function to call when the stack ran empty, to figure out what to do next.
    /// Conceptually, this is the interpreter implementation of the things that happen 'after' the
    /// Rust language entry point for this thread returns (usually implemented by the C or OS runtime).
    /// (`None` is an error, it means the callback has not been set up yet or is actively running.)
    pub(crate) on_stack_empty: Option<StackEmptyCallback<'tcx>>,

    /// The index of the topmost user-relevant frame in `stack`. This field must contain
    /// the value produced by `get_top_user_relevant_frame`.
    /// The `None` state here represents
    /// This field is a cache to reduce how often we call that method. The cache is manually
    /// maintained inside `MiriMachine::after_stack_push` and `MiriMachine::after_stack_pop`.
    top_user_relevant_frame: Option<usize>,

    /// The join status.
    join_status: ThreadJoinStatus,

    /// Stack of active panic payloads for the current thread. Used for storing
    /// the argument of the call to `miri_start_unwind` (the panic payload) when unwinding.
    /// This is pointer-sized, and matches the `Payload` type in `src/libpanic_unwind/miri.rs`.
    ///
    /// In real unwinding, the payload gets passed as an argument to the landing pad,
    /// which then forwards it to 'Resume'. However this argument is implicit in MIR,
    /// so we have to store it out-of-band. When there are multiple active unwinds,
    /// the innermost one is always caught first, so we can store them as a stack.
    pub(crate) panic_payloads: Vec<ImmTy<'tcx>>,

    /// Last OS error location in memory. It is a 32-bit integer.
    pub(crate) last_error: Option<MPlaceTy<'tcx>>,
}

pub type StackEmptyCallback<'tcx> =
    Box<dyn FnMut(&mut MiriInterpCx<'tcx>) -> InterpResult<'tcx, Poll<()>> + 'tcx>;

impl<'tcx> Thread<'tcx> {
    /// Get the name of the current thread if it was set.
    fn thread_name(&self) -> Option<&[u8]> {
        self.thread_name.as_deref()
    }

    /// Get the name of the current thread for display purposes; will include thread ID if not set.
    fn thread_display_name(&self, id: ThreadId) -> String {
        if let Some(ref thread_name) = self.thread_name {
            String::from_utf8_lossy(thread_name).into_owned()
        } else {
            format!("unnamed-{}", id.index())
        }
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

    pub fn current_span(&self) -> Span {
        self.top_user_relevant_frame()
            .map(|frame_idx| self.stack[frame_idx].current_span())
            .unwrap_or(rustc_span::DUMMY_SP)
    }
}

impl<'tcx> std::fmt::Debug for Thread<'tcx> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}({:?}, {:?})",
            String::from_utf8_lossy(self.thread_name().unwrap_or(b"<unnamed>")),
            self.state,
            self.join_status
        )
    }
}

impl<'tcx> Thread<'tcx> {
    fn new(name: Option<&str>, on_stack_empty: Option<StackEmptyCallback<'tcx>>) -> Self {
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

impl VisitProvenance for Thread<'_> {
    fn visit_provenance(&self, visit: &mut VisitWith<'_>) {
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
            payload.visit_provenance(visit);
        }
        last_error.visit_provenance(visit);
        for frame in stack {
            frame.visit_provenance(visit)
        }
    }
}

impl VisitProvenance for Frame<'_, Provenance, FrameExtra<'_>> {
    fn visit_provenance(&self, visit: &mut VisitWith<'_>) {
        let Frame {
            return_place,
            locals,
            extra,
            // There are some private fields we cannot access; they contain no tags.
            ..
        } = self;

        // Return place.
        return_place.visit_provenance(visit);
        // Locals.
        for local in locals.iter() {
            match local.as_mplace_or_imm() {
                None => {}
                Some(Either::Left((ptr, meta))) => {
                    ptr.visit_provenance(visit);
                    meta.visit_provenance(visit);
                }
                Some(Either::Right(imm)) => {
                    imm.visit_provenance(visit);
                }
            }
        }

        extra.visit_provenance(visit);
    }
}

/// The moment in time when a blocked thread should be woken up.
#[derive(Debug)]
enum Timeout {
    Monotonic(Instant),
    RealTime(SystemTime),
}

impl Timeout {
    /// How long do we have to wait from now until the specified time?
    fn get_wait_time(&self, clock: &MonotonicClock) -> Duration {
        match self {
            Timeout::Monotonic(instant) => instant.duration_since(clock.now()),
            Timeout::RealTime(time) =>
                time.duration_since(SystemTime::now()).unwrap_or(Duration::ZERO),
        }
    }

    /// Will try to add `duration`, but if that overflows it may add less.
    fn add_lossy(&self, duration: Duration) -> Self {
        match self {
            Timeout::Monotonic(i) => Timeout::Monotonic(i.add_lossy(duration)),
            Timeout::RealTime(s) => {
                // If this overflows, try adding just 1h and assume that will not overflow.
                Timeout::RealTime(
                    s.checked_add(duration)
                        .unwrap_or_else(|| s.checked_add(Duration::from_secs(3600)).unwrap()),
                )
            }
        }
    }
}

/// The clock to use for the timeout you are asking for.
#[derive(Debug, Copy, Clone)]
pub enum TimeoutClock {
    Monotonic,
    RealTime,
}

/// Whether the timeout is relative or absolute.
#[derive(Debug, Copy, Clone)]
pub enum TimeoutAnchor {
    Relative,
    Absolute,
}

/// An error signaling that the requested thread doesn't exist.
#[derive(Debug, Copy, Clone)]
pub struct ThreadNotFound;

/// A set of threads.
#[derive(Debug)]
pub struct ThreadManager<'tcx> {
    /// Identifier of the currently active thread.
    active_thread: ThreadId,
    /// Threads used in the program.
    ///
    /// Note that this vector also contains terminated threads.
    threads: IndexVec<ThreadId, Thread<'tcx>>,
    /// A mapping from a thread-local static to the thread specific allocation.
    thread_local_allocs: FxHashMap<(DefId, ThreadId), StrictPointer>,
    /// A flag that indicates that we should change the active thread.
    yield_active_thread: bool,
}

impl VisitProvenance for ThreadManager<'_> {
    fn visit_provenance(&self, visit: &mut VisitWith<'_>) {
        let ThreadManager {
            threads,
            thread_local_allocs,
            active_thread: _,
            yield_active_thread: _,
        } = self;

        for thread in threads {
            thread.visit_provenance(visit);
        }
        for ptr in thread_local_allocs.values() {
            ptr.visit_provenance(visit);
        }
    }
}

impl<'tcx> Default for ThreadManager<'tcx> {
    fn default() -> Self {
        let mut threads = IndexVec::new();
        // Create the main thread and add it to the list of threads.
        threads.push(Thread::new(Some("main"), None));
        Self {
            active_thread: ThreadId::MAIN_THREAD,
            threads,
            thread_local_allocs: Default::default(),
            yield_active_thread: false,
        }
    }
}

impl<'tcx> ThreadManager<'tcx> {
    pub(crate) fn init(
        ecx: &mut MiriInterpCx<'tcx>,
        on_main_stack_empty: StackEmptyCallback<'tcx>,
    ) {
        ecx.machine.threads.threads[ThreadId::MAIN_THREAD].on_stack_empty =
            Some(on_main_stack_empty);
        if ecx.tcx.sess.target.os.as_ref() != "windows" {
            // The main thread can *not* be joined on except on windows.
            ecx.machine.threads.threads[ThreadId::MAIN_THREAD].join_status =
                ThreadJoinStatus::Detached;
        }
    }

    pub fn thread_id_try_from(&self, id: impl TryInto<u32>) -> Result<ThreadId, ThreadNotFound> {
        if let Ok(id) = id.try_into()
            && usize::try_from(id).is_ok_and(|id| id < self.threads.len())
        {
            Ok(ThreadId(id))
        } else {
            Err(ThreadNotFound)
        }
    }

    /// Check if we have an allocation for the given thread local static for the
    /// active thread.
    fn get_thread_local_alloc_id(&self, def_id: DefId) -> Option<StrictPointer> {
        self.thread_local_allocs.get(&(def_id, self.active_thread)).cloned()
    }

    /// Set the pointer for the allocation of the given thread local
    /// static for the active thread.
    ///
    /// Panics if a thread local is initialized twice for the same thread.
    fn set_thread_local_alloc(&mut self, def_id: DefId, ptr: StrictPointer) {
        self.thread_local_allocs.try_insert((def_id, self.active_thread), ptr).unwrap();
    }

    /// Borrow the stack of the active thread.
    pub fn active_thread_stack(&self) -> &[Frame<'tcx, Provenance, FrameExtra<'tcx>>] {
        &self.threads[self.active_thread].stack
    }

    /// Mutably borrow the stack of the active thread.
    pub fn active_thread_stack_mut(
        &mut self,
    ) -> &mut Vec<Frame<'tcx, Provenance, FrameExtra<'tcx>>> {
        &mut self.threads[self.active_thread].stack
    }

    pub fn all_stacks(
        &self,
    ) -> impl Iterator<Item = (ThreadId, &[Frame<'tcx, Provenance, FrameExtra<'tcx>>])> {
        self.threads.iter_enumerated().map(|(id, t)| (id, &t.stack[..]))
    }

    /// Create a new thread and returns its id.
    fn create_thread(&mut self, on_stack_empty: StackEmptyCallback<'tcx>) -> ThreadId {
        let new_thread_id = ThreadId::new(self.threads.len());
        self.threads.push(Thread::new(None, Some(on_stack_empty)));
        new_thread_id
    }

    /// Set an active thread and return the id of the thread that was active before.
    fn set_active_thread_id(&mut self, id: ThreadId) -> ThreadId {
        assert!(id.index() < self.threads.len());
        info!(
            "---------- Now executing on thread `{}` (previous: `{}`) ----------------------------------------",
            self.get_thread_display_name(id),
            self.get_thread_display_name(self.active_thread)
        );
        std::mem::replace(&mut self.active_thread, id)
    }

    /// Get the id of the currently active thread.
    pub fn active_thread(&self) -> ThreadId {
        self.active_thread
    }

    /// Get the total number of threads that were ever spawn by this program.
    pub fn get_total_thread_count(&self) -> usize {
        self.threads.len()
    }

    /// Get the total of threads that are currently live, i.e., not yet terminated.
    /// (They might be blocked.)
    pub fn get_live_thread_count(&self) -> usize {
        self.threads.iter().filter(|t| !t.state.is_terminated()).count()
    }

    /// Has the given thread terminated?
    fn has_terminated(&self, thread_id: ThreadId) -> bool {
        self.threads[thread_id].state.is_terminated()
    }

    /// Have all threads terminated?
    fn have_all_terminated(&self) -> bool {
        self.threads.iter().all(|thread| thread.state.is_terminated())
    }

    /// Enable the thread for execution. The thread must be terminated.
    fn enable_thread(&mut self, thread_id: ThreadId) {
        assert!(self.has_terminated(thread_id));
        self.threads[thread_id].state = ThreadState::Enabled;
    }

    /// Get a mutable borrow of the currently active thread.
    pub fn active_thread_mut(&mut self) -> &mut Thread<'tcx> {
        &mut self.threads[self.active_thread]
    }

    /// Get a shared borrow of the currently active thread.
    pub fn active_thread_ref(&self) -> &Thread<'tcx> {
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

        let is_ub = if allow_terminated_joined && self.threads[id].state.is_terminated() {
            // "Detached" in particular means "not yet joined". Redundant detaching is still UB.
            self.threads[id].join_status == ThreadJoinStatus::Detached
        } else {
            self.threads[id].join_status != ThreadJoinStatus::Joinable
        };
        if is_ub {
            throw_ub_format!("trying to detach thread that was already detached or joined");
        }

        self.threads[id].join_status = ThreadJoinStatus::Detached;
        interp_ok(())
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
        if !self.threads[joined_thread_id].state.is_terminated() {
            trace!(
                "{:?} blocked on {:?} when trying to join",
                self.active_thread, joined_thread_id
            );
            // The joined thread is still running, we need to wait for it.
            // Unce we get unblocked, perform the appropriate synchronization.
            self.block_thread(
                BlockReason::Join(joined_thread_id),
                None,
                callback!(
                    @capture<'tcx> {
                        joined_thread_id: ThreadId,
                    }
                    |this, unblock: UnblockKind| {
                        assert_eq!(unblock, UnblockKind::Ready);
                        if let Some(data_race) = &mut this.machine.data_race {
                            data_race.thread_joined(&this.machine.threads, joined_thread_id);
                        }
                        interp_ok(())
                    }
                ),
            );
        } else {
            // The thread has already terminated - establish happens-before
            if let Some(data_race) = data_race {
                data_race.thread_joined(self, joined_thread_id);
            }
        }
        interp_ok(())
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

        // Sanity check `join_status`.
        assert!(
            self.threads
                .iter()
                .all(|thread| { !thread.state.is_blocked_on(BlockReason::Join(joined_thread_id)) }),
            "this thread already has threads waiting for its termination"
        );

        self.join_thread(joined_thread_id, data_race)
    }

    /// Set the name of the given thread.
    pub fn set_thread_name(&mut self, thread: ThreadId, new_thread_name: Vec<u8>) {
        self.threads[thread].thread_name = Some(new_thread_name);
    }

    /// Get the name of the given thread.
    pub fn get_thread_name(&self, thread: ThreadId) -> Option<&[u8]> {
        self.threads[thread].thread_name()
    }

    pub fn get_thread_display_name(&self, thread: ThreadId) -> String {
        self.threads[thread].thread_display_name(thread)
    }

    /// Put the thread into the blocked state.
    fn block_thread(
        &mut self,
        reason: BlockReason,
        timeout: Option<Timeout>,
        callback: DynUnblockCallback<'tcx>,
    ) {
        let state = &mut self.threads[self.active_thread].state;
        assert!(state.is_enabled());
        *state = ThreadState::Blocked { reason, timeout, callback }
    }

    /// Change the active thread to some enabled thread.
    fn yield_active_thread(&mut self) {
        // We do not yield immediately, as swapping out the current stack while executing a MIR statement
        // could lead to all sorts of confusion.
        // We should only switch stacks between steps.
        self.yield_active_thread = true;
    }

    /// Get the wait time for the next timeout, or `None` if no timeout is pending.
    fn next_callback_wait_time(&self, clock: &MonotonicClock) -> Option<Duration> {
        self.threads
            .iter()
            .filter_map(|t| {
                match &t.state {
                    ThreadState::Blocked { timeout: Some(timeout), .. } =>
                        Some(timeout.get_wait_time(clock)),
                    _ => None,
                }
            })
            .min()
    }

    /// Decide which action to take next and on which thread.
    ///
    /// The currently implemented scheduling policy is the one that is commonly
    /// used in stateless model checkers such as Loom: run the active thread as
    /// long as we can and switch only when we have to (the active thread was
    /// blocked, terminated, or has explicitly asked to be preempted).
    fn schedule(&mut self, clock: &MonotonicClock) -> InterpResult<'tcx, SchedulingAction> {
        // This thread and the program can keep going.
        if self.threads[self.active_thread].state.is_enabled() && !self.yield_active_thread {
            // The currently active thread is still enabled, just continue with it.
            return interp_ok(SchedulingAction::ExecuteStep);
        }
        // The active thread yielded or got terminated. Let's see if there are any timeouts to take
        // care of. We do this *before* running any other thread, to ensure that timeouts "in the
        // past" fire before any other thread can take an action. This ensures that for
        // `pthread_cond_timedwait`, "an error is returned if [...] the absolute time specified by
        // abstime has already been passed at the time of the call".
        // <https://pubs.opengroup.org/onlinepubs/9699919799/functions/pthread_cond_timedwait.html>
        let potential_sleep_time = self.next_callback_wait_time(clock);
        if potential_sleep_time == Some(Duration::ZERO) {
            return interp_ok(SchedulingAction::ExecuteTimeoutCallback);
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
            if thread.state.is_enabled() {
                info!(
                    "---------- Now executing on thread `{}` (previous: `{}`) ----------------------------------------",
                    self.get_thread_display_name(id),
                    self.get_thread_display_name(self.active_thread)
                );
                self.active_thread = id;
                break;
            }
        }
        self.yield_active_thread = false;
        if self.threads[self.active_thread].state.is_enabled() {
            return interp_ok(SchedulingAction::ExecuteStep);
        }
        // We have not found a thread to execute.
        if self.threads.iter().all(|thread| thread.state.is_terminated()) {
            unreachable!("all threads terminated without the main thread terminating?!");
        } else if let Some(sleep_time) = potential_sleep_time {
            // All threads are currently blocked, but we have unexecuted
            // timeout_callbacks, which may unblock some of the threads. Hence,
            // sleep until the first callback.
            interp_ok(SchedulingAction::Sleep(sleep_time))
        } else {
            throw_machine_stop!(TerminationInfo::Deadlock);
        }
    }
}

impl<'tcx> EvalContextPrivExt<'tcx> for MiriInterpCx<'tcx> {}
trait EvalContextPrivExt<'tcx>: MiriInterpCxExt<'tcx> {
    /// Execute a timeout callback on the callback's thread.
    #[inline]
    fn run_timeout_callback(&mut self) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();
        let mut found_callback = None;
        // Find a blocked thread that has timed out.
        for (id, thread) in this.machine.threads.threads.iter_enumerated_mut() {
            match &thread.state {
                ThreadState::Blocked { timeout: Some(timeout), .. }
                    if timeout.get_wait_time(&this.machine.monotonic_clock) == Duration::ZERO =>
                {
                    let old_state = mem::replace(&mut thread.state, ThreadState::Enabled);
                    let ThreadState::Blocked { callback, .. } = old_state else { unreachable!() };
                    found_callback = Some((id, callback));
                    // Run the fallback (after the loop because borrow-checking).
                    break;
                }
                _ => {}
            }
        }
        if let Some((thread, callback)) = found_callback {
            // This back-and-forth with `set_active_thread` is here because of two
            // design decisions:
            // 1. Make the caller and not the callback responsible for changing
            //    thread.
            // 2. Make the scheduler the only place that can change the active
            //    thread.
            let old_thread = this.machine.threads.set_active_thread_id(thread);
            callback.call(this, UnblockKind::TimedOut)?;
            this.machine.threads.set_active_thread_id(old_thread);
        }
        // found_callback can remain None if the computer's clock
        // was shifted after calling the scheduler and before the call
        // to get_ready_callback (see issue
        // https://github.com/rust-lang/miri/issues/1763). In this case,
        // just do nothing, which effectively just returns to the
        // scheduler.
        interp_ok(())
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
        interp_ok(res)
    }
}

// Public interface to thread management.
impl<'tcx> EvalContextExt<'tcx> for crate::MiriInterpCx<'tcx> {}
pub trait EvalContextExt<'tcx>: crate::MiriInterpCxExt<'tcx> {
    #[inline]
    fn thread_id_try_from(&self, id: impl TryInto<u32>) -> Result<ThreadId, ThreadNotFound> {
        self.eval_context_ref().machine.threads.thread_id_try_from(id)
    }

    /// Get a thread-specific allocation id for the given thread-local static.
    /// If needed, allocate a new one.
    fn get_or_create_thread_local_alloc(
        &mut self,
        def_id: DefId,
    ) -> InterpResult<'tcx, StrictPointer> {
        let this = self.eval_context_mut();
        let tcx = this.tcx;
        if let Some(old_alloc) = this.machine.threads.get_thread_local_alloc_id(def_id) {
            // We already have a thread-specific allocation id for this
            // thread-local static.
            interp_ok(old_alloc)
        } else {
            // We need to allocate a thread-specific allocation id for this
            // thread-local static.
            // First, we compute the initial value for this static.
            if tcx.is_foreign_item(def_id) {
                throw_unsup_format!("foreign thread-local statics are not supported");
            }
            let alloc = this.ctfe_query(|tcx| tcx.eval_static_initializer(def_id))?;
            // We make a full copy of this allocation.
            let mut alloc = alloc.inner().adjust_from_tcx(
                &this.tcx,
                |bytes, align| {
                    interp_ok(MiriAllocBytes::from_bytes(std::borrow::Cow::Borrowed(bytes), align))
                },
                |ptr| this.global_root_pointer(ptr),
            )?;
            // This allocation will be deallocated when the thread dies, so it is not in read-only memory.
            alloc.mutability = Mutability::Mut;
            // Create a fresh allocation with this content.
            let ptr = this.insert_allocation(alloc, MiriMemoryKind::Tls.into())?;
            this.machine.threads.set_thread_local_alloc(def_id, ptr);
            interp_ok(ptr)
        }
    }

    /// Start a regular (non-main) thread.
    #[inline]
    fn start_regular_thread(
        &mut self,
        thread: Option<MPlaceTy<'tcx>>,
        start_routine: Pointer,
        start_abi: ExternAbi,
        func_arg: ImmTy<'tcx>,
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
                &thread_info_place,
            )?;
        }

        // Finally switch to new thread so that we can push the first stackframe.
        // After this all accesses will be treated as occurring in the new thread.
        let old_thread_id = this.machine.threads.set_active_thread_id(new_thread_id);

        // The child inherits its parent's cpu affinity.
        if let Some(cpuset) = this.machine.thread_cpu_affinity.get(&old_thread_id).cloned() {
            this.machine.thread_cpu_affinity.insert(new_thread_id, cpuset);
        }

        // Perform the function pointer load in the new thread frame.
        let instance = this.get_ptr_fn(start_routine)?.as_instance()?;

        // Note: the returned value is currently ignored (see the FIXME in
        // pthread_join in shims/unix/thread.rs) because the Rust standard library does not use
        // it.
        let ret_place = this.allocate(ret_layout, MiriMemoryKind::Machine.into())?;

        this.call_function(
            instance,
            start_abi,
            &[func_arg],
            Some(&ret_place),
            StackPopCleanup::Root { cleanup: true },
        )?;

        // Restore the old active thread frame.
        this.machine.threads.set_active_thread_id(old_thread_id);

        interp_ok(new_thread_id)
    }

    /// Handles thread termination of the active thread: wakes up threads joining on this one,
    /// and deals with the thread's thread-local statics according to `tls_alloc_action`.
    ///
    /// This is called by the eval loop when a thread's on_stack_empty returns `Ready`.
    fn terminate_active_thread(&mut self, tls_alloc_action: TlsAllocAction) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();
        // Mark thread as terminated.
        let thread = this.active_thread_mut();
        assert!(thread.stack.is_empty(), "only threads with an empty stack can be terminated");
        thread.state = ThreadState::Terminated;
        if let Some(ref mut data_race) = this.machine.data_race {
            data_race.thread_terminated(&this.machine.threads);
        }
        // Deallocate TLS.
        let gone_thread = this.active_thread();
        {
            let mut free_tls_statics = Vec::new();
            this.machine.threads.thread_local_allocs.retain(|&(_def_id, thread), &mut alloc_id| {
                if thread != gone_thread {
                    // A different thread, keep this static around.
                    return true;
                }
                // Delete this static from the map and from memory.
                // We cannot free directly here as we cannot use `?` in this context.
                free_tls_statics.push(alloc_id);
                false
            });
            // Now free the TLS statics.
            for ptr in free_tls_statics {
                match tls_alloc_action {
                    TlsAllocAction::Deallocate =>
                        this.deallocate_ptr(ptr.into(), None, MiriMemoryKind::Tls.into())?,
                    TlsAllocAction::Leak =>
                        if let Some(alloc) = ptr.provenance.get_alloc_id() {
                            trace!(
                                "Thread-local static leaked and stored as static root: {:?}",
                                alloc
                            );
                            this.machine.static_roots.push(alloc);
                        },
                }
            }
        }
        // Unblock joining threads.
        let unblock_reason = BlockReason::Join(gone_thread);
        let threads = &this.machine.threads.threads;
        let joining_threads = threads
            .iter_enumerated()
            .filter(|(_, thread)| thread.state.is_blocked_on(unblock_reason))
            .map(|(id, _)| id)
            .collect::<Vec<_>>();
        for thread in joining_threads {
            this.unblock_thread(thread, unblock_reason)?;
        }

        interp_ok(())
    }

    /// Block the current thread, with an optional timeout.
    /// The callback will be invoked when the thread gets unblocked.
    #[inline]
    fn block_thread(
        &mut self,
        reason: BlockReason,
        timeout: Option<(TimeoutClock, TimeoutAnchor, Duration)>,
        callback: DynUnblockCallback<'tcx>,
    ) {
        let this = self.eval_context_mut();
        let timeout = timeout.map(|(clock, anchor, duration)| {
            let anchor = match clock {
                TimeoutClock::RealTime => {
                    assert!(
                        this.machine.communicate(),
                        "cannot have `RealTime` timeout with isolation enabled!"
                    );
                    Timeout::RealTime(match anchor {
                        TimeoutAnchor::Absolute => SystemTime::UNIX_EPOCH,
                        TimeoutAnchor::Relative => SystemTime::now(),
                    })
                }
                TimeoutClock::Monotonic =>
                    Timeout::Monotonic(match anchor {
                        TimeoutAnchor::Absolute => this.machine.monotonic_clock.epoch(),
                        TimeoutAnchor::Relative => this.machine.monotonic_clock.now(),
                    }),
            };
            anchor.add_lossy(duration)
        });
        this.machine.threads.block_thread(reason, timeout, callback);
    }

    /// Put the blocked thread into the enabled state.
    /// Sanity-checks that the thread previously was blocked for the right reason.
    fn unblock_thread(&mut self, thread: ThreadId, reason: BlockReason) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();
        let old_state =
            mem::replace(&mut this.machine.threads.threads[thread].state, ThreadState::Enabled);
        let callback = match old_state {
            ThreadState::Blocked { reason: actual_reason, callback, .. } => {
                assert_eq!(
                    reason, actual_reason,
                    "unblock_thread: thread was blocked for the wrong reason"
                );
                callback
            }
            _ => panic!("unblock_thread: thread was not blocked"),
        };
        // The callback must be executed in the previously blocked thread.
        let old_thread = this.machine.threads.set_active_thread_id(thread);
        callback.call(this, UnblockKind::Ready)?;
        this.machine.threads.set_active_thread_id(old_thread);
        interp_ok(())
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
        interp_ok(())
    }

    #[inline]
    fn join_thread_exclusive(&mut self, joined_thread_id: ThreadId) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();
        this.machine
            .threads
            .join_thread_exclusive(joined_thread_id, this.machine.data_race.as_mut())?;
        interp_ok(())
    }

    #[inline]
    fn active_thread(&self) -> ThreadId {
        let this = self.eval_context_ref();
        this.machine.threads.active_thread()
    }

    #[inline]
    fn active_thread_mut(&mut self) -> &mut Thread<'tcx> {
        let this = self.eval_context_mut();
        this.machine.threads.active_thread_mut()
    }

    #[inline]
    fn active_thread_ref(&self) -> &Thread<'tcx> {
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
    fn active_thread_stack<'a>(&'a self) -> &'a [Frame<'tcx, Provenance, FrameExtra<'tcx>>] {
        let this = self.eval_context_ref();
        this.machine.threads.active_thread_stack()
    }

    #[inline]
    fn active_thread_stack_mut<'a>(
        &'a mut self,
    ) -> &'a mut Vec<Frame<'tcx, Provenance, FrameExtra<'tcx>>> {
        let this = self.eval_context_mut();
        this.machine.threads.active_thread_stack_mut()
    }

    /// Set the name of the current thread. The buffer must not include the null terminator.
    #[inline]
    fn set_thread_name(&mut self, thread: ThreadId, new_thread_name: Vec<u8>) {
        self.eval_context_mut().machine.threads.set_thread_name(thread, new_thread_name);
    }

    #[inline]
    fn get_thread_name<'c>(&'c self, thread: ThreadId) -> Option<&'c [u8]>
    where
        'tcx: 'c,
    {
        self.eval_context_ref().machine.threads.get_thread_name(thread)
    }

    #[inline]
    fn yield_active_thread(&mut self) {
        self.eval_context_mut().machine.threads.yield_active_thread();
    }

    #[inline]
    fn maybe_preempt_active_thread(&mut self) {
        use rand::Rng as _;

        let this = self.eval_context_mut();
        if this.machine.rng.get_mut().random_bool(this.machine.preemption_rate) {
            this.yield_active_thread();
        }
    }

    /// Run the core interpreter loop. Returns only when an interrupt occurs (an error or program
    /// termination).
    fn run_threads(&mut self) -> InterpResult<'tcx, !> {
        let this = self.eval_context_mut();
        loop {
            if CTRL_C_RECEIVED.load(Relaxed) {
                this.machine.handle_abnormal_termination();
                throw_machine_stop!(TerminationInfo::Interrupted);
            }
            match this.machine.threads.schedule(&this.machine.monotonic_clock)? {
                SchedulingAction::ExecuteStep => {
                    if !this.step()? {
                        // See if this thread can do something else.
                        match this.run_on_stack_empty()? {
                            Poll::Pending => {} // keep going
                            Poll::Ready(()) =>
                                this.terminate_active_thread(TlsAllocAction::Deallocate)?,
                        }
                    }
                }
                SchedulingAction::ExecuteTimeoutCallback => {
                    this.run_timeout_callback()?;
                }
                SchedulingAction::Sleep(duration) => {
                    this.machine.monotonic_clock.sleep(duration);
                }
            }
        }
    }
}
