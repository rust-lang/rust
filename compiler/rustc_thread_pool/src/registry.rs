use std::cell::Cell;
use std::collections::hash_map::DefaultHasher;
use std::hash::Hasher;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, Once};
use std::{fmt, io, mem, ptr, thread};

use crossbeam_deque::{Injector, Steal, Stealer, Worker};

use crate::job::{JobFifo, JobRef, StackJob};
use crate::latch::{AsCoreLatch, CoreLatch, Latch, LatchRef, LockLatch, OnceLatch, SpinLatch};
use crate::sleep::Sleep;
use crate::tlv::Tlv;
use crate::{
    AcquireThreadHandler, DeadlockHandler, ErrorKind, ExitHandler, PanicHandler,
    ReleaseThreadHandler, StartHandler, ThreadPoolBuildError, ThreadPoolBuilder, Yield, unwind,
};

/// Thread builder used for customization via
/// [`ThreadPoolBuilder::spawn_handler`](struct.ThreadPoolBuilder.html#method.spawn_handler).
pub struct ThreadBuilder {
    name: Option<String>,
    stack_size: Option<usize>,
    worker: Worker<JobRef>,
    stealer: Stealer<JobRef>,
    registry: Arc<Registry>,
    index: usize,
}

impl ThreadBuilder {
    /// Gets the index of this thread in the pool, within `0..num_threads`.
    pub fn index(&self) -> usize {
        self.index
    }

    /// Gets the string that was specified by `ThreadPoolBuilder::name()`.
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    /// Gets the value that was specified by `ThreadPoolBuilder::stack_size()`.
    pub fn stack_size(&self) -> Option<usize> {
        self.stack_size
    }

    /// Executes the main loop for this thread. This will not return until the
    /// thread pool is dropped.
    pub fn run(self) {
        unsafe { main_loop(self) }
    }
}

impl fmt::Debug for ThreadBuilder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ThreadBuilder")
            .field("pool", &self.registry.id())
            .field("index", &self.index)
            .field("name", &self.name)
            .field("stack_size", &self.stack_size)
            .finish()
    }
}

/// Generalized trait for spawning a thread in the `Registry`.
///
/// This trait is pub-in-private -- E0445 forces us to make it public,
/// but we don't actually want to expose these details in the API.
pub trait ThreadSpawn {
    private_decl! {}

    /// Spawn a thread with the `ThreadBuilder` parameters, and then
    /// call `ThreadBuilder::run()`.
    fn spawn(&mut self, thread: ThreadBuilder) -> io::Result<()>;
}

/// Spawns a thread in the "normal" way with `std::thread::Builder`.
///
/// This type is pub-in-private -- E0445 forces us to make it public,
/// but we don't actually want to expose these details in the API.
#[derive(Debug, Default)]
pub struct DefaultSpawn;

impl ThreadSpawn for DefaultSpawn {
    private_impl! {}

    fn spawn(&mut self, thread: ThreadBuilder) -> io::Result<()> {
        let mut b = thread::Builder::new();
        if let Some(name) = thread.name() {
            b = b.name(name.to_owned());
        }
        if let Some(stack_size) = thread.stack_size() {
            b = b.stack_size(stack_size);
        }
        b.spawn(|| thread.run())?;
        Ok(())
    }
}

/// Spawns a thread with a user's custom callback.
///
/// This type is pub-in-private -- E0445 forces us to make it public,
/// but we don't actually want to expose these details in the API.
#[derive(Debug)]
pub struct CustomSpawn<F>(F);

impl<F> CustomSpawn<F>
where
    F: FnMut(ThreadBuilder) -> io::Result<()>,
{
    pub(super) fn new(spawn: F) -> Self {
        CustomSpawn(spawn)
    }
}

impl<F> ThreadSpawn for CustomSpawn<F>
where
    F: FnMut(ThreadBuilder) -> io::Result<()>,
{
    private_impl! {}

    #[inline]
    fn spawn(&mut self, thread: ThreadBuilder) -> io::Result<()> {
        (self.0)(thread)
    }
}

pub struct Registry {
    thread_infos: Vec<ThreadInfo>,
    sleep: Sleep,
    injected_jobs: Injector<JobRef>,
    broadcasts: Mutex<Vec<Worker<JobRef>>>,
    panic_handler: Option<Box<PanicHandler>>,
    pub(crate) deadlock_handler: Option<Box<DeadlockHandler>>,
    start_handler: Option<Box<StartHandler>>,
    exit_handler: Option<Box<ExitHandler>>,
    pub(crate) acquire_thread_handler: Option<Box<AcquireThreadHandler>>,
    pub(crate) release_thread_handler: Option<Box<ReleaseThreadHandler>>,

    // When this latch reaches 0, it means that all work on this
    // registry must be complete. This is ensured in the following ways:
    //
    // - if this is the global registry, there is a ref-count that never
    //   gets released.
    // - if this is a user-created thread-pool, then so long as the thread-pool
    //   exists, it holds a reference.
    // - when we inject a "blocking job" into the registry with `ThreadPool::install()`,
    //   no adjustment is needed; the `ThreadPool` holds the reference, and since we won't
    //   return until the blocking job is complete, that ref will continue to be held.
    // - when `join()` or `scope()` is invoked, similarly, no adjustments are needed.
    //   These are always owned by some other job (e.g., one injected by `ThreadPool::install()`)
    //   and that job will keep the pool alive.
    terminate_count: AtomicUsize,
}

/// ////////////////////////////////////////////////////////////////////////
/// Initialization

static mut THE_REGISTRY: Option<Arc<Registry>> = None;
static THE_REGISTRY_SET: Once = Once::new();

/// Starts the worker threads (if that has not already happened). If
/// initialization has not already occurred, use the default
/// configuration.
pub(super) fn global_registry() -> &'static Arc<Registry> {
    set_global_registry(default_global_registry)
        .or_else(|err| {
            // SAFETY: we only create a shared reference to `THE_REGISTRY` after the `call_once`
            // that initializes it, and there will be no more mutable accesses at all.
            debug_assert!(THE_REGISTRY_SET.is_completed());
            let the_registry = unsafe { &*ptr::addr_of!(THE_REGISTRY) };
            the_registry.as_ref().ok_or(err)
        })
        .expect("The global thread pool has not been initialized.")
}

/// Starts the worker threads (if that has not already happened) with
/// the given builder.
pub(super) fn init_global_registry<S>(
    builder: ThreadPoolBuilder<S>,
) -> Result<&'static Arc<Registry>, ThreadPoolBuildError>
where
    S: ThreadSpawn,
{
    set_global_registry(|| Registry::new(builder))
}

/// Starts the worker threads (if that has not already happened)
/// by creating a registry with the given callback.
fn set_global_registry<F>(registry: F) -> Result<&'static Arc<Registry>, ThreadPoolBuildError>
where
    F: FnOnce() -> Result<Arc<Registry>, ThreadPoolBuildError>,
{
    let mut result = Err(ThreadPoolBuildError::new(ErrorKind::GlobalPoolAlreadyInitialized));

    THE_REGISTRY_SET.call_once(|| {
        result = registry().map(|registry: Arc<Registry>| {
            // SAFETY: this is the only mutable access to `THE_REGISTRY`, thanks to `Once`, and
            // `global_registry()` only takes a shared reference **after** this `call_once`.
            unsafe {
                ptr::addr_of_mut!(THE_REGISTRY).write(Some(registry));
                (*ptr::addr_of!(THE_REGISTRY)).as_ref().unwrap_unchecked()
            }
        })
    });

    result
}

fn default_global_registry() -> Result<Arc<Registry>, ThreadPoolBuildError> {
    let result = Registry::new(ThreadPoolBuilder::new());

    // If we're running in an environment that doesn't support threads at all, we can fall back to
    // using the current thread alone. This is crude, and probably won't work for non-blocking
    // calls like `spawn` or `broadcast_spawn`, but a lot of stuff does work fine.
    //
    // Notably, this allows current WebAssembly targets to work even though their threading support
    // is stubbed out, and we won't have to change anything if they do add real threading.
    let unsupported = matches!(&result, Err(e) if e.is_unsupported());
    if unsupported && WorkerThread::current().is_null() {
        let builder = ThreadPoolBuilder::new().num_threads(1).spawn_handler(|thread| {
            // Rather than starting a new thread, we're just taking over the current thread
            // *without* running the main loop, so we can still return from here.
            // The WorkerThread is leaked, but we never shutdown the global pool anyway.
            let worker_thread = Box::leak(Box::new(WorkerThread::from(thread)));
            let registry = &*worker_thread.registry;
            let index = worker_thread.index;

            unsafe {
                WorkerThread::set_current(worker_thread);

                // let registry know we are ready to do work
                Latch::set(&registry.thread_infos[index].primed);
            }

            Ok(())
        });

        let fallback_result = Registry::new(builder);
        if fallback_result.is_ok() {
            return fallback_result;
        }
    }

    result
}

struct Terminator<'a>(&'a Arc<Registry>);

impl<'a> Drop for Terminator<'a> {
    fn drop(&mut self) {
        self.0.terminate()
    }
}

impl Registry {
    pub(super) fn new<S>(
        mut builder: ThreadPoolBuilder<S>,
    ) -> Result<Arc<Self>, ThreadPoolBuildError>
    where
        S: ThreadSpawn,
    {
        // Soft-limit the number of threads that we can actually support.
        let n_threads = Ord::min(builder.get_num_threads(), crate::max_num_threads());

        let breadth_first = builder.get_breadth_first();

        let (workers, stealers): (Vec<_>, Vec<_>) = (0..n_threads)
            .map(|_| {
                let worker = if breadth_first { Worker::new_fifo() } else { Worker::new_lifo() };

                let stealer = worker.stealer();
                (worker, stealer)
            })
            .unzip();

        let (broadcasts, broadcast_stealers): (Vec<_>, Vec<_>) = (0..n_threads)
            .map(|_| {
                let worker = Worker::new_fifo();
                let stealer = worker.stealer();
                (worker, stealer)
            })
            .unzip();

        let registry = Arc::new(Registry {
            thread_infos: stealers.into_iter().map(ThreadInfo::new).collect(),
            sleep: Sleep::new(n_threads),
            injected_jobs: Injector::new(),
            broadcasts: Mutex::new(broadcasts),
            terminate_count: AtomicUsize::new(1),
            panic_handler: builder.take_panic_handler(),
            deadlock_handler: builder.take_deadlock_handler(),
            start_handler: builder.take_start_handler(),
            exit_handler: builder.take_exit_handler(),
            acquire_thread_handler: builder.take_acquire_thread_handler(),
            release_thread_handler: builder.take_release_thread_handler(),
        });

        // If we return early or panic, make sure to terminate existing threads.
        let t1000 = Terminator(&registry);

        for (index, (worker, stealer)) in workers.into_iter().zip(broadcast_stealers).enumerate() {
            let thread = ThreadBuilder {
                name: builder.get_thread_name(index),
                stack_size: builder.get_stack_size(),
                registry: Arc::clone(&registry),
                worker,
                stealer,
                index,
            };
            if let Err(e) = builder.get_spawn_handler().spawn(thread) {
                return Err(ThreadPoolBuildError::new(ErrorKind::IOError(e)));
            }
        }

        // Returning normally now, without termination.
        mem::forget(t1000);

        Ok(registry)
    }

    pub fn current() -> Arc<Registry> {
        unsafe {
            let worker_thread = WorkerThread::current();
            let registry = if worker_thread.is_null() {
                global_registry()
            } else {
                &(*worker_thread).registry
            };
            Arc::clone(registry)
        }
    }

    /// Returns the number of threads in the current registry. This
    /// is better than `Registry::current().num_threads()` because it
    /// avoids incrementing the `Arc`.
    pub(super) fn current_num_threads() -> usize {
        unsafe {
            let worker_thread = WorkerThread::current();
            if worker_thread.is_null() {
                global_registry().num_threads()
            } else {
                (*worker_thread).registry.num_threads()
            }
        }
    }

    /// Returns the current `WorkerThread` if it's part of this `Registry`.
    pub(super) fn current_thread(&self) -> Option<&WorkerThread> {
        unsafe {
            let worker = WorkerThread::current().as_ref()?;
            if worker.registry().id() == self.id() { Some(worker) } else { None }
        }
    }

    /// Returns an opaque identifier for this registry.
    pub(super) fn id(&self) -> RegistryId {
        // We can rely on `self` not to change since we only ever create
        // registries that are boxed up in an `Arc` (see `new()` above).
        RegistryId { addr: self as *const Self as usize }
    }

    pub(super) fn num_threads(&self) -> usize {
        self.thread_infos.len()
    }

    pub(super) fn catch_unwind(&self, f: impl FnOnce()) {
        if let Err(err) = unwind::halt_unwinding(f) {
            // If there is no handler, or if that handler itself panics, then we abort.
            let abort_guard = unwind::AbortIfPanic;
            if let Some(ref handler) = self.panic_handler {
                handler(err);
                mem::forget(abort_guard);
            }
        }
    }

    /// Waits for the worker threads to get up and running. This is
    /// meant to be used for benchmarking purposes, primarily, so that
    /// you can get more consistent numbers by having everything
    /// "ready to go".
    pub(super) fn wait_until_primed(&self) {
        for info in &self.thread_infos {
            info.primed.wait();
        }
    }

    /// Waits for the worker threads to stop. This is used for testing
    /// -- so we can check that termination actually works.
    pub(super) fn wait_until_stopped(&self) {
        self.release_thread();
        for info in &self.thread_infos {
            info.stopped.wait();
        }
        self.acquire_thread();
    }

    pub(crate) fn acquire_thread(&self) {
        if let Some(ref acquire_thread_handler) = self.acquire_thread_handler {
            acquire_thread_handler();
        }
    }

    pub(crate) fn release_thread(&self) {
        if let Some(ref release_thread_handler) = self.release_thread_handler {
            release_thread_handler();
        }
    }

    /// ////////////////////////////////////////////////////////////////////////
    /// MAIN LOOP
    ///
    /// So long as all of the worker threads are hanging out in their
    /// top-level loop, there is no work to be done.

    /// Push a job into the given `registry`. If we are running on a
    /// worker thread for the registry, this will push onto the
    /// deque. Else, it will inject from the outside (which is slower).
    pub(super) fn inject_or_push(&self, job_ref: JobRef) {
        let worker_thread = WorkerThread::current();
        unsafe {
            if !worker_thread.is_null() && (*worker_thread).registry().id() == self.id() {
                (*worker_thread).push(job_ref);
            } else {
                self.inject(job_ref);
            }
        }
    }

    /// Push a job into the "external jobs" queue; it will be taken by
    /// whatever worker has nothing to do. Use this if you know that
    /// you are not on a worker of this registry.
    pub(super) fn inject(&self, injected_job: JobRef) {
        // It should not be possible for `state.terminate` to be true
        // here. It is only set to true when the user creates (and
        // drops) a `ThreadPool`; and, in that case, they cannot be
        // calling `inject()` later, since they dropped their
        // `ThreadPool`.
        debug_assert_ne!(
            self.terminate_count.load(Ordering::Acquire),
            0,
            "inject() sees state.terminate as true"
        );

        let queue_was_empty = self.injected_jobs.is_empty();

        self.injected_jobs.push(injected_job);
        self.sleep.new_injected_jobs(1, queue_was_empty);
    }

    pub(crate) fn has_injected_job(&self) -> bool {
        !self.injected_jobs.is_empty()
    }

    fn pop_injected_job(&self) -> Option<JobRef> {
        loop {
            match self.injected_jobs.steal() {
                Steal::Success(job) => return Some(job),
                Steal::Empty => return None,
                Steal::Retry => {}
            }
        }
    }

    /// Push a job into each thread's own "external jobs" queue; it will be
    /// executed only on that thread, when it has nothing else to do locally,
    /// before it tries to steal other work.
    ///
    /// **Panics** if not given exactly as many jobs as there are threads.
    pub(super) fn inject_broadcast(&self, injected_jobs: impl ExactSizeIterator<Item = JobRef>) {
        assert_eq!(self.num_threads(), injected_jobs.len());
        {
            let broadcasts = self.broadcasts.lock().unwrap();

            // It should not be possible for `state.terminate` to be true
            // here. It is only set to true when the user creates (and
            // drops) a `ThreadPool`; and, in that case, they cannot be
            // calling `inject_broadcast()` later, since they dropped their
            // `ThreadPool`.
            debug_assert_ne!(
                self.terminate_count.load(Ordering::Acquire),
                0,
                "inject_broadcast() sees state.terminate as true"
            );

            assert_eq!(broadcasts.len(), injected_jobs.len());
            for (worker, job_ref) in broadcasts.iter().zip(injected_jobs) {
                worker.push(job_ref);
            }
        }
        for i in 0..self.num_threads() {
            self.sleep.notify_worker_latch_is_set(i);
        }
    }

    /// If already in a worker-thread of this registry, just execute `op`.
    /// Otherwise, inject `op` in this thread-pool. Either way, block until `op`
    /// completes and return its return value. If `op` panics, that panic will
    /// be propagated as well. The second argument indicates `true` if injection
    /// was performed, `false` if executed directly.
    pub(super) fn in_worker<OP, R>(&self, op: OP) -> R
    where
        OP: FnOnce(&WorkerThread, bool) -> R + Send,
        R: Send,
    {
        unsafe {
            let worker_thread = WorkerThread::current();
            if worker_thread.is_null() {
                self.in_worker_cold(op)
            } else if (*worker_thread).registry().id() != self.id() {
                self.in_worker_cross(&*worker_thread, op)
            } else {
                // Perfectly valid to give them a `&T`: this is the
                // current thread, so we know the data structure won't be
                // invalidated until we return.
                op(&*worker_thread, false)
            }
        }
    }

    #[cold]
    unsafe fn in_worker_cold<OP, R>(&self, op: OP) -> R
    where
        OP: FnOnce(&WorkerThread, bool) -> R + Send,
        R: Send,
    {
        thread_local!(static LOCK_LATCH: LockLatch = LockLatch::new());

        LOCK_LATCH.with(|l| {
            // This thread isn't a member of *any* thread pool, so just block.
            debug_assert!(WorkerThread::current().is_null());
            let job = StackJob::new(
                Tlv::null(),
                |injected| {
                    let worker_thread = WorkerThread::current();
                    assert!(injected && !worker_thread.is_null());
                    op(unsafe { &*worker_thread }, true)
                },
                LatchRef::new(l),
            );
            self.inject(unsafe { job.as_job_ref() });
            self.release_thread();
            job.latch.wait_and_reset(); // Make sure we can use the same latch again next time.
            self.acquire_thread();

            unsafe { job.into_result() }
        })
    }

    #[cold]
    unsafe fn in_worker_cross<OP, R>(&self, current_thread: &WorkerThread, op: OP) -> R
    where
        OP: FnOnce(&WorkerThread, bool) -> R + Send,
        R: Send,
    {
        // This thread is a member of a different pool, so let it process
        // other work while waiting for this `op` to complete.
        debug_assert!(current_thread.registry().id() != self.id());
        let latch = SpinLatch::cross(current_thread);
        let job = StackJob::new(
            Tlv::null(),
            |injected| {
                let worker_thread = WorkerThread::current();
                assert!(injected && !worker_thread.is_null());
                op(unsafe { &*worker_thread }, true)
            },
            latch,
        );
        self.inject(unsafe { job.as_job_ref() });
        unsafe { current_thread.wait_until(&job.latch) };
        unsafe { job.into_result() }
    }

    /// Increments the terminate counter. This increment should be
    /// balanced by a call to `terminate`, which will decrement. This
    /// is used when spawning asynchronous work, which needs to
    /// prevent the registry from terminating so long as it is active.
    ///
    /// Note that blocking functions such as `join` and `scope` do not
    /// need to concern themselves with this fn; their context is
    /// responsible for ensuring the current thread-pool will not
    /// terminate until they return.
    ///
    /// The global thread-pool always has an outstanding reference
    /// (the initial one). Custom thread-pools have one outstanding
    /// reference that is dropped when the `ThreadPool` is dropped:
    /// since installing the thread-pool blocks until any joins/scopes
    /// complete, this ensures that joins/scopes are covered.
    ///
    /// The exception is `::spawn()`, which can create a job outside
    /// of any blocking scope. In that case, the job itself holds a
    /// terminate count and is responsible for invoking `terminate()`
    /// when finished.
    pub(super) fn increment_terminate_count(&self) {
        let previous = self.terminate_count.fetch_add(1, Ordering::AcqRel);
        debug_assert!(previous != 0, "registry ref count incremented from zero");
        assert!(previous != usize::MAX, "overflow in registry ref count");
    }

    /// Signals that the thread-pool which owns this registry has been
    /// dropped. The worker threads will gradually terminate, once any
    /// extant work is completed.
    pub(super) fn terminate(&self) {
        if self.terminate_count.fetch_sub(1, Ordering::AcqRel) == 1 {
            for (i, thread_info) in self.thread_infos.iter().enumerate() {
                unsafe { OnceLatch::set_and_tickle_one(&thread_info.terminate, self, i) };
            }
        }
    }

    /// Notify the worker that the latch they are sleeping on has been "set".
    pub(super) fn notify_worker_latch_is_set(&self, target_worker_index: usize) {
        self.sleep.notify_worker_latch_is_set(target_worker_index);
    }
}

/// Mark a Rayon worker thread as blocked. This triggers the deadlock handler
/// if no other worker thread is active
#[inline]
pub fn mark_blocked() {
    let worker_thread = WorkerThread::current();
    assert!(!worker_thread.is_null());
    unsafe {
        let registry = &(*worker_thread).registry;
        registry.sleep.mark_blocked(&registry.deadlock_handler)
    }
}

/// Mark a previously blocked Rayon worker thread as unblocked
#[inline]
pub fn mark_unblocked(registry: &Registry) {
    registry.sleep.mark_unblocked()
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct RegistryId {
    addr: usize,
}

struct ThreadInfo {
    /// Latch set once thread has started and we are entering into the
    /// main loop. Used to wait for worker threads to become primed,
    /// primarily of interest for benchmarking.
    primed: LockLatch,

    /// Latch is set once worker thread has completed. Used to wait
    /// until workers have stopped; only used for tests.
    stopped: LockLatch,

    /// The latch used to signal that terminated has been requested.
    /// This latch is *set* by the `terminate` method on the
    /// `Registry`, once the registry's main "terminate" counter
    /// reaches zero.
    terminate: OnceLatch,

    /// the "stealer" half of the worker's deque
    stealer: Stealer<JobRef>,
}

impl ThreadInfo {
    fn new(stealer: Stealer<JobRef>) -> ThreadInfo {
        ThreadInfo {
            primed: LockLatch::new(),
            stopped: LockLatch::new(),
            terminate: OnceLatch::new(),
            stealer,
        }
    }
}

/// ////////////////////////////////////////////////////////////////////////
/// WorkerThread identifiers

pub(super) struct WorkerThread {
    /// the "worker" half of our local deque
    worker: Worker<JobRef>,

    /// the "stealer" half of the worker's broadcast deque
    stealer: Stealer<JobRef>,

    /// local queue used for `spawn_fifo` indirection
    fifo: JobFifo,

    pub(crate) index: usize,

    /// A weak random number generator.
    rng: XorShift64Star,

    pub(crate) registry: Arc<Registry>,
}

// This is a bit sketchy, but basically: the WorkerThread is
// allocated on the stack of the worker on entry and stored into this
// thread local variable. So it will remain valid at least until the
// worker is fully unwound. Using an unsafe pointer avoids the need
// for a RefCell<T> etc.
thread_local! {
    static WORKER_THREAD_STATE: Cell<*const WorkerThread> = const { Cell::new(ptr::null()) };
}

impl From<ThreadBuilder> for WorkerThread {
    fn from(thread: ThreadBuilder) -> Self {
        Self {
            worker: thread.worker,
            stealer: thread.stealer,
            fifo: JobFifo::new(),
            index: thread.index,
            rng: XorShift64Star::new(),
            registry: thread.registry,
        }
    }
}

impl Drop for WorkerThread {
    fn drop(&mut self) {
        // Undo `set_current`
        WORKER_THREAD_STATE.with(|t| {
            assert!(t.get().eq(&(self as *const _)));
            t.set(ptr::null());
        });
    }
}

impl WorkerThread {
    /// Gets the `WorkerThread` index for the current thread; returns
    /// NULL if this is not a worker thread. This pointer is valid
    /// anywhere on the current thread.
    #[inline]
    pub(super) fn current() -> *const WorkerThread {
        WORKER_THREAD_STATE.with(Cell::get)
    }

    /// Sets `self` as the worker thread index for the current thread.
    /// This is done during worker thread startup.
    unsafe fn set_current(thread: *const WorkerThread) {
        WORKER_THREAD_STATE.with(|t| {
            assert!(t.get().is_null());
            t.set(thread);
        });
    }

    /// Returns the registry that owns this worker thread.
    #[inline]
    pub(super) fn registry(&self) -> &Arc<Registry> {
        &self.registry
    }

    /// Our index amongst the worker threads (ranges from `0..self.num_threads()`).
    #[inline]
    pub(super) fn index(&self) -> usize {
        self.index
    }

    #[inline]
    pub(super) unsafe fn push(&self, job: JobRef) {
        let queue_was_empty = self.worker.is_empty();
        self.worker.push(job);
        self.registry.sleep.new_internal_jobs(1, queue_was_empty);
    }

    #[inline]
    pub(super) unsafe fn push_fifo(&self, job: JobRef) {
        unsafe { self.push(self.fifo.push(job)) };
    }

    #[inline]
    pub(super) fn local_deque_is_empty(&self) -> bool {
        self.worker.is_empty()
    }

    /// Attempts to obtain a "local" job -- typically this means
    /// popping from the top of the stack, though if we are configured
    /// for breadth-first execution, it would mean dequeuing from the
    /// bottom.
    #[inline]
    pub(super) fn take_local_job(&self) -> Option<JobRef> {
        let popped_job = self.worker.pop();

        if popped_job.is_some() {
            return popped_job;
        }

        loop {
            match self.stealer.steal() {
                Steal::Success(job) => return Some(job),
                Steal::Empty => return None,
                Steal::Retry => {}
            }
        }
    }

    pub(super) fn has_injected_job(&self) -> bool {
        !self.stealer.is_empty() || self.registry.has_injected_job()
    }

    /// Wait until the latch is set. Try to keep busy by popping and
    /// stealing tasks as necessary.
    #[inline]
    pub(super) unsafe fn wait_until<L: AsCoreLatch + ?Sized>(&self, latch: &L) {
        let latch = latch.as_core_latch();
        if !latch.probe() {
            unsafe { self.wait_until_cold(latch) };
        }
    }

    #[cold]
    unsafe fn wait_until_cold(&self, latch: &CoreLatch) {
        // the code below should swallow all panics and hence never
        // unwind; but if something does wrong, we want to abort,
        // because otherwise other code in rayon may assume that the
        // latch has been signaled, and that can lead to random memory
        // accesses, which would be *very bad*
        let abort_guard = unwind::AbortIfPanic;

        'outer: while !latch.probe() {
            // Check for local work *before* we start marking ourself idle,
            // especially to avoid modifying shared sleep state.
            if let Some(job) = self.take_local_job() {
                unsafe { self.execute(job) };
                continue;
            }

            let mut idle_state = self.registry.sleep.start_looking(self.index);
            while !latch.probe() {
                if let Some(job) = self.find_work() {
                    self.registry.sleep.work_found();
                    unsafe { self.execute(job) };
                    // The job might have injected local work, so go back to the outer loop.
                    continue 'outer;
                } else {
                    self.registry.sleep.no_work_found(&mut idle_state, latch, &self)
                }
            }

            // If we were sleepy, we are not anymore. We "found work" --
            // whatever the surrounding thread was doing before it had to wait.
            self.registry.sleep.work_found();
            break;
        }

        mem::forget(abort_guard); // successful execution, do not abort
    }

    unsafe fn wait_until_out_of_work(&self) {
        debug_assert_eq!(self as *const _, WorkerThread::current());
        let registry = &*self.registry;
        let index = self.index;

        registry.acquire_thread();
        unsafe { self.wait_until(&registry.thread_infos[index].terminate) };

        // Should not be any work left in our queue.
        debug_assert!(self.take_local_job().is_none());

        // Let registry know we are done
        unsafe { Latch::set(&registry.thread_infos[index].stopped) };
    }

    fn find_work(&self) -> Option<JobRef> {
        // Try to find some work to do. We give preference first
        // to things in our local deque, then in other workers
        // deques, and finally to injected jobs from the
        // outside. The idea is to finish what we started before
        // we take on something new.
        self.take_local_job().or_else(|| self.steal()).or_else(|| self.registry.pop_injected_job())
    }

    pub(super) fn yield_now(&self) -> Yield {
        match self.find_work() {
            Some(job) => unsafe {
                self.execute(job);
                Yield::Executed
            },
            None => Yield::Idle,
        }
    }

    pub(super) fn yield_local(&self) -> Yield {
        match self.take_local_job() {
            Some(job) => unsafe {
                self.execute(job);
                Yield::Executed
            },
            None => Yield::Idle,
        }
    }

    #[inline]
    pub(super) unsafe fn execute(&self, job: JobRef) {
        unsafe { job.execute() };
    }

    /// Try to steal a single job and return it.
    ///
    /// This should only be done as a last resort, when there is no
    /// local work to do.
    fn steal(&self) -> Option<JobRef> {
        // we only steal when we don't have any work to do locally
        debug_assert!(self.local_deque_is_empty());

        // otherwise, try to steal
        let thread_infos = &self.registry.thread_infos.as_slice();
        let num_threads = thread_infos.len();
        if num_threads <= 1 {
            return None;
        }

        loop {
            let mut retry = false;
            let start = self.rng.next_usize(num_threads);
            let job = (start..num_threads)
                .chain(0..start)
                .filter(move |&i| i != self.index)
                .find_map(|victim_index| {
                    let victim = &thread_infos[victim_index];
                    match victim.stealer.steal() {
                        Steal::Success(job) => Some(job),
                        Steal::Empty => None,
                        Steal::Retry => {
                            retry = true;
                            None
                        }
                    }
                });
            if job.is_some() || !retry {
                return job;
            }
        }
    }
}

/// ////////////////////////////////////////////////////////////////////////

unsafe fn main_loop(thread: ThreadBuilder) {
    let worker_thread = &WorkerThread::from(thread);
    unsafe { WorkerThread::set_current(worker_thread) };
    let registry = &*worker_thread.registry;
    let index = worker_thread.index;

    // let registry know we are ready to do work
    unsafe { Latch::set(&registry.thread_infos[index].primed) };

    // Worker threads should not panic. If they do, just abort, as the
    // internal state of the threadpool is corrupted. Note that if
    // **user code** panics, we should catch that and redirect.
    let abort_guard = unwind::AbortIfPanic;

    // Inform a user callback that we started a thread.
    if let Some(ref handler) = registry.start_handler {
        registry.catch_unwind(|| handler(index));
    }

    unsafe { worker_thread.wait_until_out_of_work() };

    // Normal termination, do not abort.
    mem::forget(abort_guard);

    // Inform a user callback that we exited a thread.
    if let Some(ref handler) = registry.exit_handler {
        registry.catch_unwind(|| handler(index));
        // We're already exiting the thread, there's nothing else to do.
    }

    registry.release_thread();
}

/// If already in a worker-thread, just execute `op`. Otherwise,
/// execute `op` in the default thread-pool. Either way, block until
/// `op` completes and return its return value. If `op` panics, that
/// panic will be propagated as well. The second argument indicates
/// `true` if injection was performed, `false` if executed directly.
pub(super) fn in_worker<OP, R>(op: OP) -> R
where
    OP: FnOnce(&WorkerThread, bool) -> R + Send,
    R: Send,
{
    unsafe {
        let owner_thread = WorkerThread::current();
        if !owner_thread.is_null() {
            // Perfectly valid to give them a `&T`: this is the
            // current thread, so we know the data structure won't be
            // invalidated until we return.
            op(&*owner_thread, false)
        } else {
            global_registry().in_worker(op)
        }
    }
}

/// [xorshift*] is a fast pseudorandom number generator which will
/// even tolerate weak seeding, as long as it's not zero.
///
/// [xorshift*]: https://en.wikipedia.org/wiki/Xorshift#xorshift*
struct XorShift64Star {
    state: Cell<u64>,
}

impl XorShift64Star {
    fn new() -> Self {
        // Any non-zero seed will do -- this uses the hash of a global counter.
        let mut seed = 0;
        while seed == 0 {
            let mut hasher = DefaultHasher::new();
            static COUNTER: AtomicUsize = AtomicUsize::new(0);
            hasher.write_usize(COUNTER.fetch_add(1, Ordering::Relaxed));
            seed = hasher.finish();
        }

        XorShift64Star { state: Cell::new(seed) }
    }

    fn next(&self) -> u64 {
        let mut x = self.state.get();
        debug_assert_ne!(x, 0);
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state.set(x);
        x.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    /// Return a value from `0..n`.
    fn next_usize(&self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}
