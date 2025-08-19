use std::fmt;
use std::marker::PhantomData;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::job::{ArcJob, StackJob};
use crate::latch::{CountLatch, LatchRef};
use crate::registry::{Registry, WorkerThread};

mod tests;

/// Executes `op` within every thread in the current threadpool. If this is
/// called from a non-Rayon thread, it will execute in the global threadpool.
/// Any attempts to use `join`, `scope`, or parallel iterators will then operate
/// within that threadpool. When the call has completed on each thread, returns
/// a vector containing all of their return values.
///
/// For more information, see the [`ThreadPool::broadcast()`][m] method.
///
/// [m]: struct.ThreadPool.html#method.broadcast
pub fn broadcast<OP, R>(op: OP) -> Vec<R>
where
    OP: Fn(BroadcastContext<'_>) -> R + Sync,
    R: Send,
{
    // We assert that current registry has not terminated.
    unsafe { broadcast_in(op, &Registry::current()) }
}

/// Spawns an asynchronous task on every thread in this thread-pool. This task
/// will run in the implicit, global scope, which means that it may outlast the
/// current stack frame -- therefore, it cannot capture any references onto the
/// stack (you will likely need a `move` closure).
///
/// For more information, see the [`ThreadPool::spawn_broadcast()`][m] method.
///
/// [m]: struct.ThreadPool.html#method.spawn_broadcast
pub fn spawn_broadcast<OP>(op: OP)
where
    OP: Fn(BroadcastContext<'_>) + Send + Sync + 'static,
{
    // We assert that current registry has not terminated.
    unsafe { spawn_broadcast_in(op, &Registry::current()) }
}

/// Provides context to a closure called by `broadcast`.
pub struct BroadcastContext<'a> {
    worker: &'a WorkerThread,

    /// Make sure to prevent auto-traits like `Send` and `Sync`.
    _marker: PhantomData<&'a mut dyn Fn()>,
}

impl<'a> BroadcastContext<'a> {
    pub(super) fn with<R>(f: impl FnOnce(BroadcastContext<'_>) -> R) -> R {
        let worker_thread = WorkerThread::current();
        assert!(!worker_thread.is_null());
        f(BroadcastContext { worker: unsafe { &*worker_thread }, _marker: PhantomData })
    }

    /// Our index amongst the broadcast threads (ranges from `0..self.num_threads()`).
    #[inline]
    pub fn index(&self) -> usize {
        self.worker.index()
    }

    /// The number of threads receiving the broadcast in the thread pool.
    ///
    /// # Future compatibility note
    ///
    /// Future versions of Rayon might vary the number of threads over time, but
    /// this method will always return the number of threads which are actually
    /// receiving your particular `broadcast` call.
    #[inline]
    pub fn num_threads(&self) -> usize {
        self.worker.registry().num_threads()
    }
}

impl<'a> fmt::Debug for BroadcastContext<'a> {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt.debug_struct("BroadcastContext")
            .field("index", &self.index())
            .field("num_threads", &self.num_threads())
            .field("pool_id", &self.worker.registry().id())
            .finish()
    }
}

/// Execute `op` on every thread in the pool. It will be executed on each
/// thread when they have nothing else to do locally, before they try to
/// steal work from other threads. This function will not return until all
/// threads have completed the `op`.
///
/// Unsafe because `registry` must not yet have terminated.
pub(super) unsafe fn broadcast_in<OP, R>(op: OP, registry: &Arc<Registry>) -> Vec<R>
where
    OP: Fn(BroadcastContext<'_>) -> R + Sync,
    R: Send,
{
    let current_thread = WorkerThread::current();
    let current_thread_addr = current_thread.expose_provenance();
    let started = &AtomicBool::new(false);
    let f = move |injected: bool| {
        debug_assert!(injected);

        // Mark as started if we are the thread that initiated that broadcast.
        if current_thread_addr == WorkerThread::current().expose_provenance() {
            started.store(true, Ordering::Relaxed);
        }

        BroadcastContext::with(&op)
    };

    let n_threads = registry.num_threads();
    let current_thread = unsafe { current_thread.as_ref() };
    let tlv = crate::tlv::get();
    let latch = CountLatch::with_count(n_threads, current_thread);
    let jobs: Vec<_> =
        (0..n_threads).map(|_| StackJob::new(tlv, &f, LatchRef::new(&latch))).collect();
    let job_refs = jobs.iter().map(|job| unsafe { job.as_job_ref() });

    registry.inject_broadcast(job_refs);

    let current_thread_job_id = current_thread
        .and_then(|worker| (registry.id() == worker.registry.id()).then(|| worker))
        .map(|worker| unsafe { jobs[worker.index()].as_job_ref() }.id());

    // Wait for all jobs to complete, then collect the results, maybe propagating a panic.
    latch.wait(
        current_thread,
        || started.load(Ordering::Relaxed),
        |job| Some(job.id()) == current_thread_job_id,
    );
    jobs.into_iter().map(|job| unsafe { job.into_result() }).collect()
}

/// Execute `op` on every thread in the pool. It will be executed on each
/// thread when they have nothing else to do locally, before they try to
/// steal work from other threads. This function returns immediately after
/// injecting the jobs.
///
/// Unsafe because `registry` must not yet have terminated.
pub(super) unsafe fn spawn_broadcast_in<OP>(op: OP, registry: &Arc<Registry>)
where
    OP: Fn(BroadcastContext<'_>) + Send + Sync + 'static,
{
    let job = ArcJob::new({
        let registry = Arc::clone(registry);
        move |_| {
            registry.catch_unwind(|| BroadcastContext::with(&op));
            registry.terminate(); // (*) permit registry to terminate now
        }
    });

    let n_threads = registry.num_threads();
    let job_refs = (0..n_threads).map(|_| {
        // Ensure that registry cannot terminate until this job has executed
        // on each thread. This ref is decremented at the (*) above.
        registry.increment_terminate_count();

        ArcJob::as_static_job_ref(&job)
    });

    registry.inject_broadcast(job_refs);
}
