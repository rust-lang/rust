use std::mem;
use std::sync::Arc;

use crate::job::*;
use crate::registry::Registry;
use crate::tlv::Tlv;
use crate::unwind;

/// Puts the task into the Rayon threadpool's job queue in the "static"
/// or "global" scope. Just like a standard thread, this task is not
/// tied to the current stack frame, and hence it cannot hold any
/// references other than those with `'static` lifetime. If you want
/// to spawn a task that references stack data, use [the `scope()`
/// function][scope] to create a scope.
///
/// [scope]: fn.scope.html
///
/// Since tasks spawned with this function cannot hold references into
/// the enclosing stack frame, you almost certainly want to use a
/// `move` closure as their argument (otherwise, the closure will
/// typically hold references to any variables from the enclosing
/// function that you happen to use).
///
/// This API assumes that the closure is executed purely for its
/// side-effects (i.e., it might send messages, modify data protected
/// by a mutex, or some such thing).
///
/// There is no guaranteed order of execution for spawns, given that
/// other threads may steal tasks at any time. However, they are
/// generally prioritized in a LIFO order on the thread from which
/// they were spawned. Other threads always steal from the other end of
/// the deque, like FIFO order. The idea is that "recent" tasks are
/// most likely to be fresh in the local CPU's cache, while other
/// threads can steal older "stale" tasks. For an alternate approach,
/// consider [`spawn_fifo()`] instead.
///
/// [`spawn_fifo()`]: fn.spawn_fifo.html
///
/// # Panic handling
///
/// If this closure should panic, the resulting panic will be
/// propagated to the panic handler registered in the `ThreadPoolBuilder`,
/// if any. See [`ThreadPoolBuilder::panic_handler()`][ph] for more
/// details.
///
/// [ph]: struct.ThreadPoolBuilder.html#method.panic_handler
///
/// # Examples
///
/// This code creates a Rayon task that increments a global counter.
///
/// ```rust
/// # use rustc_thread_pool as rayon;
/// use std::sync::atomic::{AtomicUsize, Ordering, ATOMIC_USIZE_INIT};
///
/// static GLOBAL_COUNTER: AtomicUsize = ATOMIC_USIZE_INIT;
///
/// rayon::spawn(move || {
///     GLOBAL_COUNTER.fetch_add(1, Ordering::SeqCst);
/// });
/// ```
pub fn spawn<F>(func: F)
where
    F: FnOnce() + Send + 'static,
{
    // We assert that current registry has not terminated.
    unsafe { spawn_in(func, &Registry::current()) }
}

/// Spawns an asynchronous job in `registry.`
///
/// Unsafe because `registry` must not yet have terminated.
pub(super) unsafe fn spawn_in<F>(func: F, registry: &Arc<Registry>)
where
    F: FnOnce() + Send + 'static,
{
    // We assert that this does not hold any references (we know
    // this because of the `'static` bound in the interface);
    // moreover, we assert that the code below is not supposed to
    // be able to panic, and hence the data won't leak but will be
    // enqueued into some deque for later execution.
    let abort_guard = unwind::AbortIfPanic; // just in case we are wrong, and code CAN panic
    let job_ref = unsafe { spawn_job(func, registry) };
    registry.inject_or_push(job_ref);
    mem::forget(abort_guard);
}

unsafe fn spawn_job<F>(func: F, registry: &Arc<Registry>) -> JobRef
where
    F: FnOnce() + Send + 'static,
{
    // Ensure that registry cannot terminate until this job has
    // executed. This ref is decremented at the (*) below.
    registry.increment_terminate_count();

    HeapJob::new(Tlv::null(), {
        let registry = Arc::clone(registry);
        move |_| {
            registry.catch_unwind(func);
            registry.terminate(); // (*) permit registry to terminate now
        }
    })
    .into_static_job_ref()
}

/// Fires off a task into the Rayon threadpool in the "static" or
/// "global" scope. Just like a standard thread, this task is not
/// tied to the current stack frame, and hence it cannot hold any
/// references other than those with `'static` lifetime. If you want
/// to spawn a task that references stack data, use [the `scope_fifo()`
/// function](fn.scope_fifo.html) to create a scope.
///
/// The behavior is essentially the same as [the `spawn`
/// function](fn.spawn.html), except that calls from the same thread
/// will be prioritized in FIFO order. This is similar to the now-
/// deprecated [`breadth_first`] option, except the effect is isolated
/// to relative `spawn_fifo` calls, not all threadpool tasks.
///
/// For more details on this design, see Rayon [RFC #1].
///
/// [`breadth_first`]: struct.ThreadPoolBuilder.html#method.breadth_first
/// [RFC #1]: https://github.com/rayon-rs/rfcs/blob/master/accepted/rfc0001-scope-scheduling.md
///
/// # Panic handling
///
/// If this closure should panic, the resulting panic will be
/// propagated to the panic handler registered in the `ThreadPoolBuilder`,
/// if any. See [`ThreadPoolBuilder::panic_handler()`][ph] for more
/// details.
///
/// [ph]: struct.ThreadPoolBuilder.html#method.panic_handler
pub fn spawn_fifo<F>(func: F)
where
    F: FnOnce() + Send + 'static,
{
    // We assert that current registry has not terminated.
    unsafe { spawn_fifo_in(func, &Registry::current()) }
}

/// Spawns an asynchronous FIFO job in `registry.`
///
/// Unsafe because `registry` must not yet have terminated.
pub(super) unsafe fn spawn_fifo_in<F>(func: F, registry: &Arc<Registry>)
where
    F: FnOnce() + Send + 'static,
{
    // We assert that this does not hold any references (we know
    // this because of the `'static` bound in the interface);
    // moreover, we assert that the code below is not supposed to
    // be able to panic, and hence the data won't leak but will be
    // enqueued into some deque for later execution.
    let abort_guard = unwind::AbortIfPanic; // just in case we are wrong, and code CAN panic
    let job_ref = unsafe { spawn_job(func, registry) };

    // If we're in the pool, use our thread's private fifo for this thread to execute
    // in a locally-FIFO order. Otherwise, just use the pool's global injector.
    match registry.current_thread() {
        Some(worker) => unsafe { worker.push_fifo(job_ref) },
        None => registry.inject(job_ref),
    }
    mem::forget(abort_guard);
}

#[cfg(test)]
mod tests;
