//! Contains support for user-managed thread pools, represented by the
//! the [`ThreadPool`] type (see that struct for details).
//!
//! [`ThreadPool`]: struct.ThreadPool.html

use std::error::Error;
use std::fmt;
use std::sync::Arc;

use crate::broadcast::{self, BroadcastContext};
use crate::registry::{Registry, ThreadSpawn, WorkerThread};
use crate::scope::{do_in_place_scope, do_in_place_scope_fifo};
use crate::{
    Scope, ScopeFifo, ThreadPoolBuildError, ThreadPoolBuilder, join, scope, scope_fifo, spawn,
};

mod tests;

/// Represents a user created [thread-pool].
///
/// Use a [`ThreadPoolBuilder`] to specify the number and/or names of threads
/// in the pool. After calling [`ThreadPoolBuilder::build()`], you can then
/// execute functions explicitly within this [`ThreadPool`] using
/// [`ThreadPool::install()`]. By contrast, top level rayon functions
/// (like `join()`) will execute implicitly within the current thread-pool.
///
///
/// ## Creating a ThreadPool
///
/// ```rust
/// # use rustc_thread_pool as rayon;
/// let pool = rayon::ThreadPoolBuilder::new().num_threads(8).build().unwrap();
/// ```
///
/// [`install()`][`ThreadPool::install()`] executes a closure in one of the `ThreadPool`'s
/// threads. In addition, any other rayon operations called inside of `install()` will also
/// execute in the context of the `ThreadPool`.
///
/// When the `ThreadPool` is dropped, that's a signal for the threads it manages to terminate,
/// they will complete executing any remaining work that you have spawned, and automatically
/// terminate.
///
///
/// [thread-pool]: https://en.wikipedia.org/wiki/Thread_pool
/// [`ThreadPool`]: struct.ThreadPool.html
/// [`ThreadPool::new()`]: struct.ThreadPool.html#method.new
/// [`ThreadPoolBuilder`]: struct.ThreadPoolBuilder.html
/// [`ThreadPoolBuilder::build()`]: struct.ThreadPoolBuilder.html#method.build
/// [`ThreadPool::install()`]: struct.ThreadPool.html#method.install
pub struct ThreadPool {
    registry: Arc<Registry>,
}

impl ThreadPool {
    #[deprecated(note = "Use `ThreadPoolBuilder::build`")]
    #[allow(deprecated)]
    /// Deprecated in favor of `ThreadPoolBuilder::build`.
    pub fn new(configuration: crate::Configuration) -> Result<ThreadPool, Box<dyn Error>> {
        Self::build(configuration.into_builder()).map_err(Box::from)
    }

    pub(super) fn build<S>(
        builder: ThreadPoolBuilder<S>,
    ) -> Result<ThreadPool, ThreadPoolBuildError>
    where
        S: ThreadSpawn,
    {
        let registry = Registry::new(builder)?;
        Ok(ThreadPool { registry })
    }

    /// Executes `op` within the threadpool. Any attempts to use
    /// `join`, `scope`, or parallel iterators will then operate
    /// within that threadpool.
    ///
    /// # Warning: thread-local data
    ///
    /// Because `op` is executing within the Rayon thread-pool,
    /// thread-local data from the current thread will not be
    /// accessible.
    ///
    /// # Warning: execution order
    ///
    /// If the current thread is part of a different thread pool, it will try to
    /// keep busy while the `op` completes in its target pool, similar to
    /// calling [`ThreadPool::yield_now()`] in a loop. Therefore, it may
    /// potentially schedule other tasks to run on the current thread in the
    /// meantime. For example
    ///
    /// ```rust
    /// # use rustc_thread_pool as rayon;
    /// fn main() {
    ///     rayon::ThreadPoolBuilder::new().num_threads(1).build_global().unwrap();
    ///     let pool = rustc_thread_pool::ThreadPoolBuilder::default().build().unwrap();
    ///     let do_it = || {
    ///         print!("one ");
    ///         pool.install(||{});
    ///         print!("two ");
    ///     };
    ///     rayon::join(|| do_it(), || do_it());
    /// }
    /// ```
    ///
    /// Since we configured just one thread in the global pool, one might
    /// expect `do_it()` to run sequentially, producing:
    ///
    /// ```ascii
    /// one two one two
    /// ```
    ///
    /// However each call to `install()` yields implicitly, allowing rayon to
    /// run multiple instances of `do_it()` concurrently on the single, global
    /// thread. The following output would be equally valid:
    ///
    /// ```ascii
    /// one one two two
    /// ```
    ///
    /// # Panics
    ///
    /// If `op` should panic, that panic will be propagated.
    ///
    /// ## Using `install()`
    ///
    /// ```rust
    ///    # use rustc_thread_pool as rayon;
    ///    fn main() {
    ///         let pool = rayon::ThreadPoolBuilder::new().num_threads(8).build().unwrap();
    ///         let n = pool.install(|| fib(20));
    ///         println!("{}", n);
    ///    }
    ///
    ///    fn fib(n: usize) -> usize {
    ///         if n == 0 || n == 1 {
    ///             return n;
    ///         }
    ///         let (a, b) = rayon::join(|| fib(n - 1), || fib(n - 2)); // runs inside of `pool`
    ///         return a + b;
    ///     }
    /// ```
    pub fn install<OP, R>(&self, op: OP) -> R
    where
        OP: FnOnce() -> R + Send,
        R: Send,
    {
        self.registry.in_worker(|_, _| op())
    }

    /// Executes `op` within every thread in the threadpool. Any attempts to use
    /// `join`, `scope`, or parallel iterators will then operate within that
    /// threadpool.
    ///
    /// Broadcasts are executed on each thread after they have exhausted their
    /// local work queue, before they attempt work-stealing from other threads.
    /// The goal of that strategy is to run everywhere in a timely manner
    /// *without* being too disruptive to current work. There may be alternative
    /// broadcast styles added in the future for more or less aggressive
    /// injection, if the need arises.
    ///
    /// # Warning: thread-local data
    ///
    /// Because `op` is executing within the Rayon thread-pool,
    /// thread-local data from the current thread will not be
    /// accessible.
    ///
    /// # Panics
    ///
    /// If `op` should panic on one or more threads, exactly one panic
    /// will be propagated, only after all threads have completed
    /// (or panicked) their own `op`.
    ///
    /// # Examples
    ///
    /// ```
    ///    # use rustc_thread_pool as rayon;
    ///    use std::sync::atomic::{AtomicUsize, Ordering};
    ///
    ///    fn main() {
    ///         let pool = rayon::ThreadPoolBuilder::new().num_threads(5).build().unwrap();
    ///
    ///         // The argument gives context, including the index of each thread.
    ///         let v: Vec<usize> = pool.broadcast(|ctx| ctx.index() * ctx.index());
    ///         assert_eq!(v, &[0, 1, 4, 9, 16]);
    ///
    ///         // The closure can reference the local stack
    ///         let count = AtomicUsize::new(0);
    ///         pool.broadcast(|_| count.fetch_add(1, Ordering::Relaxed));
    ///         assert_eq!(count.into_inner(), 5);
    ///    }
    /// ```
    pub fn broadcast<OP, R>(&self, op: OP) -> Vec<R>
    where
        OP: Fn(BroadcastContext<'_>) -> R + Sync,
        R: Send,
    {
        // We assert that `self.registry` has not terminated.
        unsafe { broadcast::broadcast_in(op, &self.registry) }
    }

    /// Returns the (current) number of threads in the thread pool.
    ///
    /// # Future compatibility note
    ///
    /// Note that unless this thread-pool was created with a
    /// [`ThreadPoolBuilder`] that specifies the number of threads,
    /// then this number may vary over time in future versions (see [the
    /// `num_threads()` method for details][snt]).
    ///
    /// [snt]: struct.ThreadPoolBuilder.html#method.num_threads
    /// [`ThreadPoolBuilder`]: struct.ThreadPoolBuilder.html
    #[inline]
    pub fn current_num_threads(&self) -> usize {
        self.registry.num_threads()
    }

    /// If called from a Rayon worker thread in this thread-pool,
    /// returns the index of that thread; if not called from a Rayon
    /// thread, or called from a Rayon thread that belongs to a
    /// different thread-pool, returns `None`.
    ///
    /// The index for a given thread will not change over the thread's
    /// lifetime. However, multiple threads may share the same index if
    /// they are in distinct thread-pools.
    ///
    /// # Future compatibility note
    ///
    /// Currently, every thread-pool (including the global
    /// thread-pool) has a fixed number of threads, but this may
    /// change in future Rayon versions (see [the `num_threads()` method
    /// for details][snt]). In that case, the index for a
    /// thread would not change during its lifetime, but thread
    /// indices may wind up being reused if threads are terminated and
    /// restarted.
    ///
    /// [snt]: struct.ThreadPoolBuilder.html#method.num_threads
    #[inline]
    pub fn current_thread_index(&self) -> Option<usize> {
        let curr = self.registry.current_thread()?;
        Some(curr.index())
    }

    /// Returns true if the current worker thread currently has "local
    /// tasks" pending. This can be useful as part of a heuristic for
    /// deciding whether to spawn a new task or execute code on the
    /// current thread, particularly in breadth-first
    /// schedulers. However, keep in mind that this is an inherently
    /// racy check, as other worker threads may be actively "stealing"
    /// tasks from our local deque.
    ///
    /// **Background:** Rayon's uses a [work-stealing] scheduler. The
    /// key idea is that each thread has its own [deque] of
    /// tasks. Whenever a new task is spawned -- whether through
    /// `join()`, `Scope::spawn()`, or some other means -- that new
    /// task is pushed onto the thread's *local* deque. Worker threads
    /// have a preference for executing their own tasks; if however
    /// they run out of tasks, they will go try to "steal" tasks from
    /// other threads. This function therefore has an inherent race
    /// with other active worker threads, which may be removing items
    /// from the local deque.
    ///
    /// [work-stealing]: https://en.wikipedia.org/wiki/Work_stealing
    /// [deque]: https://en.wikipedia.org/wiki/Double-ended_queue
    #[inline]
    pub fn current_thread_has_pending_tasks(&self) -> Option<bool> {
        let curr = self.registry.current_thread()?;
        Some(!curr.local_deque_is_empty())
    }

    /// Execute `oper_a` and `oper_b` in the thread-pool and return
    /// the results. Equivalent to `self.install(|| join(oper_a,
    /// oper_b))`.
    pub fn join<A, B, RA, RB>(&self, oper_a: A, oper_b: B) -> (RA, RB)
    where
        A: FnOnce() -> RA + Send,
        B: FnOnce() -> RB + Send,
        RA: Send,
        RB: Send,
    {
        self.install(|| join(oper_a, oper_b))
    }

    /// Creates a scope that executes within this thread-pool.
    /// Equivalent to `self.install(|| scope(...))`.
    ///
    /// See also: [the `scope()` function][scope].
    ///
    /// [scope]: fn.scope.html
    pub fn scope<'scope, OP, R>(&self, op: OP) -> R
    where
        OP: FnOnce(&Scope<'scope>) -> R + Send,
        R: Send,
    {
        self.install(|| scope(op))
    }

    /// Creates a scope that executes within this thread-pool.
    /// Spawns from the same thread are prioritized in relative FIFO order.
    /// Equivalent to `self.install(|| scope_fifo(...))`.
    ///
    /// See also: [the `scope_fifo()` function][scope_fifo].
    ///
    /// [scope_fifo]: fn.scope_fifo.html
    pub fn scope_fifo<'scope, OP, R>(&self, op: OP) -> R
    where
        OP: FnOnce(&ScopeFifo<'scope>) -> R + Send,
        R: Send,
    {
        self.install(|| scope_fifo(op))
    }

    /// Creates a scope that spawns work into this thread-pool.
    ///
    /// See also: [the `in_place_scope()` function][in_place_scope].
    ///
    /// [in_place_scope]: fn.in_place_scope.html
    pub fn in_place_scope<'scope, OP, R>(&self, op: OP) -> R
    where
        OP: FnOnce(&Scope<'scope>) -> R,
    {
        do_in_place_scope(Some(&self.registry), op)
    }

    /// Creates a scope that spawns work into this thread-pool in FIFO order.
    ///
    /// See also: [the `in_place_scope_fifo()` function][in_place_scope_fifo].
    ///
    /// [in_place_scope_fifo]: fn.in_place_scope_fifo.html
    pub fn in_place_scope_fifo<'scope, OP, R>(&self, op: OP) -> R
    where
        OP: FnOnce(&ScopeFifo<'scope>) -> R,
    {
        do_in_place_scope_fifo(Some(&self.registry), op)
    }

    /// Spawns an asynchronous task in this thread-pool. This task will
    /// run in the implicit, global scope, which means that it may outlast
    /// the current stack frame -- therefore, it cannot capture any references
    /// onto the stack (you will likely need a `move` closure).
    ///
    /// See also: [the `spawn()` function defined on scopes][spawn].
    ///
    /// [spawn]: struct.Scope.html#method.spawn
    pub fn spawn<OP>(&self, op: OP)
    where
        OP: FnOnce() + Send + 'static,
    {
        // We assert that `self.registry` has not terminated.
        unsafe { spawn::spawn_in(op, &self.registry) }
    }

    /// Spawns an asynchronous task in this thread-pool. This task will
    /// run in the implicit, global scope, which means that it may outlast
    /// the current stack frame -- therefore, it cannot capture any references
    /// onto the stack (you will likely need a `move` closure).
    ///
    /// See also: [the `spawn_fifo()` function defined on scopes][spawn_fifo].
    ///
    /// [spawn_fifo]: struct.ScopeFifo.html#method.spawn_fifo
    pub fn spawn_fifo<OP>(&self, op: OP)
    where
        OP: FnOnce() + Send + 'static,
    {
        // We assert that `self.registry` has not terminated.
        unsafe { spawn::spawn_fifo_in(op, &self.registry) }
    }

    /// Spawns an asynchronous task on every thread in this thread-pool. This task
    /// will run in the implicit, global scope, which means that it may outlast the
    /// current stack frame -- therefore, it cannot capture any references onto the
    /// stack (you will likely need a `move` closure).
    pub fn spawn_broadcast<OP>(&self, op: OP)
    where
        OP: Fn(BroadcastContext<'_>) + Send + Sync + 'static,
    {
        // We assert that `self.registry` has not terminated.
        unsafe { broadcast::spawn_broadcast_in(op, &self.registry) }
    }

    /// Cooperatively yields execution to Rayon.
    ///
    /// This is similar to the general [`yield_now()`], but only if the current
    /// thread is part of *this* thread pool.
    ///
    /// Returns `Some(Yield::Executed)` if anything was executed, `Some(Yield::Idle)` if
    /// nothing was available, or `None` if the current thread is not part this pool.
    pub fn yield_now(&self) -> Option<Yield> {
        let curr = self.registry.current_thread()?;
        Some(curr.yield_now())
    }

    /// Cooperatively yields execution to local Rayon work.
    ///
    /// This is similar to the general [`yield_local()`], but only if the current
    /// thread is part of *this* thread pool.
    ///
    /// Returns `Some(Yield::Executed)` if anything was executed, `Some(Yield::Idle)` if
    /// nothing was available, or `None` if the current thread is not part this pool.
    pub fn yield_local(&self) -> Option<Yield> {
        let curr = self.registry.current_thread()?;
        Some(curr.yield_local())
    }

    pub(crate) fn wait_until_stopped(self) {
        let registry = Arc::clone(&self.registry);
        drop(self);
        registry.wait_until_stopped();
    }
}

impl Drop for ThreadPool {
    fn drop(&mut self) {
        self.registry.terminate();
    }
}

impl fmt::Debug for ThreadPool {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt.debug_struct("ThreadPool")
            .field("num_threads", &self.current_num_threads())
            .field("id", &self.registry.id())
            .finish()
    }
}

/// If called from a Rayon worker thread, returns the index of that
/// thread within its current pool; if not called from a Rayon thread,
/// returns `None`.
///
/// The index for a given thread will not change over the thread's
/// lifetime. However, multiple threads may share the same index if
/// they are in distinct thread-pools.
///
/// See also: [the `ThreadPool::current_thread_index()` method].
///
/// [m]: struct.ThreadPool.html#method.current_thread_index
///
/// # Future compatibility note
///
/// Currently, every thread-pool (including the global
/// thread-pool) has a fixed number of threads, but this may
/// change in future Rayon versions (see [the `num_threads()` method
/// for details][snt]). In that case, the index for a
/// thread would not change during its lifetime, but thread
/// indices may wind up being reused if threads are terminated and
/// restarted.
///
/// [snt]: struct.ThreadPoolBuilder.html#method.num_threads
#[inline]
pub fn current_thread_index() -> Option<usize> {
    unsafe {
        let curr = WorkerThread::current().as_ref()?;
        Some(curr.index())
    }
}

/// If called from a Rayon worker thread, indicates whether that
/// thread's local deque still has pending tasks. Otherwise, returns
/// `None`. For more information, see [the
/// `ThreadPool::current_thread_has_pending_tasks()` method][m].
///
/// [m]: struct.ThreadPool.html#method.current_thread_has_pending_tasks
#[inline]
pub fn current_thread_has_pending_tasks() -> Option<bool> {
    unsafe {
        let curr = WorkerThread::current().as_ref()?;
        Some(!curr.local_deque_is_empty())
    }
}

/// Cooperatively yields execution to Rayon.
///
/// If the current thread is part of a rayon thread pool, this looks for a
/// single unit of pending work in the pool, then executes it. Completion of
/// that work might include nested work or further work stealing.
///
/// This is similar to [`std::thread::yield_now()`], but does not literally make
/// that call. If you are implementing a polling loop, you may want to also
/// yield to the OS scheduler yourself if no Rayon work was found.
///
/// Returns `Some(Yield::Executed)` if anything was executed, `Some(Yield::Idle)` if
/// nothing was available, or `None` if this thread is not part of any pool at all.
pub fn yield_now() -> Option<Yield> {
    unsafe {
        let thread = WorkerThread::current().as_ref()?;
        Some(thread.yield_now())
    }
}

/// Cooperatively yields execution to local Rayon work.
///
/// If the current thread is part of a rayon thread pool, this looks for a
/// single unit of pending work in this thread's queue, then executes it.
/// Completion of that work might include nested work or further work stealing.
///
/// This is similar to [`yield_now()`], but does not steal from other threads.
///
/// Returns `Some(Yield::Executed)` if anything was executed, `Some(Yield::Idle)` if
/// nothing was available, or `None` if this thread is not part of any pool at all.
pub fn yield_local() -> Option<Yield> {
    unsafe {
        let thread = WorkerThread::current().as_ref()?;
        Some(thread.yield_local())
    }
}

/// Result of [`yield_now()`] or [`yield_local()`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Yield {
    /// Work was found and executed.
    Executed,
    /// No available work was found.
    Idle,
}
