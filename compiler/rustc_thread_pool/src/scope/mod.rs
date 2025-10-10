//! Methods for custom fork-join scopes, created by the [`scope()`]
//! and [`in_place_scope()`] functions. These are a more flexible alternative to [`join()`].
//!
//! [`scope()`]: fn.scope.html
//! [`in_place_scope()`]: fn.in_place_scope.html
//! [`join()`]: ../join/join.fn.html

use std::any::Any;
use std::collections::HashSet;
use std::marker::PhantomData;
use std::mem::ManuallyDrop;
use std::sync::atomic::{AtomicPtr, Ordering};
use std::sync::{Arc, Mutex};
use std::{fmt, ptr};

use crate::broadcast::BroadcastContext;
use crate::job::{ArcJob, HeapJob, JobFifo, JobRef, JobRefId};
use crate::latch::{CountLatch, Latch};
use crate::registry::{Registry, WorkerThread, global_registry, in_worker};
use crate::tlv::{self, Tlv};
use crate::unwind;

#[cfg(test)]
mod tests;

/// Represents a fork-join scope which can be used to spawn any number of tasks.
/// See [`scope()`] for more information.
///
///[`scope()`]: fn.scope.html
pub struct Scope<'scope> {
    base: ScopeBase<'scope>,
}

/// Represents a fork-join scope which can be used to spawn any number of tasks.
/// Those spawned from the same thread are prioritized in relative FIFO order.
/// See [`scope_fifo()`] for more information.
///
///[`scope_fifo()`]: fn.scope_fifo.html
pub struct ScopeFifo<'scope> {
    base: ScopeBase<'scope>,
    fifos: Vec<JobFifo>,
}

struct ScopeBase<'scope> {
    /// thread registry where `scope()` was executed or where `in_place_scope()`
    /// should spawn jobs.
    registry: Arc<Registry>,

    /// if some job panicked, the error is stored here; it will be
    /// propagated to the one who created the scope
    panic: AtomicPtr<Box<dyn Any + Send + 'static>>,

    /// latch to track job counts
    job_completed_latch: CountLatch,

    /// Jobs that have been spawned, but not yet started.
    #[allow(rustc::default_hash_types)]
    pending_jobs: Mutex<HashSet<JobRefId>>,

    /// The worker which will wait on scope completion, if any.
    worker: Option<usize>,

    /// You can think of a scope as containing a list of closures to execute,
    /// all of which outlive `'scope`. They're not actually required to be
    /// `Sync`, but it's still safe to let the `Scope` implement `Sync` because
    /// the closures are only *moved* across threads to be executed.
    #[allow(clippy::type_complexity)]
    marker: PhantomData<Box<dyn FnOnce(&Scope<'scope>) + Send + Sync + 'scope>>,

    /// The TLV at the scope's creation. Used to set the TLV for spawned jobs.
    tlv: Tlv,
}

/// Creates a "fork-join" scope `s` and invokes the closure with a
/// reference to `s`. This closure can then spawn asynchronous tasks
/// into `s`. Those tasks may run asynchronously with respect to the
/// closure; they may themselves spawn additional tasks into `s`. When
/// the closure returns, it will block until all tasks that have been
/// spawned into `s` complete.
///
/// `scope()` is a more flexible building block compared to `join()`,
/// since a loop can be used to spawn any number of tasks without
/// recursing. However, that flexibility comes at a performance price:
/// tasks spawned using `scope()` must be allocated onto the heap,
/// whereas `join()` can make exclusive use of the stack. **Prefer
/// `join()` (or, even better, parallel iterators) where possible.**
///
/// # Example
///
/// The Rayon `join()` function launches two closures and waits for them
/// to stop. One could implement `join()` using a scope like so, although
/// it would be less efficient than the real implementation:
///
/// ```rust
/// # use rustc_thread_pool as rayon;
/// pub fn join<A,B,RA,RB>(oper_a: A, oper_b: B) -> (RA, RB)
///     where A: FnOnce() -> RA + Send,
///           B: FnOnce() -> RB + Send,
///           RA: Send,
///           RB: Send,
/// {
///     let mut result_a: Option<RA> = None;
///     let mut result_b: Option<RB> = None;
///     rayon::scope(|s| {
///         s.spawn(|_| result_a = Some(oper_a()));
///         s.spawn(|_| result_b = Some(oper_b()));
///     });
///     (result_a.unwrap(), result_b.unwrap())
/// }
/// ```
///
/// # A note on threading
///
/// The closure given to `scope()` executes in the Rayon thread-pool,
/// as do those given to `spawn()`. This means that you can't access
/// thread-local variables (well, you can, but they may have
/// unexpected values).
///
/// # Task execution
///
/// Task execution potentially starts as soon as `spawn()` is called.
/// The task will end sometime before `scope()` returns. Note that the
/// *closure* given to scope may return much earlier. In general
/// the lifetime of a scope created like `scope(body)` goes something like this:
///
/// - Scope begins when `scope(body)` is called
/// - Scope body `body()` is invoked
///     - Scope tasks may be spawned
/// - Scope body returns
/// - Scope tasks execute, possibly spawning more tasks
/// - Once all tasks are done, scope ends and `scope()` returns
///
/// To see how and when tasks are joined, consider this example:
///
/// ```rust
/// # use rustc_thread_pool as rayon;
/// // point start
/// rayon::scope(|s| {
///     s.spawn(|s| { // task s.1
///         s.spawn(|s| { // task s.1.1
///             rayon::scope(|t| {
///                 t.spawn(|_| ()); // task t.1
///                 t.spawn(|_| ()); // task t.2
///             });
///         });
///     });
///     s.spawn(|s| { // task s.2
///     });
///     // point mid
/// });
/// // point end
/// ```
///
/// The various tasks that are run will execute roughly like so:
///
/// ```notrust
/// | (start)
/// |
/// | (scope `s` created)
/// +-----------------------------------------------+ (task s.2)
/// +-------+ (task s.1)                            |
/// |       |                                       |
/// |       +---+ (task s.1.1)                      |
/// |       |   |                                   |
/// |       |   | (scope `t` created)               |
/// |       |   +----------------+ (task t.2)       |
/// |       |   +---+ (task t.1) |                  |
/// | (mid) |   |   |            |                  |
/// :       |   + <-+------------+ (scope `t` ends) |
/// :       |   |                                   |
/// |<------+---+-----------------------------------+ (scope `s` ends)
/// |
/// | (end)
/// ```
///
/// The point here is that everything spawned into scope `s` will
/// terminate (at latest) at the same point -- right before the
/// original call to `rayon::scope` returns. This includes new
/// subtasks created by other subtasks (e.g., task `s.1.1`). If a new
/// scope is created (such as `t`), the things spawned into that scope
/// will be joined before that scope returns, which in turn occurs
/// before the creating task (task `s.1.1` in this case) finishes.
///
/// There is no guaranteed order of execution for spawns in a scope,
/// given that other threads may steal tasks at any time. However, they
/// are generally prioritized in a LIFO order on the thread from which
/// they were spawned. So in this example, absent any stealing, we can
/// expect `s.2` to execute before `s.1`, and `t.2` before `t.1`. Other
/// threads always steal from the other end of the deque, like FIFO
/// order. The idea is that "recent" tasks are most likely to be fresh
/// in the local CPU's cache, while other threads can steal older
/// "stale" tasks. For an alternate approach, consider
/// [`scope_fifo()`] instead.
///
/// [`scope_fifo()`]: fn.scope_fifo.html
///
/// # Accessing stack data
///
/// In general, spawned tasks may access stack data in place that
/// outlives the scope itself. Other data must be fully owned by the
/// spawned task.
///
/// ```rust
/// # use rustc_thread_pool as rayon;
/// let ok: Vec<i32> = vec![1, 2, 3];
/// rayon::scope(|s| {
///     let bad: Vec<i32> = vec![4, 5, 6];
///     s.spawn(|_| {
///         // We can access `ok` because outlives the scope `s`.
///         println!("ok: {:?}", ok);
///
///         // If we just try to use `bad` here, the closure will borrow `bad`
///         // (because we are just printing it out, and that only requires a
///         // borrow), which will result in a compilation error. Read on
///         // for options.
///         // println!("bad: {:?}", bad);
///    });
/// });
/// ```
///
/// As the comments example above suggest, to reference `bad` we must
/// take ownership of it. One way to do this is to detach the closure
/// from the surrounding stack frame, using the `move` keyword. This
/// will cause it to take ownership of *all* the variables it touches,
/// in this case including both `ok` *and* `bad`:
///
/// ```rust
/// # use rustc_thread_pool as rayon;
/// let ok: Vec<i32> = vec![1, 2, 3];
/// rayon::scope(|s| {
///     let bad: Vec<i32> = vec![4, 5, 6];
///     s.spawn(move |_| {
///         println!("ok: {:?}", ok);
///         println!("bad: {:?}", bad);
///     });
///
///     // That closure is fine, but now we can't use `ok` anywhere else,
///     // since it is owned by the previous task:
///     // s.spawn(|_| println!("ok: {:?}", ok));
/// });
/// ```
///
/// While this works, it could be a problem if we want to use `ok` elsewhere.
/// There are two choices. We can keep the closure as a `move` closure, but
/// instead of referencing the variable `ok`, we create a shadowed variable that
/// is a borrow of `ok` and capture *that*:
///
/// ```rust
/// # use rustc_thread_pool as rayon;
/// let ok: Vec<i32> = vec![1, 2, 3];
/// rayon::scope(|s| {
///     let bad: Vec<i32> = vec![4, 5, 6];
///     let ok: &Vec<i32> = &ok; // shadow the original `ok`
///     s.spawn(move |_| {
///         println!("ok: {:?}", ok); // captures the shadowed version
///         println!("bad: {:?}", bad);
///     });
///
///     // Now we too can use the shadowed `ok`, since `&Vec<i32>` references
///     // can be shared freely. Note that we need a `move` closure here though,
///     // because otherwise we'd be trying to borrow the shadowed `ok`,
///     // and that doesn't outlive `scope`.
///     s.spawn(move |_| println!("ok: {:?}", ok));
/// });
/// ```
///
/// Another option is not to use the `move` keyword but instead to take ownership
/// of individual variables:
///
/// ```rust
/// # use rustc_thread_pool as rayon;
/// let ok: Vec<i32> = vec![1, 2, 3];
/// rayon::scope(|s| {
///     let bad: Vec<i32> = vec![4, 5, 6];
///     s.spawn(|_| {
///         // Transfer ownership of `bad` into a local variable (also named `bad`).
///         // This will force the closure to take ownership of `bad` from the environment.
///         let bad = bad;
///         println!("ok: {:?}", ok); // `ok` is only borrowed.
///         println!("bad: {:?}", bad); // refers to our local variable, above.
///     });
///
///     s.spawn(|_| println!("ok: {:?}", ok)); // we too can borrow `ok`
/// });
/// ```
///
/// # Panics
///
/// If a panic occurs, either in the closure given to `scope()` or in
/// any of the spawned jobs, that panic will be propagated and the
/// call to `scope()` will panic. If multiple panics occurs, it is
/// non-deterministic which of their panic values will propagate.
/// Regardless, once a task is spawned using `scope.spawn()`, it will
/// execute, even if the spawning task should later panic. `scope()`
/// returns once all spawned jobs have completed, and any panics are
/// propagated at that point.
pub fn scope<'scope, OP, R>(op: OP) -> R
where
    OP: FnOnce(&Scope<'scope>) -> R + Send,
    R: Send,
{
    in_worker(|owner_thread, _| {
        let scope = Scope::<'scope>::new(Some(owner_thread), None);
        scope.base.complete(Some(owner_thread), || op(&scope))
    })
}

/// Creates a "fork-join" scope `s` with FIFO order, and invokes the
/// closure with a reference to `s`. This closure can then spawn
/// asynchronous tasks into `s`. Those tasks may run asynchronously with
/// respect to the closure; they may themselves spawn additional tasks
/// into `s`. When the closure returns, it will block until all tasks
/// that have been spawned into `s` complete.
///
/// # Task execution
///
/// Tasks in a `scope_fifo()` run similarly to [`scope()`], but there's a
/// difference in the order of execution. Consider a similar example:
///
/// [`scope()`]: fn.scope.html
///
/// ```rust
/// # use rustc_thread_pool as rayon;
/// // point start
/// rayon::scope_fifo(|s| {
///     s.spawn_fifo(|s| { // task s.1
///         s.spawn_fifo(|s| { // task s.1.1
///             rayon::scope_fifo(|t| {
///                 t.spawn_fifo(|_| ()); // task t.1
///                 t.spawn_fifo(|_| ()); // task t.2
///             });
///         });
///     });
///     s.spawn_fifo(|s| { // task s.2
///     });
///     // point mid
/// });
/// // point end
/// ```
///
/// The various tasks that are run will execute roughly like so:
///
/// ```notrust
/// | (start)
/// |
/// | (FIFO scope `s` created)
/// +--------------------+ (task s.1)
/// +-------+ (task s.2) |
/// |       |            +---+ (task s.1.1)
/// |       |            |   |
/// |       |            |   | (FIFO scope `t` created)
/// |       |            |   +----------------+ (task t.1)
/// |       |            |   +---+ (task t.2) |
/// | (mid) |            |   |   |            |
/// :       |            |   + <-+------------+ (scope `t` ends)
/// :       |            |   |
/// |<------+------------+---+ (scope `s` ends)
/// |
/// | (end)
/// ```
///
/// Under `scope_fifo()`, the spawns are prioritized in a FIFO order on
/// the thread from which they were spawned, as opposed to `scope()`'s
/// LIFO. So in this example, we can expect `s.1` to execute before
/// `s.2`, and `t.1` before `t.2`. Other threads also steal tasks in
/// FIFO order, as usual. Overall, this has roughly the same order as
/// the now-deprecated [`breadth_first`] option, except the effect is
/// isolated to a particular scope. If spawns are intermingled from any
/// combination of `scope()` and `scope_fifo()`, or from different
/// threads, their order is only specified with respect to spawns in the
/// same scope and thread.
///
/// For more details on this design, see Rayon [RFC #1].
///
/// [`breadth_first`]: struct.ThreadPoolBuilder.html#method.breadth_first
/// [RFC #1]: https://github.com/rayon-rs/rfcs/blob/master/accepted/rfc0001-scope-scheduling.md
///
/// # Panics
///
/// If a panic occurs, either in the closure given to `scope_fifo()` or
/// in any of the spawned jobs, that panic will be propagated and the
/// call to `scope_fifo()` will panic. If multiple panics occurs, it is
/// non-deterministic which of their panic values will propagate.
/// Regardless, once a task is spawned using `scope.spawn_fifo()`, it
/// will execute, even if the spawning task should later panic.
/// `scope_fifo()` returns once all spawned jobs have completed, and any
/// panics are propagated at that point.
pub fn scope_fifo<'scope, OP, R>(op: OP) -> R
where
    OP: FnOnce(&ScopeFifo<'scope>) -> R + Send,
    R: Send,
{
    in_worker(|owner_thread, _| {
        let scope = ScopeFifo::<'scope>::new(Some(owner_thread), None);
        scope.base.complete(Some(owner_thread), || op(&scope))
    })
}

/// Creates a "fork-join" scope `s` and invokes the closure with a
/// reference to `s`. This closure can then spawn asynchronous tasks
/// into `s`. Those tasks may run asynchronously with respect to the
/// closure; they may themselves spawn additional tasks into `s`. When
/// the closure returns, it will block until all tasks that have been
/// spawned into `s` complete.
///
/// This is just like `scope()` except the closure runs on the same thread
/// that calls `in_place_scope()`. Only work that it spawns runs in the
/// thread pool.
///
/// # Panics
///
/// If a panic occurs, either in the closure given to `in_place_scope()` or in
/// any of the spawned jobs, that panic will be propagated and the
/// call to `in_place_scope()` will panic. If multiple panics occurs, it is
/// non-deterministic which of their panic values will propagate.
/// Regardless, once a task is spawned using `scope.spawn()`, it will
/// execute, even if the spawning task should later panic. `in_place_scope()`
/// returns once all spawned jobs have completed, and any panics are
/// propagated at that point.
pub fn in_place_scope<'scope, OP, R>(op: OP) -> R
where
    OP: FnOnce(&Scope<'scope>) -> R,
{
    do_in_place_scope(None, op)
}

pub(crate) fn do_in_place_scope<'scope, OP, R>(registry: Option<&Arc<Registry>>, op: OP) -> R
where
    OP: FnOnce(&Scope<'scope>) -> R,
{
    let thread = unsafe { WorkerThread::current().as_ref() };
    let scope = Scope::<'scope>::new(thread, registry);
    scope.base.complete(thread, || op(&scope))
}

/// Creates a "fork-join" scope `s` with FIFO order, and invokes the
/// closure with a reference to `s`. This closure can then spawn
/// asynchronous tasks into `s`. Those tasks may run asynchronously with
/// respect to the closure; they may themselves spawn additional tasks
/// into `s`. When the closure returns, it will block until all tasks
/// that have been spawned into `s` complete.
///
/// This is just like `scope_fifo()` except the closure runs on the same thread
/// that calls `in_place_scope_fifo()`. Only work that it spawns runs in the
/// thread pool.
///
/// # Panics
///
/// If a panic occurs, either in the closure given to `in_place_scope_fifo()` or in
/// any of the spawned jobs, that panic will be propagated and the
/// call to `in_place_scope_fifo()` will panic. If multiple panics occurs, it is
/// non-deterministic which of their panic values will propagate.
/// Regardless, once a task is spawned using `scope.spawn_fifo()`, it will
/// execute, even if the spawning task should later panic. `in_place_scope_fifo()`
/// returns once all spawned jobs have completed, and any panics are
/// propagated at that point.
pub fn in_place_scope_fifo<'scope, OP, R>(op: OP) -> R
where
    OP: FnOnce(&ScopeFifo<'scope>) -> R,
{
    do_in_place_scope_fifo(None, op)
}

pub(crate) fn do_in_place_scope_fifo<'scope, OP, R>(registry: Option<&Arc<Registry>>, op: OP) -> R
where
    OP: FnOnce(&ScopeFifo<'scope>) -> R,
{
    let thread = unsafe { WorkerThread::current().as_ref() };
    let scope = ScopeFifo::<'scope>::new(thread, registry);
    scope.base.complete(thread, || op(&scope))
}

impl<'scope> Scope<'scope> {
    fn new(owner: Option<&WorkerThread>, registry: Option<&Arc<Registry>>) -> Self {
        let base = ScopeBase::new(owner, registry);
        Scope { base }
    }

    /// Spawns a job into the fork-join scope `self`. This job will
    /// execute sometime before the fork-join scope completes. The
    /// job is specified as a closure, and this closure receives its
    /// own reference to the scope `self` as argument. This can be
    /// used to inject new jobs into `self`.
    ///
    /// # Returns
    ///
    /// Nothing. The spawned closures cannot pass back values to the
    /// caller directly, though they can write to local variables on
    /// the stack (if those variables outlive the scope) or
    /// communicate through shared channels.
    ///
    /// (The intention is to eventually integrate with Rust futures to
    /// support spawns of functions that compute a value.)
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use rustc_thread_pool as rayon;
    /// let mut value_a = None;
    /// let mut value_b = None;
    /// let mut value_c = None;
    /// rayon::scope(|s| {
    ///     s.spawn(|s1| {
    ///         //   ^ this is the same scope as `s`; this handle `s1`
    ///         //     is intended for use by the spawned task,
    ///         //     since scope handles cannot cross thread boundaries.
    ///
    ///         value_a = Some(22);
    ///
    ///         // the scope `s` will not end until all these tasks are done
    ///         s1.spawn(|_| {
    ///             value_b = Some(44);
    ///         });
    ///     });
    ///
    ///     s.spawn(|_| {
    ///         value_c = Some(66);
    ///     });
    /// });
    /// assert_eq!(value_a, Some(22));
    /// assert_eq!(value_b, Some(44));
    /// assert_eq!(value_c, Some(66));
    /// ```
    ///
    /// # See also
    ///
    /// The [`scope` function] has more extensive documentation about
    /// task spawning.
    ///
    /// [`scope` function]: fn.scope.html
    pub fn spawn<BODY>(&self, body: BODY)
    where
        BODY: FnOnce(&Scope<'scope>) + Send + 'scope,
    {
        let scope_ptr = ScopePtr(self);
        let job = HeapJob::new(self.base.tlv, move |id| unsafe {
            // SAFETY: this job will execute before the scope ends.
            let scope = scope_ptr.as_ref();

            // Mark this job is started.
            scope.base.pending_jobs.lock().unwrap().remove(&id);

            ScopeBase::execute_job(&scope.base, move || body(scope))
        });
        let job_ref = self.base.heap_job_ref(job);

        // Mark this job as pending.
        self.base.pending_jobs.lock().unwrap().insert(job_ref.id());
        // Since `Scope` implements `Sync`, we can't be sure that we're still in a
        // thread of this pool, so we can't just push to the local worker thread.
        // Also, this might be an in-place scope.
        self.base.registry.inject_or_push(job_ref);
    }

    /// Spawns a job into every thread of the fork-join scope `self`. This job will
    /// execute on each thread sometime before the fork-join scope completes. The
    /// job is specified as a closure, and this closure receives its own reference
    /// to the scope `self` as argument, as well as a `BroadcastContext`.
    pub fn spawn_broadcast<BODY>(&self, body: BODY)
    where
        BODY: Fn(&Scope<'scope>, BroadcastContext<'_>) + Send + Sync + 'scope,
    {
        let scope_ptr = ScopePtr(self);
        let job = ArcJob::new(move |id| unsafe {
            // SAFETY: this job will execute before the scope ends.
            let scope = scope_ptr.as_ref();
            let body = &body;

            let current_index = WorkerThread::current().as_ref().map(|worker| worker.index());
            if current_index == scope.base.worker {
                // Mark this job as started on the scope's worker thread.
                scope.base.pending_jobs.lock().unwrap().remove(&id);
            }

            let func = move || BroadcastContext::with(move |ctx| body(scope, ctx));
            ScopeBase::execute_job(&scope.base, func)
        });
        self.base.inject_broadcast(job)
    }
}

impl<'scope> ScopeFifo<'scope> {
    fn new(owner: Option<&WorkerThread>, registry: Option<&Arc<Registry>>) -> Self {
        let base = ScopeBase::new(owner, registry);
        let num_threads = base.registry.num_threads();
        let fifos = (0..num_threads).map(|_| JobFifo::new()).collect();
        ScopeFifo { base, fifos }
    }

    /// Spawns a job into the fork-join scope `self`. This job will
    /// execute sometime before the fork-join scope completes. The
    /// job is specified as a closure, and this closure receives its
    /// own reference to the scope `self` as argument. This can be
    /// used to inject new jobs into `self`.
    ///
    /// # See also
    ///
    /// This method is akin to [`Scope::spawn()`], but with a FIFO
    /// priority. The [`scope_fifo` function] has more details about
    /// this distinction.
    ///
    /// [`Scope::spawn()`]: struct.Scope.html#method.spawn
    /// [`scope_fifo` function]: fn.scope_fifo.html
    pub fn spawn_fifo<BODY>(&self, body: BODY)
    where
        BODY: FnOnce(&ScopeFifo<'scope>) + Send + 'scope,
    {
        let scope_ptr = ScopePtr(self);
        let job = HeapJob::new(self.base.tlv, move |id| unsafe {
            // SAFETY: this job will execute before the scope ends.
            let scope = scope_ptr.as_ref();

            // Mark this job is started.
            scope.base.pending_jobs.lock().unwrap().remove(&id);

            ScopeBase::execute_job(&scope.base, move || body(scope))
        });
        let job_ref = self.base.heap_job_ref(job);

        // Mark this job as pending.
        self.base.pending_jobs.lock().unwrap().insert(job_ref.id());

        // Since `ScopeFifo` implements `Sync`, we can't be sure that we're still in a
        // thread of this pool, so we can't just push to the local worker thread.
        // Also, this might be an in-place scope.
        self.base.registry.inject_or_push(job_ref);
    }

    /// Spawns a job into every thread of the fork-join scope `self`. This job will
    /// execute on each thread sometime before the fork-join scope completes. The
    /// job is specified as a closure, and this closure receives its own reference
    /// to the scope `self` as argument, as well as a `BroadcastContext`.
    pub fn spawn_broadcast<BODY>(&self, body: BODY)
    where
        BODY: Fn(&ScopeFifo<'scope>, BroadcastContext<'_>) + Send + Sync + 'scope,
    {
        let scope_ptr = ScopePtr(self);
        let job = ArcJob::new(move |id| unsafe {
            // SAFETY: this job will execute before the scope ends.
            let scope = scope_ptr.as_ref();

            let current_index = WorkerThread::current().as_ref().map(|worker| worker.index());
            if current_index == scope.base.worker {
                // Mark this job as started on the scope's worker thread.
                scope.base.pending_jobs.lock().unwrap().remove(&id);
            }
            let body = &body;
            let func = move || BroadcastContext::with(move |ctx| body(scope, ctx));
            ScopeBase::execute_job(&scope.base, func)
        });
        self.base.inject_broadcast(job)
    }
}

impl<'scope> ScopeBase<'scope> {
    /// Creates the base of a new scope for the given registry
    fn new(owner: Option<&WorkerThread>, registry: Option<&Arc<Registry>>) -> Self {
        let registry = registry.unwrap_or_else(|| match owner {
            Some(owner) => owner.registry(),
            None => global_registry(),
        });

        ScopeBase {
            registry: Arc::clone(registry),
            panic: AtomicPtr::new(ptr::null_mut()),
            job_completed_latch: CountLatch::new(owner),
            #[allow(rustc::default_hash_types)]
            pending_jobs: Mutex::new(HashSet::new()),
            worker: owner.map(|w| w.index()),
            marker: PhantomData,
            tlv: tlv::get(),
        }
    }

    fn heap_job_ref<FUNC>(&self, job: Box<HeapJob<FUNC>>) -> JobRef
    where
        FUNC: FnOnce(JobRefId) + Send + 'scope,
    {
        unsafe {
            self.job_completed_latch.increment();
            job.into_job_ref()
        }
    }

    fn inject_broadcast<FUNC>(&self, job: Arc<ArcJob<FUNC>>)
    where
        FUNC: Fn(JobRefId) + Send + Sync + 'scope,
    {
        if self.worker.is_some() {
            let id = unsafe { ArcJob::as_job_ref(&job).id() };
            self.pending_jobs.lock().unwrap().insert(id);
        }
        let n_threads = self.registry.num_threads();
        let job_refs = (0..n_threads).map(|_| unsafe {
            self.job_completed_latch.increment();
            ArcJob::as_job_ref(&job)
        });

        self.registry.inject_broadcast(job_refs);
    }

    /// Executes `func` as a job, either aborting or executing as
    /// appropriate.
    fn complete<FUNC, R>(&self, owner: Option<&WorkerThread>, func: FUNC) -> R
    where
        FUNC: FnOnce() -> R,
    {
        let result = unsafe { Self::execute_job_closure(self, func) };
        self.job_completed_latch.wait(
            owner,
            || self.pending_jobs.lock().unwrap().is_empty(),
            |job| self.pending_jobs.lock().unwrap().contains(&job.id()),
        );

        // Restore the TLV if we ran some jobs while waiting
        tlv::set(self.tlv);

        self.maybe_propagate_panic();
        result.unwrap() // only None if `op` panicked, and that would have been propagated
    }

    /// Executes `func` as a job, either aborting or executing as
    /// appropriate.
    unsafe fn execute_job<FUNC>(this: *const Self, func: FUNC)
    where
        FUNC: FnOnce(),
    {
        let _: Option<()> = unsafe { Self::execute_job_closure(this, func) };
    }

    /// Executes `func` as a job in scope. Adjusts the "job completed"
    /// counters and also catches any panic and stores it into
    /// `scope`.
    unsafe fn execute_job_closure<FUNC, R>(this: *const Self, func: FUNC) -> Option<R>
    where
        FUNC: FnOnce() -> R,
    {
        let result = match unwind::halt_unwinding(func) {
            Ok(r) => Some(r),
            Err(err) => {
                unsafe { (*this).job_panicked(err) };
                None
            }
        };
        unsafe { Latch::set(&(*this).job_completed_latch) };
        result
    }

    fn job_panicked(&self, err: Box<dyn Any + Send + 'static>) {
        // capture the first error we see, free the rest
        if self.panic.load(Ordering::Relaxed).is_null() {
            let nil = ptr::null_mut();
            let mut err = ManuallyDrop::new(Box::new(err)); // box up the fat ptr
            let err_ptr: *mut Box<dyn Any + Send + 'static> = &mut **err;
            if self
                .panic
                .compare_exchange(nil, err_ptr, Ordering::Release, Ordering::Relaxed)
                .is_ok()
            {
                // ownership now transferred into self.panic
            } else {
                // another panic raced in ahead of us, so drop ours
                let _: Box<Box<_>> = ManuallyDrop::into_inner(err);
            }
        }
    }

    fn maybe_propagate_panic(&self) {
        // propagate panic, if any occurred; at this point, all
        // outstanding jobs have completed, so we can use a relaxed
        // ordering:
        let panic = self.panic.swap(ptr::null_mut(), Ordering::Relaxed);
        if !panic.is_null() {
            let value = unsafe { Box::from_raw(panic) };

            // Restore the TLV if we ran some jobs while waiting
            tlv::set(self.tlv);

            unwind::resume_unwinding(*value);
        }
    }
}

impl<'scope> fmt::Debug for Scope<'scope> {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt.debug_struct("Scope")
            .field("pool_id", &self.base.registry.id())
            .field("panic", &self.base.panic)
            .field("job_completed_latch", &self.base.job_completed_latch)
            .finish()
    }
}

impl<'scope> fmt::Debug for ScopeFifo<'scope> {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt.debug_struct("ScopeFifo")
            .field("num_fifos", &self.fifos.len())
            .field("pool_id", &self.base.registry.id())
            .field("panic", &self.base.panic)
            .field("job_completed_latch", &self.base.job_completed_latch)
            .finish()
    }
}

/// Used to capture a scope `&Self` pointer in jobs, without faking a lifetime.
///
/// Unsafe code is still required to dereference the pointer, but that's fine in
/// scope jobs that are guaranteed to execute before the scope ends.
struct ScopePtr<T>(*const T);

// SAFETY: !Send for raw pointers is not for safety, just as a lint
unsafe impl<T: Sync> Send for ScopePtr<T> {}

// SAFETY: !Sync for raw pointers is not for safety, just as a lint
unsafe impl<T: Sync> Sync for ScopePtr<T> {}

impl<T> ScopePtr<T> {
    // Helper to avoid disjoint captures of `scope_ptr.0`
    unsafe fn as_ref(&self) -> &T {
        unsafe { &*self.0 }
    }
}
