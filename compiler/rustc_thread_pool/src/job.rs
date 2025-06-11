use std::any::Any;
use std::cell::UnsafeCell;
use std::mem;
use std::sync::Arc;

use crossbeam_deque::{Injector, Steal};

use crate::latch::Latch;
use crate::tlv::Tlv;
use crate::{tlv, unwind};

pub(super) enum JobResult<T> {
    None,
    Ok(T),
    Panic(Box<dyn Any + Send>),
}

/// A `Job` is used to advertise work for other threads that they may
/// want to steal. In accordance with time honored tradition, jobs are
/// arranged in a deque, so that thieves can take from the top of the
/// deque while the main worker manages the bottom of the deque. This
/// deque is managed by the `thread_pool` module.
pub(super) trait Job {
    /// Unsafe: this may be called from a different thread than the one
    /// which scheduled the job, so the implementer must ensure the
    /// appropriate traits are met, whether `Send`, `Sync`, or both.
    unsafe fn execute(this: *const ());
}

/// Effectively a Job trait object. Each JobRef **must** be executed
/// exactly once, or else data may leak.
///
/// Internally, we store the job's data in a `*const ()` pointer. The
/// true type is something like `*const StackJob<...>`, but we hide
/// it. We also carry the "execute fn" from the `Job` trait.
pub(super) struct JobRef {
    pointer: *const (),
    execute_fn: unsafe fn(*const ()),
}

unsafe impl Send for JobRef {}
unsafe impl Sync for JobRef {}

impl JobRef {
    /// Unsafe: caller asserts that `data` will remain valid until the
    /// job is executed.
    pub(super) unsafe fn new<T>(data: *const T) -> JobRef
    where
        T: Job,
    {
        // erase types:
        JobRef { pointer: data as *const (), execute_fn: <T as Job>::execute }
    }

    /// Returns an opaque handle that can be saved and compared,
    /// without making `JobRef` itself `Copy + Eq`.
    #[inline]
    pub(super) fn id(&self) -> impl Eq {
        (self.pointer, self.execute_fn)
    }

    #[inline]
    pub(super) unsafe fn execute(self) {
        unsafe { (self.execute_fn)(self.pointer) }
    }
}

/// A job that will be owned by a stack slot. This means that when it
/// executes it need not free any heap data, the cleanup occurs when
/// the stack frame is later popped. The function parameter indicates
/// `true` if the job was stolen -- executed on a different thread.
pub(super) struct StackJob<L, F, R>
where
    L: Latch + Sync,
    F: FnOnce(bool) -> R + Send,
    R: Send,
{
    pub(super) latch: L,
    func: UnsafeCell<Option<F>>,
    result: UnsafeCell<JobResult<R>>,
    tlv: Tlv,
}

impl<L, F, R> StackJob<L, F, R>
where
    L: Latch + Sync,
    F: FnOnce(bool) -> R + Send,
    R: Send,
{
    pub(super) fn new(tlv: Tlv, func: F, latch: L) -> StackJob<L, F, R> {
        StackJob {
            latch,
            func: UnsafeCell::new(Some(func)),
            result: UnsafeCell::new(JobResult::None),
            tlv,
        }
    }

    pub(super) unsafe fn as_job_ref(&self) -> JobRef {
        unsafe { JobRef::new(self) }
    }

    pub(super) unsafe fn run_inline(self, stolen: bool) -> R {
        self.func.into_inner().unwrap()(stolen)
    }

    pub(super) unsafe fn into_result(self) -> R {
        self.result.into_inner().into_return_value()
    }
}

impl<L, F, R> Job for StackJob<L, F, R>
where
    L: Latch + Sync,
    F: FnOnce(bool) -> R + Send,
    R: Send,
{
    unsafe fn execute(this: *const ()) {
        let this = unsafe { &*(this as *const Self) };
        tlv::set(this.tlv);
        let abort = unwind::AbortIfPanic;
        let func = unsafe { (*this.func.get()).take().unwrap() };
        unsafe {
            (*this.result.get()) = JobResult::call(func);
        }
        unsafe {
            Latch::set(&this.latch);
        }
        mem::forget(abort);
    }
}

/// Represents a job stored in the heap. Used to implement
/// `scope`. Unlike `StackJob`, when executed, `HeapJob` simply
/// invokes a closure, which then triggers the appropriate logic to
/// signal that the job executed.
///
/// (Probably `StackJob` should be refactored in a similar fashion.)
pub(super) struct HeapJob<BODY>
where
    BODY: FnOnce() + Send,
{
    job: BODY,
    tlv: Tlv,
}

impl<BODY> HeapJob<BODY>
where
    BODY: FnOnce() + Send,
{
    pub(super) fn new(tlv: Tlv, job: BODY) -> Box<Self> {
        Box::new(HeapJob { job, tlv })
    }

    /// Creates a `JobRef` from this job -- note that this hides all
    /// lifetimes, so it is up to you to ensure that this JobRef
    /// doesn't outlive any data that it closes over.
    pub(super) unsafe fn into_job_ref(self: Box<Self>) -> JobRef {
        unsafe { JobRef::new(Box::into_raw(self)) }
    }

    /// Creates a static `JobRef` from this job.
    pub(super) fn into_static_job_ref(self: Box<Self>) -> JobRef
    where
        BODY: 'static,
    {
        unsafe { self.into_job_ref() }
    }
}

impl<BODY> Job for HeapJob<BODY>
where
    BODY: FnOnce() + Send,
{
    unsafe fn execute(this: *const ()) {
        let this = unsafe { Box::from_raw(this as *mut Self) };
        tlv::set(this.tlv);
        (this.job)();
    }
}

/// Represents a job stored in an `Arc` -- like `HeapJob`, but may
/// be turned into multiple `JobRef`s and called multiple times.
pub(super) struct ArcJob<BODY>
where
    BODY: Fn() + Send + Sync,
{
    job: BODY,
}

impl<BODY> ArcJob<BODY>
where
    BODY: Fn() + Send + Sync,
{
    pub(super) fn new(job: BODY) -> Arc<Self> {
        Arc::new(ArcJob { job })
    }

    /// Creates a `JobRef` from this job -- note that this hides all
    /// lifetimes, so it is up to you to ensure that this JobRef
    /// doesn't outlive any data that it closes over.
    pub(super) unsafe fn as_job_ref(this: &Arc<Self>) -> JobRef {
        unsafe { JobRef::new(Arc::into_raw(Arc::clone(this))) }
    }

    /// Creates a static `JobRef` from this job.
    pub(super) fn as_static_job_ref(this: &Arc<Self>) -> JobRef
    where
        BODY: 'static,
    {
        unsafe { Self::as_job_ref(this) }
    }
}

impl<BODY> Job for ArcJob<BODY>
where
    BODY: Fn() + Send + Sync,
{
    unsafe fn execute(this: *const ()) {
        let this = unsafe { Arc::from_raw(this as *mut Self) };
        (this.job)();
    }
}

impl<T> JobResult<T> {
    fn call(func: impl FnOnce(bool) -> T) -> Self {
        match unwind::halt_unwinding(|| func(true)) {
            Ok(x) => JobResult::Ok(x),
            Err(x) => JobResult::Panic(x),
        }
    }

    /// Convert the `JobResult` for a job that has finished (and hence
    /// its JobResult is populated) into its return value.
    ///
    /// NB. This will panic if the job panicked.
    pub(super) fn into_return_value(self) -> T {
        match self {
            JobResult::None => unreachable!(),
            JobResult::Ok(x) => x,
            JobResult::Panic(x) => unwind::resume_unwinding(x),
        }
    }
}

/// Indirect queue to provide FIFO job priority.
pub(super) struct JobFifo {
    inner: Injector<JobRef>,
}

impl JobFifo {
    pub(super) fn new() -> Self {
        JobFifo { inner: Injector::new() }
    }

    pub(super) unsafe fn push(&self, job_ref: JobRef) -> JobRef {
        // A little indirection ensures that spawns are always prioritized in FIFO order. The
        // jobs in a thread's deque may be popped from the back (LIFO) or stolen from the front
        // (FIFO), but either way they will end up popping from the front of this queue.
        self.inner.push(job_ref);
        unsafe { JobRef::new(self) }
    }
}

impl Job for JobFifo {
    unsafe fn execute(this: *const ()) {
        // We "execute" a queue by executing its first job, FIFO.
        let this = unsafe { &*(this as *const Self) };
        loop {
            match this.inner.steal() {
                Steal::Success(job_ref) => break unsafe { job_ref.execute() },
                Steal::Empty => panic!("FIFO is empty"),
                Steal::Retry => {}
            }
        }
    }
}
