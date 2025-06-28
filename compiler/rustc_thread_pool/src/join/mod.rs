use std::any::Any;

use crate::job::StackJob;
use crate::latch::SpinLatch;
use crate::registry::{self, WorkerThread};
use crate::tlv::{self, Tlv};
use crate::{FnContext, unwind};

#[cfg(test)]
mod tests;

/// Takes two closures and *potentially* runs them in parallel. It
/// returns a pair of the results from those closures.
///
/// Conceptually, calling `join()` is similar to spawning two threads,
/// one executing each of the two closures. However, the
/// implementation is quite different and incurs very low
/// overhead. The underlying technique is called "work stealing": the
/// Rayon runtime uses a fixed pool of worker threads and attempts to
/// only execute code in parallel when there are idle CPUs to handle
/// it.
///
/// When `join` is called from outside the thread pool, the calling
/// thread will block while the closures execute in the pool. When
/// `join` is called within the pool, the calling thread still actively
/// participates in the thread pool. It will begin by executing closure
/// A (on the current thread). While it is doing that, it will advertise
/// closure B as being available for other threads to execute. Once closure A
/// has completed, the current thread will try to execute closure B;
/// if however closure B has been stolen, then it will look for other work
/// while waiting for the thief to fully execute closure B. (This is the
/// typical work-stealing strategy).
///
/// # Examples
///
/// This example uses join to perform a quick-sort (note this is not a
/// particularly optimized implementation: if you **actually** want to
/// sort for real, you should prefer [the `par_sort` method] offered
/// by Rayon).
///
/// [the `par_sort` method]: ../rayon/slice/trait.ParallelSliceMut.html#method.par_sort
///
/// ```rust
/// # use rustc_thread_pool as rayon;
/// let mut v = vec![5, 1, 8, 22, 0, 44];
/// quick_sort(&mut v);
/// assert_eq!(v, vec![0, 1, 5, 8, 22, 44]);
///
/// fn quick_sort<T:PartialOrd+Send>(v: &mut [T]) {
///    if v.len() > 1 {
///        let mid = partition(v);
///        let (lo, hi) = v.split_at_mut(mid);
///        rayon::join(|| quick_sort(lo),
///                    || quick_sort(hi));
///    }
/// }
///
/// // Partition rearranges all items `<=` to the pivot
/// // item (arbitrary selected to be the last item in the slice)
/// // to the first half of the slice. It then returns the
/// // "dividing point" where the pivot is placed.
/// fn partition<T:PartialOrd+Send>(v: &mut [T]) -> usize {
///     let pivot = v.len() - 1;
///     let mut i = 0;
///     for j in 0..pivot {
///         if v[j] <= v[pivot] {
///             v.swap(i, j);
///             i += 1;
///         }
///     }
///     v.swap(i, pivot);
///     i
/// }
/// ```
///
/// # Warning about blocking I/O
///
/// The assumption is that the closures given to `join()` are
/// CPU-bound tasks that do not perform I/O or other blocking
/// operations. If you do perform I/O, and that I/O should block
/// (e.g., waiting for a network request), the overall performance may
/// be poor. Moreover, if you cause one closure to be blocked waiting
/// on another (for example, using a channel), that could lead to a
/// deadlock.
///
/// # Panics
///
/// No matter what happens, both closures will always be executed. If
/// a single closure panics, whether it be the first or second
/// closure, that panic will be propagated and hence `join()` will
/// panic with the same panic value. If both closures panic, `join()`
/// will panic with the panic value from the first closure.
pub fn join<A, B, RA, RB>(oper_a: A, oper_b: B) -> (RA, RB)
where
    A: FnOnce() -> RA + Send,
    B: FnOnce() -> RB + Send,
    RA: Send,
    RB: Send,
{
    #[inline]
    fn call<R>(f: impl FnOnce() -> R) -> impl FnOnce(FnContext) -> R {
        move |_| f()
    }

    join_context(call(oper_a), call(oper_b))
}

/// Identical to `join`, except that the closures have a parameter
/// that provides context for the way the closure has been called,
/// especially indicating whether they're executing on a different
/// thread than where `join_context` was called. This will occur if
/// the second job is stolen by a different thread, or if
/// `join_context` was called from outside the thread pool to begin
/// with.
pub fn join_context<A, B, RA, RB>(oper_a: A, oper_b: B) -> (RA, RB)
where
    A: FnOnce(FnContext) -> RA + Send,
    B: FnOnce(FnContext) -> RB + Send,
    RA: Send,
    RB: Send,
{
    #[inline]
    fn call_a<R>(f: impl FnOnce(FnContext) -> R, injected: bool) -> impl FnOnce() -> R {
        move || f(FnContext::new(injected))
    }

    #[inline]
    fn call_b<R>(f: impl FnOnce(FnContext) -> R) -> impl FnOnce(bool) -> R {
        move |migrated| f(FnContext::new(migrated))
    }

    registry::in_worker(|worker_thread, injected| unsafe {
        let tlv = tlv::get();
        // Create virtual wrapper for task b; this all has to be
        // done here so that the stack frame can keep it all live
        // long enough.
        let job_b = StackJob::new(tlv, call_b(oper_b), SpinLatch::new(worker_thread));
        let job_b_ref = job_b.as_job_ref();
        let job_b_id = job_b_ref.id();
        worker_thread.push(job_b_ref);

        // Execute task a; hopefully b gets stolen in the meantime.
        let status_a = unwind::halt_unwinding(call_a(oper_a, injected));
        let result_a = match status_a {
            Ok(v) => v,
            Err(err) => join_recover_from_panic(worker_thread, &job_b.latch, err, tlv),
        };

        // Now that task A has finished, try to pop job B from the
        // local stack. It may already have been popped by job A; it
        // may also have been stolen. There may also be some tasks
        // pushed on top of it in the stack, and we will have to pop
        // those off to get to it.
        while !job_b.latch.probe() {
            if let Some(job) = worker_thread.take_local_job() {
                if job_b_id == job.id() {
                    // Found it! Let's run it.
                    //
                    // Note that this could panic, but it's ok if we unwind here.

                    // Restore the TLV since we might have run some jobs overwriting it when waiting for job b.
                    tlv::set(tlv);

                    let result_b = job_b.run_inline(injected);
                    return (result_a, result_b);
                } else {
                    worker_thread.execute(job);
                }
            } else {
                // Local deque is empty. Time to steal from other
                // threads.
                worker_thread.wait_until(&job_b.latch);
                debug_assert!(job_b.latch.probe());
                break;
            }
        }

        // Restore the TLV since we might have run some jobs overwriting it when waiting for job b.
        tlv::set(tlv);

        (result_a, job_b.into_result())
    })
}

/// If job A panics, we still cannot return until we are sure that job
/// B is complete. This is because it may contain references into the
/// enclosing stack frame(s).
#[cold] // cold path
unsafe fn join_recover_from_panic(
    worker_thread: &WorkerThread,
    job_b_latch: &SpinLatch<'_>,
    err: Box<dyn Any + Send>,
    tlv: Tlv,
) -> ! {
    unsafe { worker_thread.wait_until(job_b_latch) };

    // Restore the TLV since we might have run some jobs overwriting it when waiting for job b.
    tlv::set(tlv);

    unwind::resume_unwinding(err)
}
