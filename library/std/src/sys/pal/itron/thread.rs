//! Thread implementation backed by μITRON tasks. Assumes `acre_tsk` and
//! `exd_tsk` are available.

use super::error::{ItronError, expect_success, expect_success_aborting};
use super::time::dur2reltims;
use super::{abi, task};
use crate::cell::UnsafeCell;
use crate::ffi::CStr;
use crate::mem::ManuallyDrop;
use crate::num::NonZero;
use crate::ptr::NonNull;
use crate::sync::atomic::{Atomic, AtomicUsize, Ordering};
use crate::time::Duration;
use crate::{hint, io};

pub struct Thread {
    p_inner: NonNull<ThreadInner>,

    /// The ID of the underlying task.
    task: abi::ID,
}

// Safety: There's nothing in `Thread` that ties it to the original creator. It
//         can be dropped by any threads.
unsafe impl Send for Thread {}
// Safety: `Thread` provides no methods that take `&self`.
unsafe impl Sync for Thread {}

/// State data shared between a parent thread and child thread. It's dropped on
/// a transition to one of the final states.
struct ThreadInner {
    /// This field is used on thread creation to pass a closure from
    /// `Thread::new` to the created task.
    start: UnsafeCell<ManuallyDrop<Box<dyn FnOnce()>>>,

    /// A state machine. Each transition is annotated with `[...]` in the
    /// source code.
    ///
    /// ```text
    ///
    ///    <P>: parent, <C>: child, (?): don't-care
    ///
    ///       DETACHED (-1)  -------------------->  EXITED (?)
    ///                        <C>finish/exd_tsk
    ///          ^
    ///          |
    ///          | <P>detach
    ///          |
    ///
    ///       INIT (0)  ----------------------->  FINISHED (-1)
    ///                        <C>finish
    ///          |                                    |
    ///          | <P>join/slp_tsk                    | <P>join/del_tsk
    ///          |                                    | <P>detach/del_tsk
    ///          v                                    v
    ///
    ///       JOINING                              JOINED (?)
    ///     (parent_tid)
    ///                                            ^
    ///             \                             /
    ///              \  <C>finish/wup_tsk        / <P>slp_tsk-complete/ter_tsk
    ///               \                         /                      & del_tsk
    ///                \                       /
    ///                 '--> JOIN_FINALIZE ---'
    ///                          (-1)
    ///
    lifecycle: Atomic<usize>,
}

// Safety: The only `!Sync` field, `ThreadInner::start`, is only touched by
//         the task represented by `ThreadInner`.
unsafe impl Sync for ThreadInner {}

const LIFECYCLE_INIT: usize = 0;
const LIFECYCLE_FINISHED: usize = usize::MAX;
const LIFECYCLE_DETACHED: usize = usize::MAX;
const LIFECYCLE_JOIN_FINALIZE: usize = usize::MAX;
const LIFECYCLE_DETACHED_OR_JOINED: usize = usize::MAX;
const LIFECYCLE_EXITED_OR_FINISHED_OR_JOIN_FINALIZE: usize = usize::MAX;
// there's no single value for `JOINING`

// 64KiB for 32-bit ISAs, 128KiB for 64-bit ISAs.
pub const DEFAULT_MIN_STACK_SIZE: usize = 0x4000 * size_of::<usize>();

impl Thread {
    /// # Safety
    ///
    /// See `thread::Builder::spawn_unchecked` for safety requirements.
    pub unsafe fn new(stack: usize, p: Box<dyn FnOnce()>) -> io::Result<Thread> {
        let inner = Box::new(ThreadInner {
            start: UnsafeCell::new(ManuallyDrop::new(p)),
            lifecycle: AtomicUsize::new(LIFECYCLE_INIT),
        });

        unsafe extern "C" fn trampoline(exinf: isize) {
            let p_inner: *mut ThreadInner = crate::ptr::with_exposed_provenance_mut(exinf as usize);
            // Safety: `ThreadInner` is alive at this point
            let inner = unsafe { &*p_inner };

            // Safety: Since `trampoline` is called only once for each
            //         `ThreadInner` and only `trampoline` touches `start`,
            //         `start` contains contents and is safe to mutably borrow.
            let p = unsafe { ManuallyDrop::take(&mut *inner.start.get()) };
            p();

            // Fix the current thread's state just in case, so that the
            // destructors won't abort
            // Safety: Not really unsafe
            let _ = unsafe { abi::unl_cpu() };
            let _ = unsafe { abi::ena_dsp() };

            // Run TLS destructors now because they are not
            // called automatically for terminated tasks.
            unsafe { crate::sys::thread_local::destructors::run() };

            let old_lifecycle = inner
                .lifecycle
                .swap(LIFECYCLE_EXITED_OR_FINISHED_OR_JOIN_FINALIZE, Ordering::AcqRel);

            match old_lifecycle {
                LIFECYCLE_DETACHED => {
                    // [DETACHED → EXITED]
                    // No one will ever join, so we'll ask the collector task to
                    // delete the task.

                    // In this case, `*p_inner`'s ownership has been moved to
                    // us, and we are responsible for dropping it. The acquire
                    // ordering ensures that the swap operation that wrote
                    // `LIFECYCLE_DETACHED` happens-before `Box::from_raw(
                    // p_inner)`.
                    // Safety: See above.
                    let _ = unsafe { Box::from_raw(p_inner) };

                    // Safety: There are no pinned references to the stack
                    unsafe { terminate_and_delete_current_task() };
                }
                LIFECYCLE_INIT => {
                    // [INIT → FINISHED]
                    // The parent hasn't decided whether to join or detach this
                    // thread yet. Whichever option the parent chooses,
                    // it'll have to delete this task.
                    // Since the parent might drop `*inner` as soon as it sees
                    // `FINISHED`, the release ordering must be used in the
                    // above `swap` call.
                }
                parent_tid => {
                    // Since the parent might drop `*inner` and terminate us as
                    // soon as it sees `JOIN_FINALIZE`, the release ordering
                    // must be used in the above `swap` call.
                    //
                    // To make the task referred to by `parent_tid` visible, we
                    // must use the acquire ordering in the above `swap` call.

                    // [JOINING → JOIN_FINALIZE]
                    // Wake up the parent task.
                    expect_success(
                        unsafe {
                            let mut er = abi::wup_tsk(parent_tid as _);
                            if er == abi::E_QOVR {
                                // `E_QOVR` indicates there's already
                                // a parking token
                                er = abi::E_OK;
                            }
                            er
                        },
                        &"wup_tsk",
                    );
                }
            }
        }

        // Safety: `Box::into_raw` returns a non-null pointer
        let p_inner = unsafe { NonNull::new_unchecked(Box::into_raw(inner)) };

        let new_task = ItronError::err_if_negative(unsafe {
            abi::acre_tsk(&abi::T_CTSK {
                // Activate this task immediately
                tskatr: abi::TA_ACT,
                exinf: p_inner.as_ptr().expose_provenance() as abi::EXINF,
                // The entry point
                task: Some(trampoline),
                // Inherit the calling task's base priority
                itskpri: abi::TPRI_SELF,
                stksz: stack,
                // Let the kernel allocate the stack,
                stk: crate::ptr::null_mut(),
            })
        })
        .map_err(|e| e.as_io_error())?;

        Ok(Self { p_inner, task: new_task })
    }

    pub fn yield_now() {
        expect_success(unsafe { abi::rot_rdq(abi::TPRI_SELF) }, &"rot_rdq");
    }

    pub fn set_name(_name: &CStr) {
        // nope
    }

    pub fn sleep(dur: Duration) {
        for timeout in dur2reltims(dur) {
            expect_success(unsafe { abi::dly_tsk(timeout) }, &"dly_tsk");
        }
    }

    pub fn join(self) {
        // Safety: `ThreadInner` is alive at this point
        let inner = unsafe { self.p_inner.as_ref() };
        // Get the current task ID. Panicking here would cause a resource leak,
        // so just abort on failure.
        let current_task = task::current_task_id_aborting();
        debug_assert!(usize::try_from(current_task).is_ok());
        debug_assert_ne!(current_task as usize, LIFECYCLE_INIT);
        debug_assert_ne!(current_task as usize, LIFECYCLE_DETACHED);

        let current_task = current_task as usize;

        match inner.lifecycle.swap(current_task, Ordering::AcqRel) {
            LIFECYCLE_INIT => {
                // [INIT → JOINING]
                // The child task will transition the state to `JOIN_FINALIZE`
                // and wake us up.
                //
                // To make the task referred to by `current_task` visible from
                // the child task's point of view, we must use the release
                // ordering in the above `swap` call.
                loop {
                    expect_success_aborting(unsafe { abi::slp_tsk() }, &"slp_tsk");
                    // To synchronize with the child task's memory accesses to
                    // `inner` up to the point of the assignment of
                    // `JOIN_FINALIZE`, `Ordering::Acquire` must be used for the
                    // `load`.
                    if inner.lifecycle.load(Ordering::Acquire) == LIFECYCLE_JOIN_FINALIZE {
                        break;
                    }
                }

                // [JOIN_FINALIZE → JOINED]
            }
            LIFECYCLE_FINISHED => {
                // [FINISHED → JOINED]
                // To synchronize with the child task's memory accesses to
                // `inner` up to the point of the assignment of `FINISHED`,
                // `Ordering::Acquire` must be used for the above `swap` call.
            }
            _ => unsafe { hint::unreachable_unchecked() },
        }

        // Terminate and delete the task
        // Safety: `self.task` still represents a task we own (because this
        //         method or `detach_inner` is called only once for each
        //         `Thread`). The task indicated that it's safe to delete by
        //         entering the `FINISHED` or `JOIN_FINALIZE` state.
        unsafe { terminate_and_delete_task(self.task) };

        // In either case, we are responsible for dropping `inner`.
        // Safety: The contents of `*p_inner` will not be accessed hereafter
        let _inner = unsafe { Box::from_raw(self.p_inner.as_ptr()) };

        // Skip the destructor (because it would attempt to detach the thread)
        crate::mem::forget(self);
    }
}

impl Drop for Thread {
    fn drop(&mut self) {
        // Safety: `ThreadInner` is alive at this point
        let inner = unsafe { self.p_inner.as_ref() };

        // Detach the thread.
        match inner.lifecycle.swap(LIFECYCLE_DETACHED_OR_JOINED, Ordering::AcqRel) {
            LIFECYCLE_INIT => {
                // [INIT → DETACHED]
                // When the time comes, the child will figure out that no
                // one will ever join it.
                // The ownership of `*p_inner` is moved to the child thread.
                // The release ordering ensures that the above swap operation on
                // `lifecycle` happens-before the child thread's
                // `Box::from_raw(p_inner)`.
            }
            LIFECYCLE_FINISHED => {
                // [FINISHED → JOINED]
                // The task has already decided that we should delete the task.
                // To synchronize with the child task's memory accesses to
                // `inner` up to the point of the assignment of `FINISHED`,
                // the acquire ordering is required for the above `swap` call.

                // Terminate and delete the task
                // Safety: `self.task` still represents a task we own (because
                //         this method or `join_inner` is called only once for
                //         each `Thread`). The task indicated that it's safe to
                //         delete by entering the `FINISHED` state.
                unsafe { terminate_and_delete_task(self.task) };

                // Wwe are responsible for dropping `*p_inner`.
                // Safety: The contents of `*p_inner` will not be accessed hereafter
                let _ = unsafe { Box::from_raw(self.p_inner.as_ptr()) };
            }
            _ => unsafe { hint::unreachable_unchecked() },
        }
    }
}

/// Terminates and deletes the specified task.
///
/// This function will abort if `deleted_task` refers to the calling task.
///
/// It is assumed that the specified task is solely managed by the caller -
/// i.e., other threads must not "resuscitate" the specified task or delete it
/// prematurely while this function is still in progress. It is allowed for the
/// specified task to exit by its own.
///
/// # Safety
///
/// The task must be safe to terminate. This is in general not true
/// because there might be pinned references to the task's stack.
unsafe fn terminate_and_delete_task(deleted_task: abi::ID) {
    // Terminate the task
    // Safety: Upheld by the caller
    match unsafe { abi::ter_tsk(deleted_task) } {
        // Indicates the task is already dormant, ignore it
        abi::E_OBJ => {}
        er => {
            expect_success_aborting(er, &"ter_tsk");
        }
    }

    // Delete the task
    // Safety: Upheld by the caller
    expect_success_aborting(unsafe { abi::del_tsk(deleted_task) }, &"del_tsk");
}

/// Terminates and deletes the calling task.
///
/// Atomicity is not required - i.e., it can be assumed that other threads won't
/// `ter_tsk` the calling task while this function is still in progress. (This
/// property makes it easy to implement this operation on μITRON-derived kernels
/// that don't support `exd_tsk`.)
///
/// # Safety
///
/// The task must be safe to terminate. This is in general not true
/// because there might be pinned references to the task's stack.
unsafe fn terminate_and_delete_current_task() -> ! {
    expect_success_aborting(unsafe { abi::exd_tsk() }, &"exd_tsk");
    // Safety: `exd_tsk` never returns on success
    unsafe { crate::hint::unreachable_unchecked() };
}

pub fn available_parallelism() -> io::Result<NonZero<usize>> {
    super::unsupported()
}
