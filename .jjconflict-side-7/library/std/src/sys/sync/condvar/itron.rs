//! POSIX conditional variable implementation based on user-space wait queues.

use crate::mem::replace;
use crate::ptr::NonNull;
use crate::sys::pal::itron::error::expect_success_aborting;
use crate::sys::pal::itron::spin::SpinMutex;
use crate::sys::pal::itron::time::with_tmos_strong;
use crate::sys::pal::itron::{abi, task};
use crate::sys::sync::Mutex;
use crate::time::Duration;

// The implementation is inspired by the queue-based implementation shown in
// Andrew D. Birrell's paper "Implementing Condition Variables with Semaphores"

pub struct Condvar {
    waiters: SpinMutex<waiter_queue::WaiterQueue>,
}

unsafe impl Send for Condvar {}
unsafe impl Sync for Condvar {}

impl Condvar {
    #[inline]
    pub const fn new() -> Condvar {
        Condvar { waiters: SpinMutex::new(waiter_queue::WaiterQueue::new()) }
    }

    pub fn notify_one(&self) {
        self.waiters.with_locked(|waiters| {
            if let Some(task) = waiters.pop_front() {
                // Unpark the task
                match unsafe { abi::wup_tsk(task) } {
                    // The task already has a token.
                    abi::E_QOVR => {}
                    // Can't undo the effect; abort the program on failure
                    er => {
                        expect_success_aborting(er, &"wup_tsk");
                    }
                }
            }
        });
    }

    pub fn notify_all(&self) {
        self.waiters.with_locked(|waiters| {
            while let Some(task) = waiters.pop_front() {
                // Unpark the task
                match unsafe { abi::wup_tsk(task) } {
                    // The task already has a token.
                    abi::E_QOVR => {}
                    // Can't undo the effect; abort the program on failure
                    er => {
                        expect_success_aborting(er, &"wup_tsk");
                    }
                }
            }
        });
    }

    pub unsafe fn wait(&self, mutex: &Mutex) {
        // Construct `Waiter`.
        let mut waiter = waiter_queue::Waiter::new();
        let waiter = NonNull::from(&mut waiter);

        self.waiters.with_locked(|waiters| unsafe {
            waiters.insert(waiter);
        });

        unsafe { mutex.unlock() };

        // Wait until `waiter` is removed from the queue
        loop {
            // Park the current task
            expect_success_aborting(unsafe { abi::slp_tsk() }, &"slp_tsk");

            if !self.waiters.with_locked(|waiters| unsafe { waiters.is_queued(waiter) }) {
                break;
            }
        }

        mutex.lock();
    }

    pub unsafe fn wait_timeout(&self, mutex: &Mutex, dur: Duration) -> bool {
        // Construct and pin `Waiter`
        let mut waiter = waiter_queue::Waiter::new();
        let waiter = NonNull::from(&mut waiter);

        self.waiters.with_locked(|waiters| unsafe {
            waiters.insert(waiter);
        });

        unsafe { mutex.unlock() };

        // Park the current task and do not wake up until the timeout elapses
        // or the task gets woken up by `notify_*`
        match with_tmos_strong(dur, |tmo| {
            let er = unsafe { abi::tslp_tsk(tmo) };
            if er == 0 {
                // We were unparked. Are we really dequeued?
                if self.waiters.with_locked(|waiters| unsafe { waiters.is_queued(waiter) }) {
                    // No we are not. Continue waiting.
                    return abi::E_TMOUT;
                }
            }
            er
        }) {
            abi::E_TMOUT => {}
            er => {
                expect_success_aborting(er, &"tslp_tsk");
            }
        }

        // Remove `waiter` from `self.waiters`. If `waiter` is still in
        // `waiters`, it means we woke up because of a timeout. Otherwise,
        // we woke up because of `notify_*`.
        let success = self.waiters.with_locked(|waiters| unsafe { !waiters.remove(waiter) });

        mutex.lock();
        success
    }
}

mod waiter_queue {
    use super::*;

    pub struct WaiterQueue {
        head: Option<ListHead>,
    }

    #[derive(Copy, Clone)]
    struct ListHead {
        first: NonNull<Waiter>,
        last: NonNull<Waiter>,
    }

    unsafe impl Send for ListHead {}
    unsafe impl Sync for ListHead {}

    pub struct Waiter {
        // These fields are only accessed through `&[mut] WaiterQueue`.
        /// The waiting task's ID. Will be zeroed when the task is woken up
        /// and removed from a queue.
        task: abi::ID,
        priority: abi::PRI,
        prev: Option<NonNull<Waiter>>,
        next: Option<NonNull<Waiter>>,
    }

    unsafe impl Send for Waiter {}
    unsafe impl Sync for Waiter {}

    impl Waiter {
        #[inline]
        pub fn new() -> Self {
            let task = task::current_task_id();
            let priority = task::task_priority(abi::TSK_SELF);

            // Zeroness of `Waiter::task` indicates whether the `Waiter` is
            // linked to a queue or not. This invariant is important for
            // the correctness.
            debug_assert_ne!(task, 0);

            Self { task, priority, prev: None, next: None }
        }
    }

    impl WaiterQueue {
        #[inline]
        pub const fn new() -> Self {
            Self { head: None }
        }

        /// # Safety
        ///
        ///  - The caller must own `*waiter_ptr`. The caller will lose the
        ///    ownership until `*waiter_ptr` is removed from `self`.
        ///
        ///  - `*waiter_ptr` must be valid until it's removed from the queue.
        ///
        ///  - `*waiter_ptr` must not have been previously inserted to a `WaiterQueue`.
        ///
        pub unsafe fn insert(&mut self, mut waiter_ptr: NonNull<Waiter>) {
            unsafe {
                let waiter = waiter_ptr.as_mut();

                debug_assert!(waiter.prev.is_none());
                debug_assert!(waiter.next.is_none());

                if let Some(head) = &mut self.head {
                    // Find the insertion position and insert `waiter`
                    let insert_after = {
                        let mut cursor = head.last;
                        loop {
                            if waiter.priority >= cursor.as_ref().priority {
                                // `cursor` and all previous waiters have the same or higher
                                // priority than `current_task_priority`. Insert the new
                                // waiter right after `cursor`.
                                break Some(cursor);
                            }
                            cursor = if let Some(prev) = cursor.as_ref().prev {
                                prev
                            } else {
                                break None;
                            };
                        }
                    };

                    if let Some(mut insert_after) = insert_after {
                        // Insert `waiter` after `insert_after`
                        let insert_before = insert_after.as_ref().next;

                        waiter.prev = Some(insert_after);
                        insert_after.as_mut().next = Some(waiter_ptr);

                        waiter.next = insert_before;
                        if let Some(mut insert_before) = insert_before {
                            insert_before.as_mut().prev = Some(waiter_ptr);
                        } else {
                            head.last = waiter_ptr;
                        }
                    } else {
                        // Insert `waiter` to the front
                        waiter.next = Some(head.first);
                        head.first.as_mut().prev = Some(waiter_ptr);
                        head.first = waiter_ptr;
                    }
                } else {
                    // `waiter` is the only element
                    self.head = Some(ListHead { first: waiter_ptr, last: waiter_ptr });
                }
            }
        }

        /// Given a `Waiter` that was previously inserted to `self`, remove
        /// it from `self` if it's still there.
        #[inline]
        pub unsafe fn remove(&mut self, mut waiter_ptr: NonNull<Waiter>) -> bool {
            unsafe {
                let waiter = waiter_ptr.as_mut();
                if waiter.task != 0 {
                    let head = self.head.as_mut().unwrap();

                    match (waiter.prev, waiter.next) {
                        (Some(mut prev), Some(mut next)) => {
                            prev.as_mut().next = Some(next);
                            next.as_mut().prev = Some(prev);
                        }
                        (None, Some(mut next)) => {
                            head.first = next;
                            next.as_mut().prev = None;
                        }
                        (Some(mut prev), None) => {
                            prev.as_mut().next = None;
                            head.last = prev;
                        }
                        (None, None) => {
                            self.head = None;
                        }
                    }

                    waiter.task = 0;

                    true
                } else {
                    false
                }
            }
        }

        /// Given a `Waiter` that was previously inserted to `self`, return a
        /// flag indicating whether it's still in `self`.
        #[inline]
        pub unsafe fn is_queued(&self, waiter: NonNull<Waiter>) -> bool {
            unsafe { waiter.as_ref().task != 0 }
        }

        #[inline]
        pub fn pop_front(&mut self) -> Option<abi::ID> {
            unsafe {
                let head = self.head.as_mut()?;
                let waiter = head.first.as_mut();

                // Get the ID
                let id = replace(&mut waiter.task, 0);

                // Unlink the waiter
                if let Some(mut next) = waiter.next {
                    head.first = next;
                    next.as_mut().prev = None;
                } else {
                    self.head = None;
                }

                Some(id)
            }
        }
    }
}
