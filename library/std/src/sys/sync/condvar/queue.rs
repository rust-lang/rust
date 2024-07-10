//! A generic `Condvar` implementation based on thread parking and a lockless
//! queue of threads.
//!
//! Not all platforms provide an efficient `Condvar` implementation: the UNIX
//! `pthread_condvar_t` needs memory allocation, while SGX doesn't have
//! synchronization primitives at all. Therefore, we implement our own.
//!
//! To do so, we keep a list of the [`Thread`]s waiting on the `Condvar` and
//! wake them up as needed. Access to the list is controlled by an atomic
//! counter. To notify a waiter, the counter is incremented. If the counter
//! was previously zero, the notifying thread has control over the list and
//! will wake up threads until the number of threads it has woken up equals
//! the counter value. Therefore, other threads do not need to wait for control
//! over the list because the controlling thread will take over their notification.
//!
//! This counter is embedded into the lower bits of a pointer to the list head.
//! As that limits the number of in-flight notifications, the counter increments
//! are saturated to a maximum value ([`ALL`]) that causes all threads to be woken
//! up, leading to a spurious wakeup for some threads. The API of `Condvar` permits
//! this however. Timeouts employ the same method to make sure that the current
//! thread handle is removed from the list.
//!
//! The list itself has the same structure as the one used by the queue-based
//! `RwLock` implementation, see its documentation for more information. This
//! enables the lockless enqueuing of threads and results in `Condvar` being
//! only a pointer in size.
//!
//! This implementation is loosely based upon the lockless `Condvar` in
//! [`usync`](https://github.com/kprotty/usync/blob/8937bb77963f6bf9068e56ad46133e933eb79974/src/condvar.rs).

#![forbid(unsafe_op_in_unsafe_fn)]

use crate::cell::UnsafeCell;
use crate::mem::forget;
use crate::ptr::{self, NonNull};
use crate::sync::atomic::Ordering::{AcqRel, Acquire, Relaxed, Release};
use crate::sync::atomic::{AtomicBool, AtomicPtr};
use crate::sys::sync::Mutex;
use crate::thread::{self, Thread};
use crate::time::{Duration, Instant};

type State = *mut ();

const EMPTY: State = ptr::null_mut();
const ALL: usize = 0b1111;
const MASK: usize = !ALL;

fn count(state: State) -> usize {
    state.addr() & ALL
}

unsafe fn to_node(state: State) -> NonNull<Node> {
    unsafe { NonNull::new_unchecked(state.mask(MASK)).cast() }
}

struct PanicGuard;
impl Drop for PanicGuard {
    fn drop(&mut self) {
        rtabort!("tried to drop node in intrusive list.");
    }
}

#[repr(align(16))]
struct Node {
    // Accesses to these `UnsafeCell`s may only be made from the thread that
    // first increment the wakeup count.
    next: UnsafeCell<Option<NonNull<Node>>>,
    prev: UnsafeCell<Option<NonNull<Node>>>,
    tail: UnsafeCell<Option<NonNull<Node>>>,
    notified: AtomicBool,
    thread: Thread,
}

impl Node {
    unsafe fn notify(node: NonNull<Node>) {
        let thread = unsafe { node.as_ref().thread.clone() };
        unsafe {
            node.as_ref().notified.store(true, Release);
        }
        thread.unpark();
    }
}

/// Scan through the list until the `next` pointer of the current node equals
/// `known`, then return that node. Add backlinks to all encountered nodes.
unsafe fn scan_until_known(mut scan: NonNull<Node>, known: NonNull<Node>) -> NonNull<Node> {
    loop {
        let next = unsafe { scan.as_ref().next.get().read().unwrap_unchecked() };
        if next != known {
            unsafe {
                next.as_ref().prev.get().write(Some(scan));
                scan = next;
            }
        } else {
            return scan;
        }
    }
}

/// Scan until encountering a node with a non-empty `tail` field, then return
/// the value of that field. Add backlinks to all encountered nodes.
unsafe fn scan_until_tail(mut scan: NonNull<Node>) -> NonNull<Node> {
    loop {
        let s = unsafe { scan.as_ref() };
        match unsafe { s.tail.get().read() } {
            Some(tail) => return tail,
            None => unsafe {
                let next = s.next.get().read().unwrap_unchecked();
                next.as_ref().prev.get().write(Some(scan));
                scan = next;
            },
        }
    }
}

/// Notify all nodes, going backwards starting with `tail`.
unsafe fn notify_all(mut tail: NonNull<Node>) {
    loop {
        let prev = unsafe { tail.as_ref().prev.get().read() };
        unsafe {
            Node::notify(tail);
        }
        match prev {
            Some(prev) => tail = prev,
            None => return,
        }
    }
}

pub struct Condvar {
    state: AtomicPtr<()>,
}

impl Condvar {
    #[inline]
    pub const fn new() -> Condvar {
        Condvar { state: AtomicPtr::new(ptr::null_mut()) }
    }

    pub unsafe fn wait(&self, mutex: &Mutex) {
        unsafe {
            self.wait_optional_timeout(mutex, None);
        }
    }

    pub unsafe fn wait_timeout(&self, mutex: &Mutex, timeout: Duration) -> bool {
        let timeout = Instant::now().checked_add(timeout);
        unsafe { self.wait_optional_timeout(mutex, timeout) }
    }

    unsafe fn wait_optional_timeout(&self, mutex: &Mutex, timeout: Option<Instant>) -> bool {
        let node = &Node {
            next: UnsafeCell::new(None),
            prev: UnsafeCell::new(None),
            tail: UnsafeCell::new(None),
            notified: AtomicBool::new(false),
            thread: thread::try_current().unwrap_or_else(|| Thread::new_unnamed()),
        };

        // Enqueue the node.
        let mut state = self.state.load(Relaxed);
        loop {
            unsafe {
                node.next.get().write(NonNull::new(state.mask(MASK).cast()));
                node.tail.get().write(if state == EMPTY {
                    Some(NonNull::from(node).cast())
                } else {
                    None
                });
            }

            let next = ptr::from_ref(node).wrapping_byte_add(count(state)) as State;
            match self.state.compare_exchange_weak(state, next, AcqRel, Relaxed) {
                Ok(_) => break,
                Err(new) => state = new,
            }
        }

        // The node is registered, so the structure must not be
        // mutably accessed or destroyed while other threads may
        // be accessing it. Guard against unwinds using a panic
        // guard that aborts when dropped.
        let guard = PanicGuard;

        unsafe {
            mutex.unlock();
        }

        let mut timed_out = false;
        if let Some(timeout) = timeout {
            // While we haven't timed out or been notified, keep parking this thread.
            while !node.notified.load(Acquire) {
                if let Some(remaining) = timeout.checked_duration_since(Instant::now()) {
                    unsafe {
                        node.thread.park_timeout(remaining);
                    }
                } else {
                    timed_out = true;
                    break;
                }
            }

            if timed_out {
                // The node is still in the queue. Wakeup all threads so that
                // it is removed.
                self.notify_all();
            } else {
                // The node was marked as notified, so it is no longer part of
                // the queue. Relock the mutex and return.
                forget(guard);
                mutex.lock();
                return true;
            }
        }

        // Park the thread until we are notified.
        while !node.notified.load(Acquire) {
            unsafe {
                node.thread.park();
            }
        }

        // The node was marked as notified, so it is no longer part of
        // the queue. Relock the mutex and return.
        forget(guard);
        mutex.lock();
        !timed_out
    }

    pub fn notify_one(&self) {
        // Try to increase the notification counter.
        let mut state = self.state.load(Relaxed);
        loop {
            if state == EMPTY {
                return;
            }

            if count(state) == ALL {
                // All threads are being notified, so we don't need to do another
                // notification.
                return;
            } else if count(state) != 0 {
                // Another thread is handling notifications, tell it to notify
                // one more thread.
                let next = state.wrapping_byte_add(1);
                match self.state.compare_exchange_weak(state, next, Relaxed, Relaxed) {
                    Ok(_) => return,
                    Err(new) => state = new,
                }
            } else {
                // No notifications are in progress, we should take responsibility
                // for waking up threads. Increase the notification counter to do so.
                let next = state.wrapping_byte_add(1);
                match self.state.compare_exchange_weak(state, next, Acquire, Relaxed) {
                    Ok(_) => {
                        state = next;
                        break;
                    }
                    Err(new) => state = new,
                }
            }
        }

        // At this point, we took responsibility for notifying threads, meaning
        // we have exclusive access to the queue. Wake up threads as long as there
        // are threads to notify and notifications requested.

        // Keep track of how many threads we notified already.
        let mut notified = 0;
        // This is the node that will be woken up next.
        let mut tail = unsafe { scan_until_tail(to_node(state)) };

        while count(state) != ALL {
            if notified != count(state) {
                // We haven't notified enough threads, so wake up `tail`.

                let prev = unsafe { tail.as_ref().prev.get().read() };

                unsafe {
                    Node::notify(tail);
                }

                notified += 1;

                if let Some(prev) = prev {
                    tail = prev;
                } else {
                    // We notified all threads in the queue. As long as no new
                    // nodes have been added, clear the state.
                    loop {
                        match self.state.compare_exchange_weak(state, EMPTY, Release, Acquire) {
                            Ok(_) => return,
                            Err(new) => state = new,
                        }

                        let head = unsafe { to_node(state) };
                        if head != tail {
                            // `head` has already been woken up, so we may not
                            // access it. Simply continue the main loop with
                            // the last new node.
                            tail = unsafe { scan_until_known(head, tail) };
                            break;
                        }
                    }
                }
            } else {
                // We notified enough threads. Try clearing the counter.

                let head = unsafe { to_node(state) };
                unsafe {
                    head.as_ref().tail.get().write(Some(tail));
                }

                match self.state.compare_exchange_weak(state, state.mask(MASK), Release, Acquire) {
                    Ok(_) => return,
                    Err(new) => state = new,
                }

                let scan = unsafe { to_node(state) };
                if scan != head {
                    // New nodes have been added to the queue. Link the new part
                    // of the queue to the old one.
                    let scan = unsafe { scan_until_known(scan, head) };
                    unsafe {
                        head.as_ref().prev.get().write(Some(scan));
                    }
                }
            }
        }

        // We need to wake up all threads in the queue.
        // Use a swap to reset the state so that we do not endlessly retry if
        // new nodes are constantly being added.

        let new = self.state.swap(EMPTY, Acquire);
        let head = unsafe { to_node(state) };
        let scan = unsafe { to_node(new) };
        if head != scan {
            // New nodes have been added to the queue. Link the new part
            // of the queue to the old one.
            let scan = unsafe { scan_until_known(scan, head) };
            unsafe {
                head.as_ref().prev.get().write(Some(scan));
            }
        }

        unsafe { notify_all(tail) }
    }

    pub fn notify_all(&self) {
        let mut state = self.state.load(Relaxed);
        loop {
            if state == EMPTY {
                return;
            }

            if count(state) == ALL {
                // All threads are already being notified.
                return;
            } else if count(state) != 0 {
                // Another thread is handling notifications, tell it to notify
                // all threads.
                let next = state.map_addr(|state| state | ALL);
                match self.state.compare_exchange_weak(state, next, Relaxed, Relaxed) {
                    Ok(_) => return,
                    Err(new) => state = new,
                }
            } else {
                // Take the whole queue and wake it up.
                match self.state.compare_exchange_weak(state, EMPTY, Acquire, Relaxed) {
                    Ok(_) => break,
                    Err(new) => state = new,
                }
            }
        }

        let tail = unsafe { scan_until_tail(to_node(state)) };
        unsafe { notify_all(tail) }
    }
}
