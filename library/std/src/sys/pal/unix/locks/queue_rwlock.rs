//! Efficient read-write locking without `pthread_rwlock_t`.
//!
//! The readers-writer lock provided by the `pthread` library has a number of
//! problems which make it a suboptimal choice for `std`:
//!
//! * It is non-movable, so it needs to be allocated (lazily, to make the
//! constructor `const`).
//! * `pthread` is an external library, meaning the fast path of acquiring an
//! uncontended lock cannot be inlined.
//! * Some platforms (at least glibc before version 2.25) have buggy implementations
//! that can easily lead to undefined behaviour in safe Rust code when not properly
//! guarded against.
//! * On some platforms (e.g. macOS), the lock is very slow.
//!
//! Therefore, we implement our own `RwLock`! Naively, one might reach for a
//! spinlock, but those [can be quite problematic] when the lock is contended.
//! Instead, this readers-writer lock copies its implementation strategy from
//! the Windows [SRWLOCK] and the [usync] library. Spinning is still used for the
//! fast path, but it is bounded: after spinning fails, threads will locklessly
//! add an information structure containing a [`Thread`] handle into a queue of
//! waiters associated with the lock. The lock owner, upon releasing the lock,
//! will scan through the queue and wake up threads as appropriate, which will
//! then again try to acquire the lock. The resulting [`RwLock`] is:
//!
//! * adaptive, since it spins before doing any heavywheight parking operations
//! * allocation-free, modulo the per-thread [`Thread`] handle, which is
//! allocated regardless when using threads created by `std`
//! * writer-preferring, even if some readers may still slip through
//! * unfair, which reduces context-switching and thus drastically improves
//! performance
//!
//! and also quite fast in most cases.
//!
//! [can be quite problematic]: https://matklad.github.io/2020/01/02/spinlocks-considered-harmful.html
//! [SRWLOCK]: https://learn.microsoft.com/en-us/windows/win32/sync/slim-reader-writer--srw--locks
//! [usync]: https://crates.io/crates/usync
//!
//! # Implementation
//!
//! ## State
//!
//! A single [`AtomicPtr`] is used as state variable. The lowest three bits are used
//! to indicate the meaning of the remaining bits:
//!
//! | [`LOCKED`] | [`QUEUED`] | [`QUEUE_LOCKED`] | Remaining    |                                                                                                                             |
//! |:-----------|:-----------|:-----------------|:-------------|:----------------------------------------------------------------------------------------------------------------------------|
//! | 0          | 0          | 0                | 0            | The lock is unlocked, no threads are waiting                                                                                |
//! | 1          | 0          | 0                | 0            | The lock is write-locked, no threads waiting                                                                                |
//! | 1          | 0          | 0                | n > 0        | The lock is read-locked with n readers                                                                                      |
//! | 0          | 1          | *                | `*mut Node`  | The lock is unlocked, but some threads are waiting. Only writers may lock the lock                                          |
//! | 1          | 1          | *                | `*mut Node`  | The lock is locked, but some threads are waiting. If the lock is read-locked, the last queue node contains the reader count |
//!
//! ## Waiter queue
//!
//! When threads are waiting on the lock (`QUEUE` is set), the lock state
//! points to a queue of waiters, which is implemented as a linked list of
//! nodes stored on the stack to avoid memory allocation. To enable lockless
//! enqueuing of new nodes to the queue, the linked list is single-linked upon
//! creation. Since when the lock is read-locked, the lock count is stored in
//! the last link of the queue, threads have to traverse the queue to find the
//! last element upon releasing the lock. To avoid having to traverse the whole
//! list again and again, a pointer to the found tail is cached in the (current)
//! first element of the queue.
//!
//! Also, while the lock is unfair for performance reasons, it is still best to
//! wake the tail node first, which requires backlinks to previous nodes to be
//! created. This is done at the same time as finding the tail, and thus a set
//! tail field indicates the remaining portion of the queue is initialized.
//!
//! TLDR: Here's a diagram of what the queue looks like:
//!
//! ```text
//! state
//!   │
//!   ▼
//! ╭───────╮ next ╭───────╮ next ╭───────╮ next ╭───────╮
//! │       ├─────►│       ├─────►│       ├─────►│ count │
//! │       │      │       │      │       │      │       │
//! │       │      │       │◄─────┤       │◄─────┤       │
//! ╰───────╯      ╰───────╯ prev ╰───────╯ prev ╰───────╯
//!                      │                           ▲
//!                      └───────────────────────────┘
//!                                  tail
//! ```
//!
//! Invariants:
//! 1. At least one node must contain a non-null, current `tail` field.
//! 2. The first non-null `tail` field must be valid and current.
//! 3. All nodes preceding this node must have a correct, non-null `next` field.
//! 4. All nodes following this node must have a correct, non-null `prev` field.
//!
//! Access to the queue is controlled by the `QUEUE_LOCKED` bit, which threads
//! try to set both after enqueuing themselves to eagerly add backlinks to the
//! queue and after unlocking the lock to wake the next waiter(s). This is done
//! atomically at the same time as the enqueuing/unlocking operation. The thread
//! releasing the `QUEUE_LOCK` bit will check the state of the lock and wake up
//! waiters as appropriate. This guarantees forward-progress even if the unlocking
//! thread could not acquire the queue lock.
//!
//! ## Memory orderings
//!
//! To properly synchronize changes to the data protected by the lock, the lock
//! is acquired and released with [`Acquire`] and [`Release`] ordering, respectively.
//! To propagate the initialization of nodes, changes to the queue lock are also
//! performed using these orderings.

#![forbid(unsafe_op_in_unsafe_fn)]

use crate::cell::OnceCell;
use crate::hint::spin_loop;
use crate::mem;
use crate::ptr::{self, invalid_mut, null_mut, NonNull};
use crate::sync::atomic::{
    AtomicBool, AtomicPtr,
    Ordering::{AcqRel, Acquire, Relaxed, Release},
};
use crate::sys_common::thread_info;
use crate::thread::Thread;

// Locking uses exponential backoff. `SPIN_COUNT` indicates how many times the
// locking operation will be retried.
// `spin_loop` will be called `2.pow(SPIN_COUNT) - 1` times.
const SPIN_COUNT: usize = 7;

type State = *mut ();
type AtomicState = AtomicPtr<()>;

const UNLOCKED: State = invalid_mut(0);
const LOCKED: usize = 1;
const QUEUED: usize = 2;
const QUEUE_LOCKED: usize = 4;
const SINGLE: usize = 8;
const MASK: usize = !(QUEUE_LOCKED | QUEUED | LOCKED);

/// Returns a closure that changes the state to the lock state corresponding to
/// the lock mode indicated in `write`.
#[inline]
fn lock(write: bool) -> impl Fn(State) -> Option<State> {
    move |state| {
        if write {
            let state = state.wrapping_byte_add(LOCKED);
            if state.addr() & LOCKED == LOCKED { Some(state) } else { None }
        } else {
            if state.addr() & QUEUED == 0 && state.addr() != LOCKED {
                Some(invalid_mut(state.addr().checked_add(SINGLE)? | LOCKED))
            } else {
                None
            }
        }
    }
}

/// Masks the state, assuming it points to a queue node.
///
/// # Safety
/// The state must contain a valid pointer to a queue node.
#[inline]
unsafe fn to_node(state: State) -> NonNull<Node> {
    unsafe { NonNull::new_unchecked(state.mask(MASK)).cast() }
}

/// An atomic node pointer with relaxed operations.
struct AtomicLink(AtomicPtr<Node>);

impl AtomicLink {
    fn new(v: Option<NonNull<Node>>) -> AtomicLink {
        AtomicLink(AtomicPtr::new(v.map_or(null_mut(), NonNull::as_ptr)))
    }

    fn get(&self) -> Option<NonNull<Node>> {
        NonNull::new(self.0.load(Relaxed))
    }

    fn set(&self, v: Option<NonNull<Node>>) {
        self.0.store(v.map_or(null_mut(), NonNull::as_ptr), Relaxed);
    }
}

#[repr(align(8))]
struct Node {
    next: AtomicLink,
    prev: AtomicLink,
    tail: AtomicLink,
    write: bool,
    thread: OnceCell<Thread>,
    completed: AtomicBool,
}

impl Node {
    /// Create a new queue node.
    fn new(write: bool) -> Node {
        Node {
            next: AtomicLink::new(None),
            prev: AtomicLink::new(None),
            tail: AtomicLink::new(None),
            write,
            thread: OnceCell::new(),
            completed: AtomicBool::new(false),
        }
    }

    /// Set the `next` field depending on the lock state. If there are threads
    /// queued, the `next` field will be set to a pointer to the next node in
    /// the queue. Otherwise the `next` field will be set to the lock count if
    /// the state is read-locked or to zero if it is write-locked.
    fn set_state(&mut self, state: State) {
        self.next.0 = AtomicPtr::new(state.mask(MASK).cast());
    }

    /// Assuming the node contains a reader lock count, decrement that count.
    /// Returns `true` if this thread was the last lock owner.
    fn decrement_count(&self) -> bool {
        self.next.0.fetch_byte_sub(SINGLE, AcqRel).addr() - SINGLE == 0
    }

    /// Prepare this node for waiting.
    fn prepare(&mut self) {
        // Fall back to creating an unnamed `Thread` handle to allow locking in
        // TLS destructors.
        self.thread
            .get_or_init(|| thread_info::current_thread().unwrap_or_else(|| Thread::new(None)));
        self.completed = AtomicBool::new(false);
    }

    /// Wait until this node is marked as completed.
    ///
    /// # Safety
    /// May only be called from the thread that created the node.
    unsafe fn wait(&self) {
        while !self.completed.load(Acquire) {
            unsafe {
                self.thread.get().unwrap().park();
            }
        }
    }

    /// Atomically mark this node as completed. The node may not outlive this call.
    unsafe fn complete(this: NonNull<Node>) {
        // Since the node may be destroyed immediately after the completed flag
        // is set, clone the thread handle before that.
        let thread = unsafe { this.as_ref().thread.get().unwrap().clone() };
        unsafe {
            this.as_ref().completed.store(true, Release);
        }
        thread.unpark();
    }
}

struct PanicGuard;

impl Drop for PanicGuard {
    fn drop(&mut self) {
        rtabort!("tried to drop node in intrusive list.");
    }
}

/// Find the tail of the queue beginning with `head`, caching the result in `head`.
///
/// May be called from multiple threads at the same time, while the queue is not
/// modified (this happens when unlocking multiple readers).
///
/// # Safety
/// * `head` must point to a node in a valid queue.
/// * `head` must be or be in front of the head of the queue at the time of the
/// last removal.
/// * The part of the queue starting with `head` must not be modified during this
/// call.
unsafe fn find_tail(head: NonNull<Node>) -> NonNull<Node> {
    let mut current = head;
    let tail = loop {
        let c = unsafe { current.as_ref() };
        match c.tail.get() {
            Some(tail) => break tail,
            // SAFETY:
            // All `next` fields before the first node with a `set` tail are
            // non-null and valid (invariant 3).
            None => unsafe {
                let next = c.next.get().unwrap_unchecked();
                next.as_ref().prev.set(Some(current));
                current = next;
            },
        }
    };

    unsafe {
        head.as_ref().tail.set(Some(tail));
        tail
    }
}

pub struct RwLock {
    state: AtomicState,
}

impl RwLock {
    #[inline]
    pub const fn new() -> RwLock {
        RwLock { state: AtomicPtr::new(UNLOCKED) }
    }

    #[inline]
    pub fn try_read(&self) -> bool {
        self.state.fetch_update(Acquire, Relaxed, lock(false)).is_ok()
    }

    #[inline]
    pub fn read(&self) {
        if !self.try_read() {
            self.lock_contended(false)
        }
    }

    #[inline]
    pub fn try_write(&self) -> bool {
        // This is lowered to a single atomic instruction on most modern processors
        // (e.g. "lock bts" on x86 and "ldseta" on modern AArch64), and therefore
        // is more efficient than `fetch_update(lock(true))`, which can spuriously
        // fail if a new node is appended to the queue.
        self.state.fetch_or(LOCKED, Acquire).addr() & LOCKED == 0
    }

    #[inline]
    pub fn write(&self) {
        if !self.try_write() {
            self.lock_contended(true)
        }
    }

    #[cold]
    fn lock_contended(&self, write: bool) {
        let update = lock(write);
        let mut node = Node::new(write);
        let mut state = self.state.load(Relaxed);
        let mut count = 0;
        loop {
            if let Some(next) = update(state) {
                // The lock is available, try locking it.
                match self.state.compare_exchange_weak(state, next, Acquire, Relaxed) {
                    Ok(_) => return,
                    Err(new) => state = new,
                }
            } else if state.addr() & QUEUED == 0 && count < SPIN_COUNT {
                // If the lock is not available and no threads are queued, spin
                // for a while, using exponential backoff to decrease cache
                // contention.
                for _ in 0..(1 << count) {
                    spin_loop();
                }
                state = self.state.load(Relaxed);
                count += 1;
            } else {
                // Fall back to parking. First, prepare the node.
                node.prepare();
                node.set_state(state);
                node.prev = AtomicLink::new(None);
                let mut next = ptr::from_ref(&node)
                    .map_addr(|addr| addr | QUEUED | (state.addr() & LOCKED))
                    as State;

                if state.addr() & QUEUED == 0 {
                    // If this is the first node in the queue, set the tail field to
                    // the node itself to ensure there is a current `tail` field in
                    // the queue (invariants 1 and 2). This needs to use `set` to
                    // avoid invalidating the new pointer.
                    node.tail.set(Some(NonNull::from(&node)));
                } else {
                    // Otherwise, the tail of the queue is not known.
                    node.tail.set(None);
                    // Try locking the queue to fully link it.
                    next = next.map_addr(|addr| addr | QUEUE_LOCKED);
                }

                // Use release ordering to propagate our changes to the waking
                // thread.
                if let Err(new) = self.state.compare_exchange_weak(state, next, AcqRel, Relaxed) {
                    // The state has changed, just try again.
                    state = new;
                    continue;
                }

                // The node is registered, so the structure must not be
                // mutably accessed or destroyed while other threads may
                // be accessing it. Guard against unwinds using a panic
                // guard that aborts when dropped.
                let guard = PanicGuard;

                // If the current thread locked the queue, unlock it again,
                // linking it in the process.
                if state.addr() & (QUEUE_LOCKED | QUEUED) == QUEUED {
                    unsafe {
                        self.unlock_queue(next);
                    }
                }

                // Wait until the node is removed from the queue.
                // SAFETY: the node was created by the current thread.
                unsafe {
                    node.wait();
                }

                // The node was removed from the queue, disarm the guard.
                mem::forget(guard);

                // Reload the state and try again.
                state = self.state.load(Relaxed);
                count = 0;
            }
        }
    }

    #[inline]
    pub unsafe fn read_unlock(&self) {
        match self.state.fetch_update(Release, Acquire, |state| {
            if state.addr() & QUEUED == 0 {
                let count = state.addr() - (SINGLE | LOCKED);
                Some(if count > 0 { invalid_mut(count | LOCKED) } else { UNLOCKED })
            } else {
                None
            }
        }) {
            Ok(_) => {}
            // There are waiters queued and the lock count was moved to the
            // tail of the queue.
            Err(state) => unsafe { self.read_unlock_contended(state) },
        }
    }

    #[cold]
    unsafe fn read_unlock_contended(&self, state: State) {
        // The state was observed with acquire ordering above, so the current
        // thread will observe all node initializations.

        let tail = unsafe { find_tail(to_node(state)) };
        let was_last = unsafe { tail.as_ref().decrement_count() };
        if was_last {
            // SAFETY:
            // Other threads cannot read-lock while threads are queued. Also,
            // the `LOCKED` bit is still set, so there are no writers. Therefore,
            // the current thread exclusively owns the lock.
            unsafe { self.unlock_contended(state) }
        }
    }

    #[inline]
    pub unsafe fn write_unlock(&self) {
        if let Err(state) =
            self.state.compare_exchange(invalid_mut(LOCKED), UNLOCKED, Release, Relaxed)
        {
            // SAFETY:
            // Since other threads cannot acquire the lock, the state can only
            // have changed because there are threads queued on the lock.
            unsafe { self.unlock_contended(state) }
        }
    }

    /// # Safety
    /// * The lock must be exclusively owned by this thread.
    /// * There must be threads queued on the lock.
    #[cold]
    unsafe fn unlock_contended(&self, mut state: State) {
        loop {
            // Atomically release the lock and try to acquire the queue lock.
            let next = state.map_addr(|a| (a & !LOCKED) | QUEUE_LOCKED);
            match self.state.compare_exchange_weak(state, next, AcqRel, Relaxed) {
                // The queue lock was acquired. Release it, waking up the next
                // waiter in the process.
                Ok(_) if state.addr() & QUEUE_LOCKED == 0 => unsafe {
                    return self.unlock_queue(next);
                },
                // Another thread already holds the queue lock, leave waking up
                // waiters to it.
                Ok(_) => return,
                Err(new) => state = new,
            }
        }
    }

    /// # Safety
    /// The queue lock must be held by the current thread.
    unsafe fn unlock_queue(&self, mut state: State) {
        debug_assert_eq!(state.addr() & (QUEUED | QUEUE_LOCKED), QUEUED | QUEUE_LOCKED);

        loop {
            // Find the last node in the linked list.
            let tail = unsafe { find_tail(to_node(state)) };

            if state.addr() & LOCKED == LOCKED {
                // Another thread has locked the lock. Leave waking up waiters
                // to them by releasing the queue lock.
                match self.state.compare_exchange_weak(
                    state,
                    state.mask(!QUEUE_LOCKED),
                    Release,
                    Acquire,
                ) {
                    Ok(_) => return,
                    Err(new) => {
                        state = new;
                        continue;
                    }
                }
            }

            let is_writer = unsafe { tail.as_ref().write };
            if is_writer && let Some(prev) = unsafe { tail.as_ref().prev.get() } {
                // `tail` is a writer and there is a node before `tail`.
                // Split off `tail`.

                // There are no set `tail` links before the node pointed to by
                // `state`, so the first non-null tail field will be current
                // (invariant 2). Invariant 4 is fullfilled since `find_tail`
                // was called on this node, which ensures all backlinks are set.
                unsafe { to_node(state).as_ref().tail.set(Some(prev)); }

                // Release the queue lock. Doing this by subtraction is more
                // efficient on modern processors since it is a single instruction
                // instead of an update loop, which will fail if new threads are
                // added to the list.
                self.state.fetch_byte_sub(QUEUE_LOCKED, Release);

                // The tail was split off and the lock released. Mark the node as
                // completed.
                unsafe { return Node::complete(tail); }
            } else {
                // The next waiter is a reader or the queue only consists of one
                // waiter. Just wake all threads.

                // The lock cannot be locked (checked above), so mark it as
                // unlocked to reset the queue.
                if let Err(new) = self.state.compare_exchange_weak(state, UNLOCKED, Release, Acquire) {
                    state = new;
                    continue;
                }

                let mut current = tail;
                loop {
                    let prev = unsafe { current.as_ref().prev.get() };
                    unsafe { Node::complete(current); }
                    match prev {
                        Some(prev) => current = prev,
                        None => return,
                    }
                }
            }
        }
    }
}
