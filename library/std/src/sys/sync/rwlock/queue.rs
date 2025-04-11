//! Efficient read-write locking without `pthread_rwlock_t`.
//!
//! The readers-writer lock provided by the `pthread` library has a number of problems which make it
//! a suboptimal choice for `std`:
//!
//! * It is non-movable, so it needs to be allocated (lazily, to make the constructor `const`).
//! * `pthread` is an external library, meaning the fast path of acquiring an uncontended lock
//! cannot be inlined.
//! * Some platforms (at least glibc before version 2.25) have buggy implementations that can easily
//! lead to undefined behaviour in safe Rust code when not properly guarded against.
//! * On some platforms (e.g. macOS), the lock is very slow.
//!
//! Therefore, we implement our own [`RwLock`]! Naively, one might reach for a spinlock, but those
//! can be quite [problematic] when the lock is contended.
//!
//! Instead, this [`RwLock`] copies its implementation strategy from the Windows [SRWLOCK] and the
//! [usync] library implementations.
//!
//! Spinning is still used for the fast path, but it is bounded: after spinning fails, threads will
//! locklessly add an information structure ([`Node`]) containing a [`Thread`] handle into a queue
//! of waiters associated with the lock. The lock owner, upon releasing the lock, will scan through
//! the queue and wake up threads as appropriate, and the newly-awoken threads will then try to
//! acquire the lock themselves.
//!
//! The resulting [`RwLock`] is:
//!
//! * adaptive, since it spins before doing any heavyweight parking operations
//! * allocation-free, modulo the per-thread [`Thread`] handle, which is allocated anyways when
//! using threads created by `std`
//! * writer-preferring, even if some readers may still slip through
//! * unfair, which reduces context-switching and thus drastically improves performance
//!
//! and also quite fast in most cases.
//!
//! [problematic]: https://matklad.github.io/2020/01/02/spinlocks-considered-harmful.html
//! [SRWLOCK]: https://learn.microsoft.com/en-us/windows/win32/sync/slim-reader-writer--srw--locks
//! [usync]: https://crates.io/crates/usync
//!
//! # Implementation
//!
//! ## State
//!
//! A single [`AtomicPtr`] is used as state variable. The lowest four bits are used to indicate the
//! meaning of the remaining bits:
//!
//! | [`LOCKED`] | [`QUEUED`] | [`QUEUE_LOCKED`] | [`DOWNGRADED`] | Remaining    |                                                                                                                             |
//! |------------|:-----------|:-----------------|:---------------|:-------------|:----------------------------------------------------------------------------------------------------------------------------|
//! | 0          | 0          | 0                | 0              | 0            | The lock is unlocked, no threads are waiting                                                                                |
//! | 1          | 0          | 0                | 0              | 0            | The lock is write-locked, no threads waiting                                                                                |
//! | 1          | 0          | 0                | 0              | n > 0        | The lock is read-locked with n readers                                                                                      |
//! | 0          | 1          | *                | 0              | `*mut Node`  | The lock is unlocked, but some threads are waiting. Only writers may lock the lock                                          |
//! | 1          | 1          | *                | *              | `*mut Node`  | The lock is locked, but some threads are waiting. If the lock is read-locked, the last queue node contains the reader count |
//!
//! ## Waiter Queue
//!
//! When threads are waiting on the lock (the `QUEUE` bit is set), the lock state points to a queue
//! of waiters, which is implemented as a linked list of nodes stored on the stack to avoid memory
//! allocation.
//!
//! To enable lock-free enqueuing of new nodes to the queue, the linked list is singly-linked upon
//! creation.
//!
//! When the lock is read-locked, the lock count (number of readers) is stored in the last link of
//! the queue. Threads have to traverse the queue to find the last element upon releasing the lock.
//! To avoid having to traverse the entire list every time we want to access the reader count, a
//! pointer to the found tail is cached in the (current) first element of the queue.
//!
//! Also, while the lock is unfair for performance reasons, it is still best to wake the tail node
//! first (FIFO ordering). Since we always pop nodes off the tail of the queue, we must store
//! backlinks to previous nodes so that we can update the `tail` field of the (current) first
//! element of the queue. Adding backlinks is done at the same time as finding the tail (via the
//! function [`find_tail_and_add_backlinks`]), and thus encountering a set tail field on a node
//! indicates that all following nodes in the queue are initialized.
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
//! Access to the queue is controlled by the `QUEUE_LOCKED` bit. Threads will try to set this bit
//! in two cases: one is when a thread enqueues itself and eagerly adds backlinks to the queue
//! (which drastically improves performance), and the other is after a thread unlocks the lock to
//! wake up the next waiter(s).
//!
//! `QUEUE_LOCKED` is set atomically at the same time as the enqueuing/unlocking operations. The
//! thread releasing the `QUEUE_LOCKED` bit will check the state of the lock (in particular, whether
//! a downgrade was requested using the [`DOWNGRADED`] bit) and wake up waiters as appropriate. This
//! guarantees forward progress even if the unlocking thread could not acquire the queue lock.
//!
//! ## Memory Orderings
//!
//! To properly synchronize changes to the data protected by the lock, the lock is acquired and
//! released with [`Acquire`] and [`Release`] ordering, respectively. To propagate the
//! initialization of nodes, changes to the queue lock are also performed using these orderings.

#![forbid(unsafe_op_in_unsafe_fn)]

use crate::cell::OnceCell;
use crate::hint::spin_loop;
use crate::mem;
use crate::ptr::{self, NonNull, null_mut, without_provenance_mut};
use crate::sync::atomic::Ordering::{AcqRel, Acquire, Relaxed, Release};
use crate::sync::atomic::{Atomic, AtomicBool, AtomicPtr};
use crate::thread::{self, Thread};

/// The atomic lock state.
type AtomicState = Atomic<State>;
/// The inner lock state.
type State = *mut ();

const UNLOCKED: State = without_provenance_mut(0);
const LOCKED: usize = 1 << 0;
const QUEUED: usize = 1 << 1;
const QUEUE_LOCKED: usize = 1 << 2;
const DOWNGRADED: usize = 1 << 3;
const SINGLE: usize = 1 << 4;
const STATE: usize = DOWNGRADED | QUEUE_LOCKED | QUEUED | LOCKED;
const NODE_MASK: usize = !STATE;

/// Locking uses exponential backoff. `SPIN_COUNT` indicates how many times the locking operation
/// will be retried.
///
/// In other words, `spin_loop` will be called `2.pow(SPIN_COUNT) - 1` times.
const SPIN_COUNT: usize = 7;

/// Marks the state as write-locked, if possible.
#[inline]
fn write_lock(state: State) -> Option<State> {
    if state.addr() & LOCKED == 0 { Some(state.map_addr(|addr| addr | LOCKED)) } else { None }
}

/// Marks the state as read-locked, if possible.
#[inline]
fn read_lock(state: State) -> Option<State> {
    if state.addr() & QUEUED == 0 && state.addr() != LOCKED {
        Some(without_provenance_mut(state.addr().checked_add(SINGLE)? | LOCKED))
    } else {
        None
    }
}

/// Converts a `State` into a `Node` by masking out the bottom bits of the state, assuming that the
/// state points to a queue node.
///
/// # Safety
///
/// The state must contain a valid pointer to a queue node.
#[inline]
unsafe fn to_node(state: State) -> NonNull<Node> {
    unsafe { NonNull::new_unchecked(state.mask(NODE_MASK)).cast() }
}

/// The representation of a thread waiting on the lock queue.
///
/// We initialize these `Node`s on thread execution stacks to avoid allocation.
///
/// Note that we need an alignment of 16 to ensure that the last 4 bits of any
/// pointers to `Node`s are always zeroed (for the bit flags described in the
/// module-level documentation).
#[repr(align(16))]
struct Node {
    next: AtomicLink,
    prev: AtomicLink,
    tail: AtomicLink,
    write: bool,
    thread: OnceCell<Thread>,
    completed: Atomic<bool>,
}

/// An atomic node pointer with relaxed operations.
struct AtomicLink(Atomic<*mut Node>);

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

impl Node {
    /// Creates a new queue node.
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

    /// Prepare this node for waiting.
    fn prepare(&mut self) {
        // Fall back to creating an unnamed `Thread` handle to allow locking in TLS destructors.
        self.thread.get_or_init(thread::current_or_unnamed);
        self.completed = AtomicBool::new(false);
    }

    /// Wait until this node is marked as [`complete`](Node::complete)d by another thread.
    ///
    /// # Safety
    ///
    /// May only be called from the thread that created the node.
    unsafe fn wait(&self) {
        while !self.completed.load(Acquire) {
            unsafe {
                self.thread.get().unwrap().park();
            }
        }
    }

    /// Atomically mark this node as completed.
    ///
    /// # Safety
    ///
    /// `node` must point to a valid `Node`, and the node may not outlive this call.
    unsafe fn complete(node: NonNull<Node>) {
        // Since the node may be destroyed immediately after the completed flag is set, clone the
        // thread handle before that.
        let thread = unsafe { node.as_ref().thread.get().unwrap().clone() };
        unsafe {
            node.as_ref().completed.store(true, Release);
        }
        thread.unpark();
    }
}

/// Traverse the queue and find the tail, adding backlinks to the queue while traversing.
///
/// This may be called from multiple threads at the same time as long as the queue is not being
/// modified (this happens when unlocking multiple readers).
///
/// # Safety
///
/// * `head` must point to a node in a valid queue.
/// * `head` must be in front of the previous head node that was used to perform the last removal.
/// * The part of the queue starting with `head` must not be modified during this call.
unsafe fn find_tail_and_add_backlinks(head: NonNull<Node>) -> NonNull<Node> {
    let mut current = head;

    // Traverse the queue until we find a node that has a set `tail`.
    let tail = loop {
        let c = unsafe { current.as_ref() };
        if let Some(tail) = c.tail.get() {
            break tail;
        }

        // SAFETY: All `next` fields before the first node with a set `tail` are non-null and valid
        // (by Invariant 3).
        unsafe {
            let next = c.next.get().unwrap_unchecked();
            next.as_ref().prev.set(Some(current));
            current = next;
        }
    };

    unsafe {
        head.as_ref().tail.set(Some(tail));
        tail
    }
}

/// [`complete`](Node::complete)s all threads in the queue ending with `tail`.
///
/// # Safety
///
/// * `tail` must be a valid tail of a fully linked queue.
/// * The current thread must have exclusive access to that queue.
unsafe fn complete_all(tail: NonNull<Node>) {
    let mut current = tail;

    // Traverse backwards through the queue (FIFO) and `complete` all of the nodes.
    loop {
        let prev = unsafe { current.as_ref().prev.get() };
        unsafe {
            Node::complete(current);
        }
        match prev {
            Some(prev) => current = prev,
            None => return,
        }
    }
}

/// A type to guard against the unwinds of stacks that nodes are located on due to panics.
struct PanicGuard;

impl Drop for PanicGuard {
    fn drop(&mut self) {
        rtabort!("tried to drop node in intrusive list.");
    }
}

/// The public inner `RwLock` type.
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
        self.state.fetch_update(Acquire, Relaxed, read_lock).is_ok()
    }

    #[inline]
    pub fn read(&self) {
        if !self.try_read() {
            self.lock_contended(false)
        }
    }

    #[inline]
    pub fn try_write(&self) -> bool {
        // Atomically set the `LOCKED` bit. This is lowered to a single atomic instruction on most
        // modern processors (e.g. "lock bts" on x86 and "ldseta" on modern AArch64), and therefore
        // is more efficient than `fetch_update(lock(true))`, which can spuriously fail if a new
        // node is appended to the queue.
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
        let mut node = Node::new(write);
        let mut state = self.state.load(Relaxed);
        let mut count = 0;
        let update_fn = if write { write_lock } else { read_lock };

        loop {
            // Optimistically update the state.
            if let Some(next) = update_fn(state) {
                // The lock is available, try locking it.
                match self.state.compare_exchange_weak(state, next, Acquire, Relaxed) {
                    Ok(_) => return,
                    Err(new) => state = new,
                }
                continue;
            } else if state.addr() & QUEUED == 0 && count < SPIN_COUNT {
                // If the lock is not available and no threads are queued, optimistically spin for a
                // while, using exponential backoff to decrease cache contention.
                for _ in 0..(1 << count) {
                    spin_loop();
                }
                state = self.state.load(Relaxed);
                count += 1;
                continue;
            }
            // The optimistic paths did not succeed, so fall back to parking the thread.

            // First, prepare the node.
            node.prepare();

            // If there are threads queued, this will set the `next` field to be a pointer to the
            // first node in the queue.
            // If the state is read-locked, this will set `next` to the lock count.
            // If it is write-locked, it will set `next` to zero.
            node.next.0 = AtomicPtr::new(state.mask(NODE_MASK).cast());
            node.prev = AtomicLink::new(None);

            // Set the `QUEUED` bit and preserve the `LOCKED` and `DOWNGRADED` bit.
            let mut next = ptr::from_ref(&node)
                .map_addr(|addr| addr | QUEUED | (state.addr() & (DOWNGRADED | LOCKED)))
                as State;

            let mut is_queue_locked = false;
            if state.addr() & QUEUED == 0 {
                // If this is the first node in the queue, set the `tail` field to the node itself
                // to ensure there is a valid `tail` field in the queue (Invariants 1 & 2).
                // This needs to use `set` to avoid invalidating the new pointer.
                node.tail.set(Some(NonNull::from(&node)));
            } else {
                // Otherwise, the tail of the queue is not known.
                node.tail.set(None);

                // Try locking the queue to eagerly add backlinks.
                next = next.map_addr(|addr| addr | QUEUE_LOCKED);

                // Track if we changed the `QUEUE_LOCKED` bit from off to on.
                is_queue_locked = state.addr() & QUEUE_LOCKED == 0;
            }

            // Register the node, using release ordering to propagate our changes to the waking
            // thread.
            if let Err(new) = self.state.compare_exchange_weak(state, next, AcqRel, Relaxed) {
                // The state has changed, just try again.
                state = new;
                continue;
            }
            // The node has been registered, so the structure must not be mutably accessed or
            // destroyed while other threads may be accessing it.

            // Guard against unwinds using a `PanicGuard` that aborts when dropped.
            let guard = PanicGuard;

            // If the current thread locked the queue, unlock it to eagerly adding backlinks.
            if is_queue_locked {
                // SAFETY: This thread set the `QUEUE_LOCKED` bit above.
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

    #[inline]
    pub unsafe fn read_unlock(&self) {
        match self.state.fetch_update(Release, Acquire, |state| {
            if state.addr() & QUEUED == 0 {
                // If there are no threads queued, simply decrement the reader count.
                let count = state.addr() - (SINGLE | LOCKED);
                Some(if count > 0 { without_provenance_mut(count | LOCKED) } else { UNLOCKED })
            } else if state.addr() & DOWNGRADED != 0 {
                // This thread used to have exclusive access, but requested a downgrade. This has
                // not been completed yet, so we still have exclusive access.
                // Retract the downgrade request and unlock, but leave waking up new threads to the
                // thread that already holds the queue lock.
                Some(state.mask(!(DOWNGRADED | LOCKED)))
            } else {
                None
            }
        }) {
            Ok(_) => {}
            // There are waiters queued and the lock count was moved to the tail of the queue.
            Err(state) => unsafe { self.read_unlock_contended(state) },
        }
    }

    /// # Safety
    ///
    /// * There must be threads queued on the lock.
    /// * `state` must be a pointer to a node in a valid queue.
    /// * There cannot be a `downgrade` in progress.
    #[cold]
    unsafe fn read_unlock_contended(&self, state: State) {
        // SAFETY:
        // The state was observed with acquire ordering above, so the current thread will have
        // observed all node initializations.
        // We also know that no threads can be modifying the queue starting at `state`: because new
        // read-locks cannot be acquired while there are any threads queued on the lock, all
        // queue-lock owners will observe a set `LOCKED` bit in `self.state` and will not modify
        // the queue. The other case that a thread could modify the queue is if a downgrade is in
        // progress (removal of the entire queue), but since that is part of this function's safety
        // contract, we can guarantee that no other threads can modify the queue.
        let tail = unsafe { find_tail_and_add_backlinks(to_node(state)).as_ref() };

        // The lock count is stored in the `next` field of `tail`.
        // Decrement it, making sure to observe all changes made to the queue by the other lock
        // owners by using acquire-release ordering.
        let was_last = tail.next.0.fetch_byte_sub(SINGLE, AcqRel).addr() - SINGLE == 0;
        if was_last {
            // SAFETY: Other threads cannot read-lock while threads are queued. Also, the `LOCKED`
            // bit is still set, so there are no writers. Thus the current thread exclusively owns
            // this lock, even though it is a reader.
            unsafe { self.unlock_contended(state) }
        }
    }

    #[inline]
    pub unsafe fn write_unlock(&self) {
        if let Err(state) =
            self.state.compare_exchange(without_provenance_mut(LOCKED), UNLOCKED, Release, Relaxed)
        {
            // SAFETY: Since other threads cannot acquire the lock, the state can only have changed
            // because there are threads queued on the lock.
            unsafe { self.unlock_contended(state) }
        }
    }

    /// # Safety
    ///
    /// * The lock must be exclusively owned by this thread.
    /// * There must be threads queued on the lock.
    /// * `state` must be a pointer to a node in a valid queue.
    /// * There cannot be a `downgrade` in progress.
    #[cold]
    unsafe fn unlock_contended(&self, state: State) {
        debug_assert_eq!(state.addr() & (DOWNGRADED | QUEUED | LOCKED), QUEUED | LOCKED);

        let mut current = state;

        // We want to atomically release the lock and try to acquire the queue lock.
        loop {
            // First check if the queue lock is already held.
            if current.addr() & QUEUE_LOCKED != 0 {
                // Another thread holds the queue lock, so let them wake up waiters for us.
                let next = current.mask(!LOCKED);
                match self.state.compare_exchange_weak(current, next, Release, Relaxed) {
                    Ok(_) => return,
                    Err(new) => {
                        current = new;
                        continue;
                    }
                }
            }

            // Atomically release the lock and try to acquire the queue lock.
            let next = current.map_addr(|addr| (addr & !LOCKED) | QUEUE_LOCKED);
            match self.state.compare_exchange_weak(current, next, AcqRel, Relaxed) {
                // Now that we have the queue lock, we can wake up the next waiter.
                Ok(_) => {
                    // SAFETY: This thread just acquired the queue lock, and this function's safety
                    // contract requires that there are threads already queued on the lock.
                    unsafe { self.unlock_queue(next) };
                    return;
                }
                Err(new) => current = new,
            }
        }
    }

    /// # Safety
    ///
    /// * The lock must be write-locked by this thread.
    #[inline]
    pub unsafe fn downgrade(&self) {
        // Optimistically change the state from write-locked with a single writer and no waiters to
        // read-locked with a single reader and no waiters.
        if let Err(state) = self.state.compare_exchange(
            without_provenance_mut(LOCKED),
            without_provenance_mut(SINGLE | LOCKED),
            Release,
            Relaxed,
        ) {
            // SAFETY: The only way the state can have changed is if there are threads queued.
            // Wake all of them up.
            unsafe { self.downgrade_slow(state) }
        }
    }

    /// Downgrades the lock from write-locked to read-locked in the case that there are threads
    /// waiting on the wait queue.
    ///
    /// This function will either wake up all of the waiters on the wait queue or designate the
    /// current holder of the queue lock to wake up all of the waiters instead. Once the waiters
    /// wake up, they will continue in the execution loop of `lock_contended`.
    ///
    /// # Safety
    ///
    /// * The lock must be write-locked by this thread.
    /// * `state` must be a pointer to a node in a valid queue.
    /// * There must be threads queued on the lock.
    #[cold]
    unsafe fn downgrade_slow(&self, mut state: State) {
        debug_assert_eq!(state.addr() & (DOWNGRADED | QUEUED | LOCKED), QUEUED | LOCKED);

        // Attempt to wake up all waiters by taking ownership of the entire waiter queue.
        loop {
            if state.addr() & QUEUE_LOCKED != 0 {
                // Another thread already holds the queue lock. Tell it to wake up all waiters.
                // If the other thread succeeds in waking up waiters before we release our lock, the
                // effect will be just the same as if we had changed the state below.
                // Otherwise, the `DOWNGRADED` bit will still be set, meaning that when this thread
                // calls `read_unlock` later (because it holds a read lock and must unlock
                // eventually), it will realize that the lock is still exclusively locked and act
                // accordingly.
                let next = state.map_addr(|addr| addr | DOWNGRADED);
                match self.state.compare_exchange_weak(state, next, Release, Relaxed) {
                    Ok(_) => return,
                    Err(new) => state = new,
                }
            } else {
                // Grab the entire queue by swapping the `state` with a single reader.
                let next = ptr::without_provenance_mut(SINGLE | LOCKED);
                if let Err(new) = self.state.compare_exchange_weak(state, next, AcqRel, Relaxed) {
                    state = new;
                    continue;
                }

                // SAFETY: We have full ownership of this queue now, so nobody else can modify it.
                let tail = unsafe { find_tail_and_add_backlinks(to_node(state)) };

                // Wake up all waiters.
                // SAFETY: `tail` was just computed, meaning the whole queue is linked, and we have
                // full ownership of the queue, so we have exclusive access.
                unsafe { complete_all(tail) };

                return;
            }
        }
    }

    /// Unlocks the queue. Wakes up all threads if a downgrade was requested, otherwise wakes up the
    /// next eligible thread(s) if the lock is unlocked.
    ///
    /// # Safety
    ///
    /// * The queue lock must be held by the current thread.
    /// * `state` must be a pointer to a node in a valid queue.
    /// * There must be threads queued on the lock.
    unsafe fn unlock_queue(&self, mut state: State) {
        debug_assert_eq!(state.addr() & (QUEUED | QUEUE_LOCKED), QUEUED | QUEUE_LOCKED);

        loop {
            // SAFETY: Since we have the queue lock, nobody else can be modifying the queue.
            let tail = unsafe { find_tail_and_add_backlinks(to_node(state)) };

            if state.addr() & (DOWNGRADED | LOCKED) == LOCKED {
                // Another thread has locked the lock and no downgrade was requested.
                // Leave waking up waiters to them by releasing the queue lock.
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

            // Since we hold the queue lock and downgrades cannot be requested if the lock is
            // already read-locked, we have exclusive control over the queue here and can make
            // modifications.

            let downgrade = state.addr() & DOWNGRADED != 0;
            let is_writer = unsafe { tail.as_ref().write };
            if !downgrade
                && is_writer
                && let Some(prev) = unsafe { tail.as_ref().prev.get() }
            {
                // If we are not downgrading and the next thread is a writer, only wake up that
                // writing thread.

                // Split off `tail`.
                // There are no set `tail` links before the node pointed to by `state`, so the first
                // non-null tail field will be current (Invariant 2).
                // We also fulfill Invariant 4 since `find_tail` was called on this node, which
                // ensures all backlinks are set.
                unsafe {
                    to_node(state).as_ref().tail.set(Some(prev));
                }

                // Try to release the queue lock. We need to check the state again since another
                // thread might have acquired the lock and requested a downgrade.
                let next = state.mask(!QUEUE_LOCKED);
                if let Err(new) = self.state.compare_exchange_weak(state, next, Release, Acquire) {
                    // Undo the tail modification above, so that we can find the tail again above.
                    // As mentioned above, we have exclusive control over the queue, so no other
                    // thread could have noticed the change.
                    unsafe {
                        to_node(state).as_ref().tail.set(Some(tail));
                    }
                    state = new;
                    continue;
                }

                // The tail was split off and the lock was released. Mark the node as completed.
                unsafe {
                    return Node::complete(tail);
                }
            } else {
                // We are either downgrading, the next waiter is a reader, or the queue only
                // consists of one waiter. In any case, just wake all threads.

                // Clear the queue.
                let next =
                    if downgrade { ptr::without_provenance_mut(SINGLE | LOCKED) } else { UNLOCKED };
                if let Err(new) = self.state.compare_exchange_weak(state, next, Release, Acquire) {
                    state = new;
                    continue;
                }

                // SAFETY: we computed `tail` above, and no new nodes can have been added since
                // (otherwise the CAS above would have failed).
                // Thus we have complete control over the whole queue.
                unsafe {
                    return complete_all(tail);
                }
            }
        }
    }
}
