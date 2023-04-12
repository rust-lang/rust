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
//! A single [`AtomicPtr`] is used as state variable. The lowest two bits are used
//! to indicate the meaning of the remaining bits:
//!
//! | `LOCKED`  | `QUEUED`  | Remaining    |                                                                                                                             |
//! |:----------|:----------|:-------------|:----------------------------------------------------------------------------------------------------------------------------|
//! | 0         | 0         | 0            | The lock is unlocked, no threads are waiting                                                                                |
//! | 1         | 0         | 0            | The lock is write-locked, no threads waiting                                                                                |
//! | 1         | 0         | n > 0        | The lock is read-locked with n readers                                                                                      |
//! | 0         | 1         | `*mut Node`  | The lock is unlocked, but some threads are waiting. Only writers may lock the lock                                          |
//! | 1         | 1         | `*mut Node`  | The lock is locked, but some threads are waiting. If the lock is read-locked, the last queue node contains the reader count |
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
//! 1. The `next` field always points to a valid node, except in the tail node.
//! 2. The `next` field of the tail node must be null while the queue is unlocked.
//! 3. At least one node must contain a non-null, current `tail` field.
//! 4. The first non-null `tail` field must be valid and current.
//! 5. All nodes following this node must have a correct, non-null `prev` field.
//!
//! While adding a new node to the queue may be done by any thread at any time,
//! removing nodes may only be done by a single thread. Instead of using a
//! separate lock bit for the queue like usync does, this implementation
//! only allows the (last) lock owner to modify the queue.
//!
//! ## Memory orderings
//!
//! To properly synchronize changes to the data protected by the lock, the lock
//! is acquired and released with [`Acquire`] and [`Release`] ordering, respectively.
//! To propagate the initialization of nodes, changes to the list are also propagated
//! using these orderings.

#![forbid(unsafe_op_in_unsafe_fn)]

use crate::cell::OnceCell;
use crate::hint::spin_loop;
use crate::ptr::{self, invalid_mut, null_mut, NonNull};
use crate::sync::atomic::{
    AtomicBool, AtomicPtr,
    Ordering::{AcqRel, Acquire, Relaxed, Release},
};
use crate::sys_common::thread_info;
use crate::thread::Thread;

const SPIN_COUNT: usize = 100;

type State = *mut ();
type AtomicState = AtomicPtr<()>;

const UNLOCKED: State = invalid_mut(0);
const LOCKED: usize = 1;
const QUEUED: usize = 2;
const SINGLE: usize = 4;
const MASK: usize = !(LOCKED | QUEUED);

/// Returns a closure that changes the state to the lock state corresponding to
/// the lock mode indicated in `read`.
#[inline]
fn lock(read: bool) -> impl Fn(State) -> Option<State> {
    move |state| {
        if read {
            if state.addr() & QUEUED == 0 && state.addr() != LOCKED {
                Some(invalid_mut(state.addr().checked_add(SINGLE)? | LOCKED))
            } else {
                None
            }
        } else {
            let state = state.wrapping_byte_add(LOCKED);
            if state.addr() & LOCKED == LOCKED { Some(state) } else { None }
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

#[repr(align(4))]
struct Node {
    next: AtomicLink,
    prev: AtomicLink,
    tail: AtomicLink,
    read: bool,
    thread: OnceCell<Thread>,
    completed: AtomicBool,
}

impl Node {
    /// Create a new queue node.
    fn new(read: bool) -> Node {
        Node {
            next: AtomicLink::new(None),
            prev: AtomicLink::new(None),
            tail: AtomicLink::new(None),
            read,
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
    /// Returns `true` if there are other lock owners.
    fn decrement_count(&self) -> bool {
        self.next.0.fetch_byte_sub(SINGLE, AcqRel).addr() > SINGLE
    }

    /// Prepare this node for waiting.
    fn prepare(&mut self) {
        // Fall back to creating an unnamed `Thread` handle to allow locking in
        // TLS destructors.
        self.thread.get_or_init(|| thread_info::current_thread().unwrap_or(Thread::new(None)));
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
            // Only the `next` field of the tail is null (invariants 1. and 2.)
            // Since at least one element in the queue has a non-null tail (invariant 3.),
            // this code will never be run for `current == tail`.
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
        self.state.fetch_update(Acquire, Relaxed, lock(true)).is_ok()
    }

    #[inline]
    pub fn read(&self) {
        if !self.try_read() {
            self.lock_contended(true)
        }
    }

    #[inline]
    pub fn try_write(&self) -> bool {
        // This is lowered to a single atomic instruction on most modern processors
        // (e.g. "lock bts" on x86 and "ldseta" on modern AArch64), and therefore
        // is more efficient than `fetch_update(lock(false))`, which can spuriously
        // fail if a new node is appended to the queue.
        self.state.fetch_or(LOCKED, Acquire).addr() & LOCKED != LOCKED
    }

    #[inline]
    pub fn write(&self) {
        if !self.try_write() {
            self.lock_contended(false)
        }
    }

    #[cold]
    fn lock_contended(&self, read: bool) {
        let update = lock(read);
        let mut node = Node::new(read);
        let mut state = self.state.load(Relaxed);
        let mut count = 0;
        loop {
            if let Some(next) = update(state) {
                // The lock is available, try locking it.
                match self.state.compare_exchange_weak(state, next, Acquire, Relaxed) {
                    Ok(_) => return,
                    Err(new) => state = new,
                }
            } else if count < SPIN_COUNT {
                // If the lock is not available, spin for a while.
                spin_loop();
                state = self.state.load(Relaxed);
                count += 1;
            } else {
                // Fall back to parking. First, prepare the node.
                node.prepare();
                node.set_state(state);
                node.prev = AtomicLink::new(None);
                // If this is the first node in the queue, set the tail field to
                // the node itself to ensure there is a current `tail` field in
                // the queue (invariants 3. and 4.). This needs to use `set` to
                // avoid invalidating the new pointer.
                node.tail.set((state.addr() & QUEUED == 0).then_some(NonNull::from(&node)));

                let next = ptr::from_ref(&node)
                    .map_addr(|addr| addr | QUEUED | (state.addr() & LOCKED))
                    as State;
                // Use release ordering to propagate our changes to the waking
                // thread.
                if let Err(new) = self.state.compare_exchange_weak(state, next, Release, Relaxed) {
                    // The state has changed, just try again.
                    state = new;
                    continue;
                }

                // The node is registered, so the structure must not be
                // mutably accessed or destroyed while other threads may
                // be accessing it. Just wait until it is completed.

                // SAFETY: the node was created by the current thread.
                unsafe {
                    node.wait();
                }

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
            Err(state) => unsafe { self.unlock_contended(state, true) },
        }
    }

    #[inline]
    pub unsafe fn write_unlock(&self) {
        match self.state.compare_exchange(invalid_mut(LOCKED), UNLOCKED, Release, Acquire) {
            Ok(_) => {}
            // Since other threads cannot acquire the lock, the state can only
            // have changed because there are threads queued on the lock.
            Err(state) => unsafe { self.unlock_contended(state, false) },
        }
    }

    /// # Safety
    /// The lock must be locked by the current thread and threads must be queued on it.
    #[cold]
    unsafe fn unlock_contended(&self, mut state: State, read: bool) {
        // Find the last node in the linked queue.
        let tail = unsafe { find_tail(to_node(state)) };
        let not_last = unsafe { read && tail.as_ref().decrement_count() };
        if not_last {
            // There are other lock owners, leave waking up the next waiters to them.
            return;
        }

        // At this point, the `next` field on `tail` will always be null
        // (invariant 2).

        let next_read = unsafe { tail.as_ref().read };
        if next_read {
            // The next waiter is a reader. Just wake all threads.
            //
            // SAFETY:
            // `current` is the head of a valid queue, which no thread except the
            // the current can observe.
            unsafe {
                let mut current = to_node(self.state.swap(UNLOCKED, AcqRel));
                loop {
                    let next = current.as_ref().next.get();
                    Node::complete(current);
                    match next {
                        Some(next) => current = next,
                        None => break,
                    }
                }
            }
        } else {
            // The next waiter is a writer. Remove it from the queue and wake it.
            let prev = match unsafe { tail.as_ref().prev.get() } {
                // If the lock was read-locked, multiple threads have invoked
                // `find_tail` above. Therefore, it is possible that one of
                // them observed a newer state than this thread did, meaning
                // there is a set `tail` field in a node before `state`. To
                // make sure that the queue is valid after the link update
                // below, reload the state and relink the queue.
                //
                // SAFETY: since the current thread holds the lock, the queue
                // was not removed from since the last time and therefore is
                // still valid.
                Some(prev) if read => unsafe {
                    let new = self.state.load(Acquire);
                    if new != state {
                        state = new;
                        find_tail(to_node(state));
                    }
                    Some(prev)
                },
                Some(prev) => Some(prev),
                // The current node is the only one in the queue that we observed.
                // Try setting the state to UNLOCKED.
                None => self.state.compare_exchange(state, UNLOCKED, Release, Acquire).err().map(
                    |new| {
                        state = new;
                        // Since the state was locked, it can only have changed
                        // because a new node was added since `state` was loaded.
                        // Relink the queue and get a pointer to the node before
                        // `tail`.
                        unsafe {
                            find_tail(to_node(state));
                            tail.as_ref().prev.get().unwrap()
                        }
                    },
                ),
            };

            if let Some(prev) = prev {
                unsafe {
                    // The `next` field of the tail field must be zero when
                    // releasing the lock (queue invariant 2).
                    prev.as_ref().next.set(None);
                    // There are no set `tail` links before the node pointed to by
                    // `state`, so the first non-null tail field will be current
                    // (queue invariant 4).
                    to_node(state).as_ref().tail.set(Some(prev));
                }

                // Release the lock. Doing this by subtraction is more efficient
                // on modern processors since it is a single instruction instead
                // of an update loop, which will fail if new threads are added
                // to the queue.
                self.state.fetch_byte_sub(LOCKED, Release);
            }

            // The tail was split off and the lock released. Mark the node as
            // completed.
            unsafe {
                Node::complete(tail);
            }
        }
    }
}
