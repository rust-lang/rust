// Each `Once` has one word of atomic state, and this state is CAS'd on to
// determine what to do. There are four possible state of a `Once`:
//
// * Incomplete - no initialization has run yet, and no thread is currently
//                using the Once.
// * Poisoned - some thread has previously attempted to initialize the Once, but
//              it panicked, so the Once is now poisoned. There are no other
//              threads currently accessing this Once.
// * Running - some thread is currently attempting to run initialization. It may
//             succeed, so all future threads need to wait for it to finish.
//             Note that this state is accompanied with a payload, described
//             below.
// * Complete - initialization has completed and all future calls should finish
//              immediately.
//
// With 4 states we need 2 bits to encode this, and we use the remaining bits
// in the word we have allocated as a queue of threads waiting for the thread
// responsible for entering the RUNNING state. This queue is just a linked list
// of Waiter nodes which is monotonically increasing in size. Each node is
// allocated on the stack, and whenever the running closure finishes it will
// consume the entire queue and notify all waiters they should try again.
//
// You'll find a few more details in the implementation, but that's the gist of
// it!
//
// Futex orderings:
// When running `Once` we deal with multiple atomics:
// `Once.state_and_queue` and an unknown number of `Waiter.signaled`.
// * `state_and_queue` is used (1) as a state flag, (2) for synchronizing the
//   result of the `Once`, and (3) for synchronizing `Waiter` nodes.
//     - At the end of the `call` function we have to make sure the result
//       of the `Once` is acquired. So every load which can be the only one to
//       load COMPLETED must have at least acquire ordering, which means all
//       three of them.
//     - `WaiterQueue::drop` is the only place that may store COMPLETED, and
//       must do so with release ordering to make the result available.
//     - `wait` inserts `Waiter` nodes as a pointer in `state_and_queue`, and
//       needs to make the nodes available with release ordering. The load in
//       its `compare_exchange` can be relaxed because it only has to compare
//       the atomic, not to read other data.
//     - `WaiterQueue::drop` must see the `Waiter` nodes, so it must load
//       `state_and_queue` with acquire ordering.
//     - There is just one store where `state_and_queue` is used only as a
//       state flag, without having to synchronize data: switching the state
//       from INCOMPLETE to RUNNING in `call`. This store can be Relaxed,
//       but the read has to be Acquire because of the requirements mentioned
//       above.
// * `Waiter.signaled` is both used as a flag, and to protect a field with
//   interior mutability in `Waiter`. `Waiter.thread` is changed in
//   `WaiterQueue::drop` which then sets `signaled` with release ordering.
//   After `wait` loads `signaled` with acquire ordering and sees it is true,
//   it needs to see the changes to drop the `Waiter` struct correctly.
// * There is one place where the two atomics `Once.state_and_queue` and
//   `Waiter.signaled` come together, and might be reordered by the compiler or
//   processor. Because both use acquire ordering such a reordering is not
//   allowed, so no need for `SeqCst`.

use crate::cell::Cell;
use crate::sync::atomic::Ordering::{AcqRel, Acquire, Release};
use crate::sync::atomic::{Atomic, AtomicBool, AtomicPtr};
use crate::sync::poison::once::ExclusiveState;
use crate::thread::{self, Thread};
use crate::{fmt, ptr, sync as public};

type StateAndQueue = *mut ();

pub struct Once {
    state_and_queue: Atomic<*mut ()>,
}

pub struct OnceState {
    poisoned: bool,
    set_state_on_drop_to: Cell<StateAndQueue>,
}

// Four states that a Once can be in, encoded into the lower bits of
// `state_and_queue` in the Once structure. By choosing COMPLETE as the all-zero
// state the `is_completed` check can be a bit faster on some platforms.
const INCOMPLETE: usize = 0x3;
const POISONED: usize = 0x2;
const RUNNING: usize = 0x1;
const COMPLETE: usize = 0x0;

// Mask to learn about the state. All other bits are the queue of waiters if
// this is in the RUNNING state.
const STATE_MASK: usize = 0b11;
const QUEUE_MASK: usize = !STATE_MASK;

// Representation of a node in the linked list of waiters, used while in the
// RUNNING state.
// Note: `Waiter` can't hold a mutable pointer to the next thread, because then
// `wait` would both hand out a mutable reference to its `Waiter` node, and keep
// a shared reference to check `signaled`. Instead we hold shared references and
// use interior mutability.
#[repr(align(4))] // Ensure the two lower bits are free to use as state bits.
struct Waiter {
    thread: Thread,
    signaled: Atomic<bool>,
    next: Cell<*const Waiter>,
}

// Head of a linked list of waiters.
// Every node is a struct on the stack of a waiting thread.
// Will wake up the waiters when it gets dropped, i.e. also on panic.
struct WaiterQueue<'a> {
    state_and_queue: &'a Atomic<*mut ()>,
    set_state_on_drop_to: StateAndQueue,
}

fn to_queue(current: StateAndQueue) -> *const Waiter {
    current.mask(QUEUE_MASK).cast()
}

fn to_state(current: StateAndQueue) -> usize {
    current.addr() & STATE_MASK
}

impl Once {
    #[inline]
    pub const fn new() -> Once {
        Once { state_and_queue: AtomicPtr::new(ptr::without_provenance_mut(INCOMPLETE)) }
    }

    #[inline]
    pub fn is_completed(&self) -> bool {
        // An `Acquire` load is enough because that makes all the initialization
        // operations visible to us, and, this being a fast path, weaker
        // ordering helps with performance. This `Acquire` synchronizes with
        // `Release` operations on the slow path.
        self.state_and_queue.load(Acquire).addr() == COMPLETE
    }

    #[inline]
    pub(crate) fn state(&mut self) -> ExclusiveState {
        match self.state_and_queue.get_mut().addr() {
            INCOMPLETE => ExclusiveState::Incomplete,
            POISONED => ExclusiveState::Poisoned,
            COMPLETE => ExclusiveState::Complete,
            _ => unreachable!("invalid Once state"),
        }
    }

    #[inline]
    pub(crate) fn set_state(&mut self, new_state: ExclusiveState) {
        *self.state_and_queue.get_mut() = match new_state {
            ExclusiveState::Incomplete => ptr::without_provenance_mut(INCOMPLETE),
            ExclusiveState::Poisoned => ptr::without_provenance_mut(POISONED),
            ExclusiveState::Complete => ptr::without_provenance_mut(COMPLETE),
        };
    }

    #[cold]
    #[track_caller]
    pub fn wait(&self, ignore_poisoning: bool) {
        let mut current = self.state_and_queue.load(Acquire);
        loop {
            let state = to_state(current);
            match state {
                COMPLETE => return,
                POISONED if !ignore_poisoning => {
                    // Panic to propagate the poison.
                    panic!("Once instance has previously been poisoned");
                }
                _ => {
                    current = wait(&self.state_and_queue, current, !ignore_poisoning);
                }
            }
        }
    }

    // This is a non-generic function to reduce the monomorphization cost of
    // using `call_once` (this isn't exactly a trivial or small implementation).
    //
    // Additionally, this is tagged with `#[cold]` as it should indeed be cold
    // and it helps let LLVM know that calls to this function should be off the
    // fast path. Essentially, this should help generate more straight line code
    // in LLVM.
    //
    // Finally, this takes an `FnMut` instead of a `FnOnce` because there's
    // currently no way to take an `FnOnce` and call it via virtual dispatch
    // without some allocation overhead.
    #[cold]
    #[track_caller]
    pub fn call(&self, ignore_poisoning: bool, init: &mut dyn FnMut(&public::OnceState)) {
        let mut current = self.state_and_queue.load(Acquire);
        loop {
            let state = to_state(current);
            match state {
                COMPLETE => break,
                POISONED if !ignore_poisoning => {
                    // Panic to propagate the poison.
                    panic!("Once instance has previously been poisoned");
                }
                POISONED | INCOMPLETE => {
                    // Try to register this thread as the one RUNNING.
                    if let Err(new) = self.state_and_queue.compare_exchange_weak(
                        current,
                        current.mask(QUEUE_MASK).wrapping_byte_add(RUNNING),
                        Acquire,
                        Acquire,
                    ) {
                        current = new;
                        continue;
                    }

                    // `waiter_queue` will manage other waiting threads, and
                    // wake them up on drop.
                    let mut waiter_queue = WaiterQueue {
                        state_and_queue: &self.state_and_queue,
                        set_state_on_drop_to: ptr::without_provenance_mut(POISONED),
                    };
                    // Run the initialization function, letting it know if we're
                    // poisoned or not.
                    let init_state = public::OnceState {
                        inner: OnceState {
                            poisoned: state == POISONED,
                            set_state_on_drop_to: Cell::new(ptr::without_provenance_mut(COMPLETE)),
                        },
                    };
                    init(&init_state);
                    waiter_queue.set_state_on_drop_to = init_state.inner.set_state_on_drop_to.get();
                    return;
                }
                _ => {
                    // All other values must be RUNNING with possibly a
                    // pointer to the waiter queue in the more significant bits.
                    assert!(state == RUNNING);
                    current = wait(&self.state_and_queue, current, true);
                }
            }
        }
    }
}

fn wait(
    state_and_queue: &Atomic<*mut ()>,
    mut current: StateAndQueue,
    return_on_poisoned: bool,
) -> StateAndQueue {
    let node = &Waiter {
        thread: thread::current_or_unnamed(),
        signaled: AtomicBool::new(false),
        next: Cell::new(ptr::null()),
    };

    loop {
        let state = to_state(current);
        let queue = to_queue(current);

        // If initialization has finished, return.
        if state == COMPLETE || (return_on_poisoned && state == POISONED) {
            return current;
        }

        // Update the node for our current thread.
        node.next.set(queue);

        // Try to slide in the node at the head of the linked list, making sure
        // that another thread didn't just replace the head of the linked list.
        if let Err(new) = state_and_queue.compare_exchange_weak(
            current,
            ptr::from_ref(node).wrapping_byte_add(state) as StateAndQueue,
            Release,
            Acquire,
        ) {
            current = new;
            continue;
        }

        // We have enqueued ourselves, now lets wait.
        // It is important not to return before being signaled, otherwise we
        // would drop our `Waiter` node and leave a hole in the linked list
        // (and a dangling reference). Guard against spurious wakeups by
        // reparking ourselves until we are signaled.
        while !node.signaled.load(Acquire) {
            // If the managing thread happens to signal and unpark us before we
            // can park ourselves, the result could be this thread never gets
            // unparked. Luckily `park` comes with the guarantee that if it got
            // an `unpark` just before on an unparked thread it does not park. Crucially, we know
            // the `unpark` must have happened between the `compare_exchange_weak` above and here,
            // and there's no other `park` in that code that could steal our token.
            // SAFETY: we retrieved this handle on the current thread above.
            unsafe { node.thread.park() }
        }

        return state_and_queue.load(Acquire);
    }
}

#[stable(feature = "std_debug", since = "1.16.0")]
impl fmt::Debug for Once {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Once").finish_non_exhaustive()
    }
}

impl Drop for WaiterQueue<'_> {
    fn drop(&mut self) {
        // Swap out our state with however we finished.
        let current = self.state_and_queue.swap(self.set_state_on_drop_to, AcqRel);

        // We should only ever see an old state which was RUNNING.
        assert_eq!(current.addr() & STATE_MASK, RUNNING);

        // Walk the entire linked list of waiters and wake them up (in lifo
        // order, last to register is first to wake up).
        unsafe {
            // Right after setting `node.signaled = true` the other thread may
            // free `node` if there happens to be has a spurious wakeup.
            // So we have to take out the `thread` field and copy the pointer to
            // `next` first.
            let mut queue = to_queue(current);
            while !queue.is_null() {
                let next = (*queue).next.get();
                let thread = (*queue).thread.clone();
                (*queue).signaled.store(true, Release);
                thread.unpark();
                queue = next;
            }
        }
    }
}

impl OnceState {
    #[inline]
    pub fn is_poisoned(&self) -> bool {
        self.poisoned
    }

    #[inline]
    pub fn poison(&self) {
        self.set_state_on_drop_to.set(ptr::without_provenance_mut(POISONED));
    }
}
