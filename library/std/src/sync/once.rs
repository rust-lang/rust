//! A "once initialization" primitive
//!
//! This primitive is meant to be used to run one-time initialization. An
//! example use case would be for initializing an FFI library.

// A "once" is a relatively simple primitive, and it's also typically provided
// by the OS as well (see `pthread_once` or `InitOnceExecuteOnce`). The OS
// primitives, however, tend to have surprising restrictions, such as the Unix
// one doesn't allow an argument to be passed to the function.
//
// As a result, we end up implementing it ourselves in the standard library.
// This also gives us the opportunity to optimize the implementation a bit which
// should help the fast path on call sites. Consequently, let's explain how this
// primitive works now!
//
// So to recap, the guarantees of a Once are that it will call the
// initialization closure at most once, and it will never return until the one
// that's running has finished running. This means that we need some form of
// blocking here while the custom callback is running at the very least.
// Additionally, we add on the restriction of **poisoning**. Whenever an
// initialization closure panics, the Once enters a "poisoned" state which means
// that all future calls will immediately panic as well.
//
// So to implement this, one might first reach for a `Mutex`, but those cannot
// be put into a `static`. It also gets a lot harder with poisoning to figure
// out when the mutex needs to be deallocated because it's not after the closure
// finishes, but after the first successful closure finishes.
//
// All in all, this is instead implemented with atomics and lock-free
// operations! Whee! Each `Once` has one word of atomic state, and this state is
// CAS'd on to determine what to do. There are four possible state of a `Once`:
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
// Atomic orderings:
// When running `Once` we deal with multiple atomics:
// `Once.state_and_queue` and an unknown number of `Waiter.signaled`.
// * `state_and_queue` is used (1) as a state flag, (2) for synchronizing the
//   result of the `Once`, and (3) for synchronizing `Waiter` nodes.
//     - At the end of the `call_inner` function we have to make sure the result
//       of the `Once` is acquired. So every load which can be the only one to
//       load COMPLETED must have at least Acquire ordering, which means all
//       three of them.
//     - `WaiterQueue::Drop` is the only place that may store COMPLETED, and
//       must do so with Release ordering to make the result available.
//     - `wait` inserts `Waiter` nodes as a pointer in `state_and_queue`, and
//       needs to make the nodes available with Release ordering. The load in
//       its `compare_exchange` can be Relaxed because it only has to compare
//       the atomic, not to read other data.
//     - `WaiterQueue::Drop` must see the `Waiter` nodes, so it must load
//       `state_and_queue` with Acquire ordering.
//     - There is just one store where `state_and_queue` is used only as a
//       state flag, without having to synchronize data: switching the state
//       from INCOMPLETE to RUNNING in `call_inner`. This store can be Relaxed,
//       but the read has to be Acquire because of the requirements mentioned
//       above.
// * `Waiter.signaled` is both used as a flag, and to protect a field with
//   interior mutability in `Waiter`. `Waiter.thread` is changed in
//   `WaiterQueue::Drop` which then sets `signaled` with Release ordering.
//   After `wait` loads `signaled` with Acquire and sees it is true, it needs to
//   see the changes to drop the `Waiter` struct correctly.
// * There is one place where the two atomics `Once.state_and_queue` and
//   `Waiter.signaled` come together, and might be reordered by the compiler or
//   processor. Because both use Acquire ordering such a reordering is not
//   allowed, so no need for SeqCst.

#[cfg(all(test, not(target_os = "emscripten")))]
mod tests;

use crate::cell::Cell;
use crate::fmt;
use crate::marker;
use crate::panic::{RefUnwindSafe, UnwindSafe};
use crate::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use crate::thread::{self, Thread};

/// A synchronization primitive which can be used to run a one-time global
/// initialization. Useful for one-time initialization for FFI or related
/// functionality. This type can only be constructed with [`Once::new()`].
///
/// # Examples
///
/// ```
/// use std::sync::Once;
///
/// static START: Once = Once::new();
///
/// START.call_once(|| {
///     // run initialization here
/// });
/// ```
#[stable(feature = "rust1", since = "1.0.0")]
pub struct Once {
    // `state_and_queue` is actually a pointer to a `Waiter` with extra state
    // bits, so we add the `PhantomData` appropriately.
    state_and_queue: AtomicUsize,
    _marker: marker::PhantomData<*const Waiter>,
}

// The `PhantomData` of a raw pointer removes these two auto traits, but we
// enforce both below in the implementation so this should be safe to add.
#[stable(feature = "rust1", since = "1.0.0")]
unsafe impl Sync for Once {}
#[stable(feature = "rust1", since = "1.0.0")]
unsafe impl Send for Once {}

#[stable(feature = "sync_once_unwind_safe", since = "1.59.0")]
impl UnwindSafe for Once {}

#[stable(feature = "sync_once_unwind_safe", since = "1.59.0")]
impl RefUnwindSafe for Once {}

/// State yielded to [`Once::call_once_force()`]’s closure parameter. The state
/// can be used to query the poison status of the [`Once`].
#[stable(feature = "once_poison", since = "1.51.0")]
#[derive(Debug)]
pub struct OnceState {
    poisoned: bool,
    set_state_on_drop_to: Cell<usize>,
}

/// Initialization value for static [`Once`] values.
///
/// # Examples
///
/// ```
/// use std::sync::{Once, ONCE_INIT};
///
/// static START: Once = ONCE_INIT;
/// ```
#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_deprecated(
    since = "1.38.0",
    reason = "the `new` function is now preferred",
    suggestion = "Once::new()"
)]
pub const ONCE_INIT: Once = Once::new();

// Four states that a Once can be in, encoded into the lower bits of
// `state_and_queue` in the Once structure.
const INCOMPLETE: usize = 0x0;
const POISONED: usize = 0x1;
const RUNNING: usize = 0x2;
const COMPLETE: usize = 0x3;

// Mask to learn about the state. All other bits are the queue of waiters if
// this is in the RUNNING state.
const STATE_MASK: usize = 0x3;

// Representation of a node in the linked list of waiters, used while in the
// RUNNING state.
// Note: `Waiter` can't hold a mutable pointer to the next thread, because then
// `wait` would both hand out a mutable reference to its `Waiter` node, and keep
// a shared reference to check `signaled`. Instead we hold shared references and
// use interior mutability.
#[repr(align(4))] // Ensure the two lower bits are free to use as state bits.
struct Waiter {
    thread: Cell<Option<Thread>>,
    signaled: AtomicBool,
    next: *const Waiter,
}

// Head of a linked list of waiters.
// Every node is a struct on the stack of a waiting thread.
// Will wake up the waiters when it gets dropped, i.e. also on panic.
struct WaiterQueue<'a> {
    state_and_queue: &'a AtomicUsize,
    set_state_on_drop_to: usize,
}

impl Once {
    /// Creates a new `Once` value.
    #[inline]
    #[stable(feature = "once_new", since = "1.2.0")]
    #[rustc_const_stable(feature = "const_once_new", since = "1.32.0")]
    #[must_use]
    pub const fn new() -> Once {
        Once { state_and_queue: AtomicUsize::new(INCOMPLETE), _marker: marker::PhantomData }
    }

    /// Performs an initialization routine once and only once. The given closure
    /// will be executed if this is the first time `call_once` has been called,
    /// and otherwise the routine will *not* be invoked.
    ///
    /// This method will block the calling thread if another initialization
    /// routine is currently running.
    ///
    /// When this function returns, it is guaranteed that some initialization
    /// has run and completed (it might not be the closure specified). It is also
    /// guaranteed that any memory writes performed by the executed closure can
    /// be reliably observed by other threads at this point (there is a
    /// happens-before relation between the closure and code executing after the
    /// return).
    ///
    /// If the given closure recursively invokes `call_once` on the same [`Once`]
    /// instance the exact behavior is not specified, allowed outcomes are
    /// a panic or a deadlock.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::Once;
    ///
    /// static mut VAL: usize = 0;
    /// static INIT: Once = Once::new();
    ///
    /// // Accessing a `static mut` is unsafe much of the time, but if we do so
    /// // in a synchronized fashion (e.g., write once or read all) then we're
    /// // good to go!
    /// //
    /// // This function will only call `expensive_computation` once, and will
    /// // otherwise always return the value returned from the first invocation.
    /// fn get_cached_val() -> usize {
    ///     unsafe {
    ///         INIT.call_once(|| {
    ///             VAL = expensive_computation();
    ///         });
    ///         VAL
    ///     }
    /// }
    ///
    /// fn expensive_computation() -> usize {
    ///     // ...
    /// # 2
    /// }
    /// ```
    ///
    /// # Panics
    ///
    /// The closure `f` will only be executed once if this is called
    /// concurrently amongst many threads. If that closure panics, however, then
    /// it will *poison* this [`Once`] instance, causing all future invocations of
    /// `call_once` to also panic.
    ///
    /// This is similar to [poisoning with mutexes][poison].
    ///
    /// [poison]: struct.Mutex.html#poisoning
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn call_once<F>(&self, f: F)
    where
        F: FnOnce(),
    {
        // Fast path check
        if self.is_completed() {
            return;
        }

        let mut f = Some(f);
        self.call_inner(false, &mut |_| f.take().unwrap()());
    }

    /// Performs the same function as [`call_once()`] except ignores poisoning.
    ///
    /// Unlike [`call_once()`], if this [`Once`] has been poisoned (i.e., a previous
    /// call to [`call_once()`] or [`call_once_force()`] caused a panic), calling
    /// [`call_once_force()`] will still invoke the closure `f` and will _not_
    /// result in an immediate panic. If `f` panics, the [`Once`] will remain
    /// in a poison state. If `f` does _not_ panic, the [`Once`] will no
    /// longer be in a poison state and all future calls to [`call_once()`] or
    /// [`call_once_force()`] will be no-ops.
    ///
    /// The closure `f` is yielded a [`OnceState`] structure which can be used
    /// to query the poison status of the [`Once`].
    ///
    /// [`call_once()`]: Once::call_once
    /// [`call_once_force()`]: Once::call_once_force
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::Once;
    /// use std::thread;
    ///
    /// static INIT: Once = Once::new();
    ///
    /// // poison the once
    /// let handle = thread::spawn(|| {
    ///     INIT.call_once(|| panic!());
    /// });
    /// assert!(handle.join().is_err());
    ///
    /// // poisoning propagates
    /// let handle = thread::spawn(|| {
    ///     INIT.call_once(|| {});
    /// });
    /// assert!(handle.join().is_err());
    ///
    /// // call_once_force will still run and reset the poisoned state
    /// INIT.call_once_force(|state| {
    ///     assert!(state.is_poisoned());
    /// });
    ///
    /// // once any success happens, we stop propagating the poison
    /// INIT.call_once(|| {});
    /// ```
    #[stable(feature = "once_poison", since = "1.51.0")]
    pub fn call_once_force<F>(&self, f: F)
    where
        F: FnOnce(&OnceState),
    {
        // Fast path check
        if self.is_completed() {
            return;
        }

        let mut f = Some(f);
        self.call_inner(true, &mut |p| f.take().unwrap()(p));
    }

    /// Returns `true` if some [`call_once()`] call has completed
    /// successfully. Specifically, `is_completed` will return false in
    /// the following situations:
    ///   * [`call_once()`] was not called at all,
    ///   * [`call_once()`] was called, but has not yet completed,
    ///   * the [`Once`] instance is poisoned
    ///
    /// This function returning `false` does not mean that [`Once`] has not been
    /// executed. For example, it may have been executed in the time between
    /// when `is_completed` starts executing and when it returns, in which case
    /// the `false` return value would be stale (but still permissible).
    ///
    /// [`call_once()`]: Once::call_once
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::Once;
    ///
    /// static INIT: Once = Once::new();
    ///
    /// assert_eq!(INIT.is_completed(), false);
    /// INIT.call_once(|| {
    ///     assert_eq!(INIT.is_completed(), false);
    /// });
    /// assert_eq!(INIT.is_completed(), true);
    /// ```
    ///
    /// ```
    /// use std::sync::Once;
    /// use std::thread;
    ///
    /// static INIT: Once = Once::new();
    ///
    /// assert_eq!(INIT.is_completed(), false);
    /// let handle = thread::spawn(|| {
    ///     INIT.call_once(|| panic!());
    /// });
    /// assert!(handle.join().is_err());
    /// assert_eq!(INIT.is_completed(), false);
    /// ```
    #[stable(feature = "once_is_completed", since = "1.43.0")]
    #[inline]
    pub fn is_completed(&self) -> bool {
        // An `Acquire` load is enough because that makes all the initialization
        // operations visible to us, and, this being a fast path, weaker
        // ordering helps with performance. This `Acquire` synchronizes with
        // `Release` operations on the slow path.
        self.state_and_queue.load(Ordering::Acquire) == COMPLETE
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
    fn call_inner(&self, ignore_poisoning: bool, init: &mut dyn FnMut(&OnceState)) {
        let mut state_and_queue = self.state_and_queue.load(Ordering::Acquire);
        loop {
            match state_and_queue {
                COMPLETE => break,
                POISONED if !ignore_poisoning => {
                    // Panic to propagate the poison.
                    panic!("Once instance has previously been poisoned");
                }
                POISONED | INCOMPLETE => {
                    // Try to register this thread as the one RUNNING.
                    let exchange_result = self.state_and_queue.compare_exchange(
                        state_and_queue,
                        RUNNING,
                        Ordering::Acquire,
                        Ordering::Acquire,
                    );
                    if let Err(old) = exchange_result {
                        state_and_queue = old;
                        continue;
                    }
                    // `waiter_queue` will manage other waiting threads, and
                    // wake them up on drop.
                    let mut waiter_queue = WaiterQueue {
                        state_and_queue: &self.state_and_queue,
                        set_state_on_drop_to: POISONED,
                    };
                    // Run the initialization function, letting it know if we're
                    // poisoned or not.
                    let init_state = OnceState {
                        poisoned: state_and_queue == POISONED,
                        set_state_on_drop_to: Cell::new(COMPLETE),
                    };
                    init(&init_state);
                    waiter_queue.set_state_on_drop_to = init_state.set_state_on_drop_to.get();
                    break;
                }
                _ => {
                    // All other values must be RUNNING with possibly a
                    // pointer to the waiter queue in the more significant bits.
                    assert!(state_and_queue & STATE_MASK == RUNNING);
                    wait(&self.state_and_queue, state_and_queue);
                    state_and_queue = self.state_and_queue.load(Ordering::Acquire);
                }
            }
        }
    }
}

fn wait(state_and_queue: &AtomicUsize, mut current_state: usize) {
    // Note: the following code was carefully written to avoid creating a
    // mutable reference to `node` that gets aliased.
    loop {
        // Don't queue this thread if the status is no longer running,
        // otherwise we will not be woken up.
        if current_state & STATE_MASK != RUNNING {
            return;
        }

        // Create the node for our current thread.
        let node = Waiter {
            thread: Cell::new(Some(thread::current())),
            signaled: AtomicBool::new(false),
            next: (current_state & !STATE_MASK) as *const Waiter,
        };
        let me = &node as *const Waiter as usize;

        // Try to slide in the node at the head of the linked list, making sure
        // that another thread didn't just replace the head of the linked list.
        let exchange_result = state_and_queue.compare_exchange(
            current_state,
            me | RUNNING,
            Ordering::Release,
            Ordering::Relaxed,
        );
        if let Err(old) = exchange_result {
            current_state = old;
            continue;
        }

        // We have enqueued ourselves, now lets wait.
        // It is important not to return before being signaled, otherwise we
        // would drop our `Waiter` node and leave a hole in the linked list
        // (and a dangling reference). Guard against spurious wakeups by
        // reparking ourselves until we are signaled.
        while !node.signaled.load(Ordering::Acquire) {
            // If the managing thread happens to signal and unpark us before we
            // can park ourselves, the result could be this thread never gets
            // unparked. Luckily `park` comes with the guarantee that if it got
            // an `unpark` just before on an unparked thread it does not park.
            thread::park();
        }
        break;
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
        let state_and_queue =
            self.state_and_queue.swap(self.set_state_on_drop_to, Ordering::AcqRel);

        // We should only ever see an old state which was RUNNING.
        assert_eq!(state_and_queue & STATE_MASK, RUNNING);

        // Walk the entire linked list of waiters and wake them up (in lifo
        // order, last to register is first to wake up).
        unsafe {
            // Right after setting `node.signaled = true` the other thread may
            // free `node` if there happens to be has a spurious wakeup.
            // So we have to take out the `thread` field and copy the pointer to
            // `next` first.
            let mut queue = (state_and_queue & !STATE_MASK) as *const Waiter;
            while !queue.is_null() {
                let next = (*queue).next;
                let thread = (*queue).thread.take().unwrap();
                (*queue).signaled.store(true, Ordering::Release);
                // ^- FIXME (maybe): This is another case of issue #55005
                // `store()` has a potentially dangling ref to `signaled`.
                queue = next;
                thread.unpark();
            }
        }
    }
}

impl OnceState {
    /// Returns `true` if the associated [`Once`] was poisoned prior to the
    /// invocation of the closure passed to [`Once::call_once_force()`].
    ///
    /// # Examples
    ///
    /// A poisoned [`Once`]:
    ///
    /// ```
    /// use std::sync::Once;
    /// use std::thread;
    ///
    /// static INIT: Once = Once::new();
    ///
    /// // poison the once
    /// let handle = thread::spawn(|| {
    ///     INIT.call_once(|| panic!());
    /// });
    /// assert!(handle.join().is_err());
    ///
    /// INIT.call_once_force(|state| {
    ///     assert!(state.is_poisoned());
    /// });
    /// ```
    ///
    /// An unpoisoned [`Once`]:
    ///
    /// ```
    /// use std::sync::Once;
    ///
    /// static INIT: Once = Once::new();
    ///
    /// INIT.call_once_force(|state| {
    ///     assert!(!state.is_poisoned());
    /// });
    #[stable(feature = "once_poison", since = "1.51.0")]
    pub fn is_poisoned(&self) -> bool {
        self.poisoned
    }

    /// Poison the associated [`Once`] without explicitly panicking.
    // NOTE: This is currently only exposed for the `lazy` module
    pub(crate) fn poison(&self) {
        self.set_state_on_drop_to.set(POISONED);
    }
}
