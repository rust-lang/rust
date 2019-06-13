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

use crate::fmt;
use crate::marker;
use crate::ptr;
use crate::sync::atomic::{AtomicUsize, AtomicBool, Ordering};
use crate::thread::{self, Thread};

/// A synchronization primitive which can be used to run a one-time global
/// initialization. Useful for one-time initialization for FFI or related
/// functionality. This type can only be constructed with the [`ONCE_INIT`]
/// value or the equivalent [`Once::new`] constructor.
///
/// [`ONCE_INIT`]: constant.ONCE_INIT.html
/// [`Once::new`]: struct.Once.html#method.new
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
    // This `state` word is actually an encoded version of just a pointer to a
    // `Waiter`, so we add the `PhantomData` appropriately.
    state: AtomicUsize,
    _marker: marker::PhantomData<*mut Waiter>,
}

// The `PhantomData` of a raw pointer removes these two auto traits, but we
// enforce both below in the implementation so this should be safe to add.
#[stable(feature = "rust1", since = "1.0.0")]
unsafe impl Sync for Once {}
#[stable(feature = "rust1", since = "1.0.0")]
unsafe impl Send for Once {}

/// State yielded to [`call_once_force`]’s closure parameter. The state can be
/// used to query the poison status of the [`Once`].
///
/// [`call_once_force`]: struct.Once.html#method.call_once_force
/// [`Once`]: struct.Once.html
#[unstable(feature = "once_poison", issue = "33577")]
#[derive(Debug)]
pub struct OnceState {
    poisoned: bool,
}

/// Initialization value for static [`Once`] values.
///
/// [`Once`]: struct.Once.html
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
    suggestion = "Once::new()",
)]
pub const ONCE_INIT: Once = Once::new();

// Four states that a Once can be in, encoded into the lower bits of `state` in
// the Once structure.
const INCOMPLETE: usize = 0x0;
const POISONED: usize = 0x1;
const RUNNING: usize = 0x2;
const COMPLETE: usize = 0x3;

// Mask to learn about the state. All other bits are the queue of waiters if
// this is in the RUNNING state.
const STATE_MASK: usize = 0x3;

// Representation of a node in the linked list of waiters in the RUNNING state.
struct Waiter {
    thread: Option<Thread>,
    signaled: AtomicBool,
    next: *mut Waiter,
}

// Helper struct used to clean up after a closure call with a `Drop`
// implementation to also run on panic.
struct Finish<'a> {
    panicked: bool,
    me: &'a Once,
}

impl Once {
    /// Creates a new `Once` value.
    #[stable(feature = "once_new", since = "1.2.0")]
    pub const fn new() -> Once {
        Once {
            state: AtomicUsize::new(INCOMPLETE),
            _marker: marker::PhantomData,
        }
    }

    /// Performs an initialization routine once and only once. The given closure
    /// will be executed if this is the first time `call_once` has been called,
    /// and otherwise the routine will *not* be invoked.
    ///
    /// This method will block the calling thread if another initialization
    /// routine is currently running.
    ///
    /// When this function returns, it is guaranteed that some initialization
    /// has run and completed (it may not be the closure specified). It is also
    /// guaranteed that any memory writes performed by the executed closure can
    /// be reliably observed by other threads at this point (there is a
    /// happens-before relation between the closure and code executing after the
    /// return).
    ///
    /// If the given closure recursively invokes `call_once` on the same `Once`
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
    /// it will *poison* this `Once` instance, causing all future invocations of
    /// `call_once` to also panic.
    ///
    /// This is similar to [poisoning with mutexes][poison].
    ///
    /// [poison]: struct.Mutex.html#poisoning
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn call_once<F>(&self, f: F) where F: FnOnce() {
        // Fast path check
        if self.is_completed() {
            return;
        }

        let mut f = Some(f);
        self.call_inner(false, &mut |_| f.take().unwrap()());
    }

    /// Performs the same function as [`call_once`] except ignores poisoning.
    ///
    /// Unlike [`call_once`], if this `Once` has been poisoned (i.e., a previous
    /// call to `call_once` or `call_once_force` caused a panic), calling
    /// `call_once_force` will still invoke the closure `f` and will _not_
    /// result in an immediate panic. If `f` panics, the `Once` will remain
    /// in a poison state. If `f` does _not_ panic, the `Once` will no
    /// longer be in a poison state and all future calls to `call_once` or
    /// `call_one_force` will be no-ops.
    ///
    /// The closure `f` is yielded a [`OnceState`] structure which can be used
    /// to query the poison status of the `Once`.
    ///
    /// [`call_once`]: struct.Once.html#method.call_once
    /// [`OnceState`]: struct.OnceState.html
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(once_poison)]
    ///
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
    ///     assert!(state.poisoned());
    /// });
    ///
    /// // once any success happens, we stop propagating the poison
    /// INIT.call_once(|| {});
    /// ```
    #[unstable(feature = "once_poison", issue = "33577")]
    pub fn call_once_force<F>(&self, f: F) where F: FnOnce(&OnceState) {
        // Fast path check
        if self.is_completed() {
            return;
        }

        let mut f = Some(f);
        self.call_inner(true, &mut |p| {
            f.take().unwrap()(&OnceState { poisoned: p })
        });
    }

    /// Returns `true` if some `call_once` call has completed
    /// successfully. Specifically, `is_completed` will return false in
    /// the following situations:
    ///   * `call_once` was not called at all,
    ///   * `call_once` was called, but has not yet completed,
    ///   * the `Once` instance is poisoned
    ///
    /// It is also possible that immediately after `is_completed`
    /// returns false, some other thread finishes executing
    /// `call_once`.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(once_is_completed)]
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
    /// #![feature(once_is_completed)]
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
    #[unstable(feature = "once_is_completed", issue = "54890")]
    #[inline]
    pub fn is_completed(&self) -> bool {
        // An `Acquire` load is enough because that makes all the initialization
        // operations visible to us, and, this being a fast path, weaker
        // ordering helps with performance. This `Acquire` synchronizes with
        // `SeqCst` operations on the slow path.
        self.state.load(Ordering::Acquire) == COMPLETE
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
    fn call_inner(&self,
                  ignore_poisoning: bool,
                  init: &mut dyn FnMut(bool)) {

        // This cold path uses SeqCst consistently because the
        // performance difference really does not matter there, and
        // SeqCst minimizes the chances of something going wrong.
        let mut state = self.state.load(Ordering::SeqCst);

        'outer: loop {
            match state {
                // If we're complete, then there's nothing to do, we just
                // jettison out as we shouldn't run the closure.
                COMPLETE => return,

                // If we're poisoned and we're not in a mode to ignore
                // poisoning, then we panic here to propagate the poison.
                POISONED if !ignore_poisoning => {
                    panic!("Once instance has previously been poisoned");
                }

                // Otherwise if we see a poisoned or otherwise incomplete state
                // we will attempt to move ourselves into the RUNNING state. If
                // we succeed, then the queue of waiters starts at null (all 0
                // bits).
                POISONED |
                INCOMPLETE => {
                    let old = self.state.compare_and_swap(state, RUNNING,
                                                          Ordering::SeqCst);
                    if old != state {
                        state = old;
                        continue
                    }

                    // Run the initialization routine, letting it know if we're
                    // poisoned or not. The `Finish` struct is then dropped, and
                    // the `Drop` implementation here is responsible for waking
                    // up other waiters both in the normal return and panicking
                    // case.
                    let mut complete = Finish {
                        panicked: true,
                        me: self,
                    };
                    init(state == POISONED);
                    complete.panicked = false;
                    return
                }

                // All other values we find should correspond to the RUNNING
                // state with an encoded waiter list in the more significant
                // bits. We attempt to enqueue ourselves by moving us to the
                // head of the list and bail out if we ever see a state that's
                // not RUNNING.
                _ => {
                    assert!(state & STATE_MASK == RUNNING);
                    let mut node = Waiter {
                        thread: Some(thread::current()),
                        signaled: AtomicBool::new(false),
                        next: ptr::null_mut(),
                    };
                    let me = &mut node as *mut Waiter as usize;
                    assert!(me & STATE_MASK == 0);

                    while state & STATE_MASK == RUNNING {
                        node.next = (state & !STATE_MASK) as *mut Waiter;
                        let old = self.state.compare_and_swap(state,
                                                              me | RUNNING,
                                                              Ordering::SeqCst);
                        if old != state {
                            state = old;
                            continue
                        }

                        // Once we've enqueued ourselves, wait in a loop.
                        // Afterwards reload the state and continue with what we
                        // were doing from before.
                        while !node.signaled.load(Ordering::SeqCst) {
                            thread::park();
                        }
                        state = self.state.load(Ordering::SeqCst);
                        continue 'outer
                    }
                }
            }
        }
    }
}

#[stable(feature = "std_debug", since = "1.16.0")]
impl fmt::Debug for Once {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.pad("Once { .. }")
    }
}

impl Drop for Finish<'_> {
    fn drop(&mut self) {
        // Swap out our state with however we finished. We should only ever see
        // an old state which was RUNNING.
        let queue = if self.panicked {
            self.me.state.swap(POISONED, Ordering::SeqCst)
        } else {
            self.me.state.swap(COMPLETE, Ordering::SeqCst)
        };
        assert_eq!(queue & STATE_MASK, RUNNING);

        // Decode the RUNNING to a list of waiters, then walk that entire list
        // and wake them up. Note that it is crucial that after we store `true`
        // in the node it can be free'd! As a result we load the `thread` to
        // signal ahead of time and then unpark it after the store.
        unsafe {
            let mut queue = (queue & !STATE_MASK) as *mut Waiter;
            while !queue.is_null() {
                let next = (*queue).next;
                let thread = (*queue).thread.take().unwrap();
                (*queue).signaled.store(true, Ordering::SeqCst);
                thread.unpark();
                queue = next;
            }
        }
    }
}

impl OnceState {
    /// Returns `true` if the associated [`Once`] was poisoned prior to the
    /// invocation of the closure passed to [`call_once_force`].
    ///
    /// [`call_once_force`]: struct.Once.html#method.call_once_force
    /// [`Once`]: struct.Once.html
    ///
    /// # Examples
    ///
    /// A poisoned `Once`:
    ///
    /// ```
    /// #![feature(once_poison)]
    ///
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
    ///     assert!(state.poisoned());
    /// });
    /// ```
    ///
    /// An unpoisoned `Once`:
    ///
    /// ```
    /// #![feature(once_poison)]
    ///
    /// use std::sync::Once;
    ///
    /// static INIT: Once = Once::new();
    ///
    /// INIT.call_once_force(|state| {
    ///     assert!(!state.poisoned());
    /// });
    #[unstable(feature = "once_poison", issue = "33577")]
    pub fn poisoned(&self) -> bool {
        self.poisoned
    }
}

#[cfg(all(test, not(target_os = "emscripten")))]
mod tests {
    use crate::panic;
    use crate::sync::mpsc::channel;
    use crate::thread;
    use super::Once;

    #[test]
    fn smoke_once() {
        static O: Once = Once::new();
        let mut a = 0;
        O.call_once(|| a += 1);
        assert_eq!(a, 1);
        O.call_once(|| a += 1);
        assert_eq!(a, 1);
    }

    #[test]
    fn stampede_once() {
        static O: Once = Once::new();
        static mut RUN: bool = false;

        let (tx, rx) = channel();
        for _ in 0..10 {
            let tx = tx.clone();
            thread::spawn(move|| {
                for _ in 0..4 { thread::yield_now() }
                unsafe {
                    O.call_once(|| {
                        assert!(!RUN);
                        RUN = true;
                    });
                    assert!(RUN);
                }
                tx.send(()).unwrap();
            });
        }

        unsafe {
            O.call_once(|| {
                assert!(!RUN);
                RUN = true;
            });
            assert!(RUN);
        }

        for _ in 0..10 {
            rx.recv().unwrap();
        }
    }

    #[test]
    fn poison_bad() {
        static O: Once = Once::new();

        // poison the once
        let t = panic::catch_unwind(|| {
            O.call_once(|| panic!());
        });
        assert!(t.is_err());

        // poisoning propagates
        let t = panic::catch_unwind(|| {
            O.call_once(|| {});
        });
        assert!(t.is_err());

        // we can subvert poisoning, however
        let mut called = false;
        O.call_once_force(|p| {
            called = true;
            assert!(p.poisoned())
        });
        assert!(called);

        // once any success happens, we stop propagating the poison
        O.call_once(|| {});
    }

    #[test]
    fn wait_for_force_to_finish() {
        static O: Once = Once::new();

        // poison the once
        let t = panic::catch_unwind(|| {
            O.call_once(|| panic!());
        });
        assert!(t.is_err());

        // make sure someone's waiting inside the once via a force
        let (tx1, rx1) = channel();
        let (tx2, rx2) = channel();
        let t1 = thread::spawn(move || {
            O.call_once_force(|p| {
                assert!(p.poisoned());
                tx1.send(()).unwrap();
                rx2.recv().unwrap();
            });
        });

        rx1.recv().unwrap();

        // put another waiter on the once
        let t2 = thread::spawn(|| {
            let mut called = false;
            O.call_once(|| {
                called = true;
            });
            assert!(!called);
        });

        tx2.send(()).unwrap();

        assert!(t1.join().is_ok());
        assert!(t2.join().is_ok());

    }
}
