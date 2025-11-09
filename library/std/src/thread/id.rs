use crate::num::NonZero;
use crate::sync::atomic::{Atomic, Ordering};

/// A unique identifier for a running thread.
///
/// A `ThreadId` is an opaque object that uniquely identifies each thread
/// created during the lifetime of a process. `ThreadId`s are guaranteed not to
/// be reused, even when a thread terminates. `ThreadId`s are under the control
/// of Rust's standard library and there may not be any relationship between
/// `ThreadId` and the underlying platform's notion of a thread identifier --
/// the two concepts cannot, therefore, be used interchangeably. A `ThreadId`
/// can be retrieved from the [`id`] method on a [`Thread`].
///
/// # Examples
///
/// ```
/// use std::thread;
///
/// let other_thread = thread::spawn(|| {
///     thread::current().id()
/// });
///
/// let other_thread_id = other_thread.join().unwrap();
/// assert!(thread::current().id() != other_thread_id);
/// ```
///
/// [`Thread`]: super::Thread
/// [`id`]: super::Thread::id
#[stable(feature = "thread_id", since = "1.19.0")]
#[derive(Eq, PartialEq, Clone, Copy, Hash, Debug)]
pub struct ThreadId(NonZero<u64>);

impl ThreadId {
    // Generate a new unique thread ID.
    pub(crate) fn new() -> ThreadId {
        #[cold]
        fn exhausted() -> ! {
            panic!("failed to generate unique thread ID: bitspace exhausted")
        }

        cfg_select! {
            target_has_atomic = "64" => {
                use crate::sync::atomic::AtomicU64;

                static COUNTER: Atomic<u64> = AtomicU64::new(0);

                let mut last = COUNTER.load(Ordering::Relaxed);
                loop {
                    let Some(id) = last.checked_add(1) else {
                        exhausted();
                    };

                    match COUNTER.compare_exchange_weak(last, id, Ordering::Relaxed, Ordering::Relaxed) {
                        Ok(_) => return ThreadId(NonZero::new(id).unwrap()),
                        Err(id) => last = id,
                    }
                }
            }
            _ => {
                use crate::cell::SyncUnsafeCell;
                use crate::hint::spin_loop;
                use crate::sync::atomic::AtomicBool;
                use crate::thread::yield_now;

                // If we don't have a 64-bit atomic we use a small spinlock. We don't use Mutex
                // here as we might be trying to get the current thread id in the global allocator,
                // and on some platforms Mutex requires allocation.
                static COUNTER_LOCKED: Atomic<bool> = AtomicBool::new(false);
                static COUNTER: SyncUnsafeCell<u64> = SyncUnsafeCell::new(0);

                // Acquire lock.
                let mut spin = 0;
                while COUNTER_LOCKED.compare_exchange_weak(false, true, Ordering::Acquire, Ordering::Relaxed).is_err() {
                    if spin <= 3 {
                        for _ in 0..(1 << spin) {
                            spin_loop();
                        }
                    } else {
                        yield_now();
                    }
                    spin += 1;
                }

                // SAFETY: we have an exclusive lock on the counter.
                unsafe {
                    if let Some(id) = (*COUNTER.get()).checked_add(1) {
                        *COUNTER.get() = id;
                        COUNTER_LOCKED.store(false, Ordering::Release);
                        ThreadId(NonZero::new(id).unwrap())
                    } else {
                        COUNTER_LOCKED.store(false, Ordering::Release);
                        exhausted()
                    }
                }
            }
        }
    }

    #[cfg(any(not(target_thread_local), target_has_atomic = "64"))]
    pub(super) fn from_u64(v: u64) -> Option<ThreadId> {
        NonZero::new(v).map(ThreadId)
    }

    /// This returns a numeric identifier for the thread identified by this
    /// `ThreadId`.
    ///
    /// As noted in the documentation for the type itself, it is essentially an
    /// opaque ID, but is guaranteed to be unique for each thread. The returned
    /// value is entirely opaque -- only equality testing is stable. Note that
    /// it is not guaranteed which values new threads will return, and this may
    /// change across Rust versions.
    #[must_use]
    #[unstable(feature = "thread_id_value", issue = "67939")]
    pub fn as_u64(&self) -> NonZero<u64> {
        self.0
    }
}
