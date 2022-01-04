use super::{current, park, Builder, JoinInner, Result, Thread};
use crate::any::Any;
use crate::fmt;
use crate::io;
use crate::marker::PhantomData;
use crate::panic::{catch_unwind, resume_unwind, AssertUnwindSafe};
use crate::sync::atomic::{AtomicUsize, Ordering};
use crate::sync::Mutex;

/// TODO: documentation
pub struct Scope<'env> {
    data: ScopeData,
    env: PhantomData<&'env ()>,
}

/// TODO: documentation
pub struct ScopedJoinHandle<'scope, T>(JoinInner<'scope, T>);

pub(super) struct ScopeData {
    n_running_threads: AtomicUsize,
    main_thread: Thread,
    pub(super) panic_payload: Mutex<Option<Box<dyn Any + Send>>>,
}

impl ScopeData {
    pub(super) fn increment_n_running_threads(&self) {
        // We check for 'overflow' with usize::MAX / 2, to make sure there's no
        // chance it overflows to 0, which would result in unsoundness.
        if self.n_running_threads.fetch_add(1, Ordering::Relaxed) == usize::MAX / 2 {
            // This can only reasonably happen by mem::forget()'ing many many ScopedJoinHandles.
            self.decrement_n_running_threads();
            panic!("too many running threads in thread scope");
        }
    }
    pub(super) fn decrement_n_running_threads(&self) {
        if self.n_running_threads.fetch_sub(1, Ordering::Release) == 1 {
            self.main_thread.unpark();
        }
    }
}

/// TODO: documentation
pub fn scope<'env, F, T>(f: F) -> T
where
    F: FnOnce(&Scope<'env>) -> T,
{
    let mut scope = Scope {
        data: ScopeData {
            n_running_threads: AtomicUsize::new(0),
            main_thread: current(),
            panic_payload: Mutex::new(None),
        },
        env: PhantomData,
    };

    // Run `f`, but catch panics so we can make sure to wait for all the threads to join.
    let result = catch_unwind(AssertUnwindSafe(|| f(&scope)));

    // Wait until all the threads are finished.
    while scope.data.n_running_threads.load(Ordering::Acquire) != 0 {
        park();
    }

    // Throw any panic from `f` or from any panicked thread, or the return value of `f` otherwise.
    match result {
        Err(e) => {
            // `f` itself panicked.
            resume_unwind(e);
        }
        Ok(result) => {
            if let Some(panic_payload) = scope.data.panic_payload.get_mut().unwrap().take() {
                // A thread panicked.
                resume_unwind(panic_payload);
            } else {
                // Nothing panicked.
                result
            }
        }
    }
}

impl<'env> Scope<'env> {
    /// TODO: documentation
    pub fn spawn<'scope, F, T>(&'scope self, f: F) -> ScopedJoinHandle<'scope, T>
    where
        F: FnOnce(&Scope<'env>) -> T + Send + 'env,
        T: Send + 'env,
    {
        Builder::new().spawn_scoped(self, f).expect("failed to spawn thread")
    }
}

impl Builder {
    fn spawn_scoped<'scope, 'env, F, T>(
        self,
        scope: &'scope Scope<'env>,
        f: F,
    ) -> io::Result<ScopedJoinHandle<'scope, T>>
    where
        F: FnOnce(&Scope<'env>) -> T + Send + 'env,
        T: Send + 'env,
    {
        Ok(ScopedJoinHandle(unsafe { self.spawn_unchecked_(|| f(scope), Some(&scope.data)) }?))
    }
}

impl<'scope, T> ScopedJoinHandle<'scope, T> {
    /// TODO
    pub fn join(self) -> Result<T> {
        self.0.join()
    }

    /// TODO
    pub fn thread(&self) -> &Thread {
        &self.0.thread
    }
}

impl<'env> fmt::Debug for Scope<'env> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Scope")
            .field("n_running_threads", &self.data.n_running_threads.load(Ordering::Relaxed))
            .field("panic_payload", &self.data.panic_payload)
            .finish_non_exhaustive()
    }
}

impl<'scope, T> fmt::Debug for ScopedJoinHandle<'scope, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ScopedJoinHandle").finish_non_exhaustive()
    }
}
