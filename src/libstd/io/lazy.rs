use crate::cell::UnsafeCell;
use crate::sync::atomic::AtomicUsize;
use crate::sync::atomic::Ordering;
use crate::sys_common;
use crate::sys_common::mutex::Mutex;

/// Helper for lazy initialization of a static, with a destructor that runs when the main (Rust)
/// thread exits.
///
/// Currently used only inside the standard library, by the stdio types.
///
/// # Safety
/// - `UnsafeCell`: We only create a mutable reference during initialization and during the shutdown
///   phase. At both times there can't exist any other references.
/// - Destruction. The `Drop` implementation of `T` should not access references to anything except
///   itself, they are not guaranteed to exist. It should also not rely on other machinery of the
///   standard library to be available.
/// - Initialization. The `init` function for `get` should not call `get` itself,  to prevent
///   infinite recursion and acquiring the guard mutex reentrantly.
/// - We use the `Mutex` from `sys::common` because it has a `const` constructor. It currently has
///   UB when acquired reentrantly without calling `init`.
pub struct Lazy<T> {
    guard: Mutex, // Only used to protect initialization.
    status: AtomicUsize,
    data: UnsafeCell<Option<T>>,
}

unsafe impl<T> Sync for Lazy<T> {}

const UNINITIALIZED: usize = 0;
const SHUTDOWN: usize = 1;
const AVAILABLE: usize = 2;

impl<T> Lazy<T> {
    pub const fn new() -> Lazy<T> {
        Lazy {
            guard: Mutex::new(),
            status: AtomicUsize::new(UNINITIALIZED),
            data: UnsafeCell::new(None),
        }
    }
}

impl<T: Send + Sync + 'static> Lazy<T> {
    pub unsafe fn get(&'static self, init: fn() -> T) -> Option<&T> {
        match self.status.load(Ordering::Acquire) {
            UNINITIALIZED => {
                let _guard = self.guard.lock();
                // Double-check to make sure this `Lazy` didn't get initialized by another
                // thread in the small window before we acquired the mutex.
                if self.status.load(Ordering::Relaxed) != UNINITIALIZED {
                    return self.get(init);
                }

                // Register an `at_exit` handler that drops `data` when the main thread exits.
                let registered = sys_common::at_exit(move || {
                    *self.data.get() = None; // `T` gets dropped here
                    self.status.store(SHUTDOWN, Ordering::Release);
                });
                if registered.is_err() {
                    // Registering the handler will only fail if we are already in the shutdown
                    // phase. In that case don't attempt to initialize.
                    self.status.store(SHUTDOWN, Ordering::Release);
                    return None;
                }

                // Run the initializer of `T`.
                *self.data.get() = Some(init());
                self.status.store(AVAILABLE, Ordering::Release);

                (*self.data.get()).as_ref()
            },
            SHUTDOWN => None,
            _ => (*self.data.get()).as_ref(),
        }
    }
}
