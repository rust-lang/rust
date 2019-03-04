use crate::cell::UnsafeCell;
use crate::sync::Arc;
use crate::sync::atomic::AtomicUsize;
use crate::sync::atomic::Ordering;
use crate::sys_common;
use crate::sys_common::mutex::Mutex;

/// Helper for lazy initialization of a static, with a destructor that attempts to run when the main
/// (Rust) thread exits.
///
/// Currently used only inside the standard library, by the stdio types.
///
/// If there are still child threads around when the main thread exits, they get terminated. But
/// there is a small window where they are not yet terminated and may hold a reference to the
/// the data. We therefore store the data in an `Arc<T>`, keep one of the `Arc`'s in the static, and
/// hand out clones. When the `Arc` in the static gets dropped by the `at_exit` handler, the
/// contents will only be dropped if there where no childs threads holding a reference.
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
    data: UnsafeCell<Option<Arc<T>>>,
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
    pub unsafe fn get(&'static self, init: fn() -> T) -> Option<Arc<T>> {
        if self.status.load(Ordering::Acquire) == UNINITIALIZED {
            let _guard = self.guard.lock();
            // Double-check to make sure this `Lazy` didn't get initialized by another
            // thread in the small window before we acquired the mutex.
            if self.status.load(Ordering::Relaxed) != UNINITIALIZED {
                return self.get(init);
            }

            // Register an `at_exit` handler.
            let registered = sys_common::at_exit(move || {
                *self.data.get() = None;
                // The reference to `Arc<T>` gets dropped above. If there are no other references
                // in child threads `T` will be dropped.
                self.status.store(SHUTDOWN, Ordering::Release);
            });
            if registered.is_err() {
                // Registering the handler will only fail if we are already in the shutdown
                // phase. In that case don't attempt to initialize.
                return None;
            }

            // Run the initializer of `T`.
            *self.data.get() = Some(Arc::new(init()));
            self.status.store(AVAILABLE, Ordering::Release);
        }
        (*self.data.get()).as_ref().cloned()
    }
}
