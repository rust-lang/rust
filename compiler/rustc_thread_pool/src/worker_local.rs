use std::cell::{Cell, OnceCell};
use std::num::NonZero;
use std::ops::Deref;
use std::ptr;
use std::sync::{Arc, Mutex};

use crossbeam_utils::CachePadded;

/// A pointer to the `ComplementaryRegistryData` which uniquely identifies a complementary registry.
/// This identifier can be reused if the registry gets freed.
#[derive(Clone, Copy, PartialEq)]
struct ComplementaryRegistryId(*const ComplementaryRegistryData);

impl ComplementaryRegistryId {
    #[inline(always)]
    /// Verifies that the current thread is associated with the registry and returns its unique
    /// index within the registry. This panics if the current thread is not associated with this
    /// registry.
    ///
    /// Note that there's a race possible where the identifier in `THREAD_DATA` could be reused
    /// so this can succeed from a different registry.
    fn verify(self) -> usize {
        let (id, index) = THREAD_DATA.with(|data| (data.registry_id.get(), data.index.get()));

        if id == self { index } else { ComplementaryRegistryId::verification_error() }
    }

    #[cold]
    #[inline(never)]
    fn verification_error() -> ! {
        panic!("Unable to verify registry association")
    }
}

struct ComplementaryRegistryData {
    thread_limit: NonZero<usize>,
    threads: Mutex<usize>,
}

/// Represents a list of threads which can access worker locals.
#[derive(Clone)]
pub struct ComplementaryRegistry(Arc<ComplementaryRegistryData>);

thread_local! {
    /// The complementary registry associated with the thread.
    /// This allows the `WorkerLocal` type to clone the registry in its constructor.
    static REGISTRY: OnceCell<ComplementaryRegistry> = const { OnceCell::new() };
}

struct ThreadData {
    registry_id: Cell<ComplementaryRegistryId>,
    index: Cell<usize>,
}

thread_local! {
    /// A thread local which contains the identifier of `REGISTRY` but allows for faster access.
    /// It also holds the index of the current thread.
    static THREAD_DATA: ThreadData = const { ThreadData {
        registry_id: Cell::new(ComplementaryRegistryId(ptr::null())),
        index: Cell::new(0),
    }};
}

impl ComplementaryRegistry {
    /// Creates a registry which can hold up to `thread_limit` threads.
    pub fn new(thread_limit: NonZero<usize>) -> Self {
        ComplementaryRegistry(Arc::new(ComplementaryRegistryData {
            thread_limit,
            threads: Mutex::new(0),
        }))
    }

    /// Gets the registry associated with the current thread. Panics if there's no such registry.
    pub fn current() -> Self {
        REGISTRY.with(|registry| registry.get().cloned().expect("No associated registry"))
    }

    /// Registers the current thread with the registry so worker locals can be used on it.
    /// Panics if the thread limit is hit or if the thread already has an associated registry.
    pub fn register(&self) {
        let mut threads = self.0.threads.lock().unwrap();
        if *threads < self.0.thread_limit.get() {
            REGISTRY.with(|registry| {
                if registry.get().is_some() {
                    drop(threads);
                    panic!("Thread already has a registry");
                }
                registry.set(self.clone()).ok();
                THREAD_DATA.with(|data| {
                    data.registry_id.set(self.id());
                    data.index.set(*threads);
                });
                *threads += 1;
            });
        } else {
            drop(threads);
            panic!("Thread limit reached");
        }
    }

    /// Gets the identifier of this registry.
    fn id(&self) -> ComplementaryRegistryId {
        ComplementaryRegistryId(&*self.0)
    }
}

/// Holds worker local values for each possible thread in a complementary registry.
/// You can only access the worker local value through the `Deref` impl on the registry associated
/// with the thread it was created on. It will panic otherwise.
pub struct WorkerLocal<T> {
    locals: Box<[CachePadded<T>]>,
    registry: ComplementaryRegistry,
}

// This is safe because the `deref` call will return a reference to a `T` unique to each thread
// or it will panic for threads without an associated local. So there isn't a need for `T` to do
// it's own synchronization. The `verify` method on `ComplementaryRegistryId` has an issue where the
// id can be reused, but `WorkerLocal` has a reference to `ComplementaryRegistry` which will prevent
// any reuse.
unsafe impl<T: Send> Sync for WorkerLocal<T> {}

impl<T> WorkerLocal<T> {
    /// Creates a new worker local where the `initial` closure computes the
    /// value this worker local should take for each thread in the registry.
    #[inline]
    #[track_caller]
    pub fn new<F: FnMut(usize) -> T>(mut initial: F) -> WorkerLocal<T> {
        let registry = ComplementaryRegistry::current();
        WorkerLocal {
            locals: (0..registry.0.thread_limit.get())
                .map(|i| CachePadded::new(initial(i)))
                .collect(),
            registry,
        }
    }

    /// Returns the worker-local values for each thread
    #[inline]
    pub fn into_inner(self) -> impl Iterator<Item = T> {
        self.locals.into_vec().into_iter().map(CachePadded::into_inner)
    }
}

impl<T> Deref for WorkerLocal<T> {
    type Target = T;

    #[inline(always)]
    fn deref(&self) -> &T {
        // This is safe because `verify` will only return values less than
        // `self.registry.thread_limit` which is the size of the `self.locals` array.
        unsafe { &*self.locals.get_unchecked(self.registry.id().verify()) }
    }
}

impl<T: Default> Default for WorkerLocal<T> {
    fn default() -> Self {
        WorkerLocal::new(|_| T::default())
    }
}
