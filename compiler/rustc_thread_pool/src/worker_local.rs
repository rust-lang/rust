use std::fmt;
use std::ops::Deref;
use std::sync::Arc;

use crate::registry::{Registry, WorkerThread};

#[repr(align(64))]
#[derive(Debug)]
struct CacheAligned<T>(T);

/// Holds worker-locals values for each thread in a thread pool.
/// You can only access the worker local value through the Deref impl
/// on the thread pool it was constructed on. It will panic otherwise
pub struct WorkerLocal<T> {
    locals: Vec<CacheAligned<T>>,
    registry: Arc<Registry>,
}

/// We prevent concurrent access to the underlying value in the
/// Deref impl, thus any values safe to send across threads can
/// be used with WorkerLocal.
unsafe impl<T: Send> Sync for WorkerLocal<T> {}

impl<T> WorkerLocal<T> {
    /// Creates a new worker local where the `initial` closure computes the
    /// value this worker local should take for each thread in the thread pool.
    #[inline]
    pub fn new<F: FnMut(usize) -> T>(mut initial: F) -> WorkerLocal<T> {
        let registry = Registry::current();
        WorkerLocal {
            locals: (0..registry.num_threads()).map(|i| CacheAligned(initial(i))).collect(),
            registry,
        }
    }

    /// Returns the worker-local value for each thread
    #[inline]
    pub fn into_inner(self) -> Vec<T> {
        self.locals.into_iter().map(|c| c.0).collect()
    }

    fn current(&self) -> &T {
        unsafe {
            let worker_thread = WorkerThread::current();
            if worker_thread.is_null()
                || &*(*worker_thread).registry as *const _ != &*self.registry as *const _
            {
                panic!("WorkerLocal can only be used on the thread pool it was created on")
            }
            &self.locals[(*worker_thread).index].0
        }
    }
}

impl<T> WorkerLocal<Vec<T>> {
    /// Joins the elements of all the worker locals into one Vec
    pub fn join(self) -> Vec<T> {
        self.into_inner().into_iter().flat_map(|v| v).collect()
    }
}

impl<T: fmt::Debug> fmt::Debug for WorkerLocal<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("WorkerLocal").field("registry", &self.registry.id()).finish()
    }
}

impl<T> Deref for WorkerLocal<T> {
    type Target = T;

    #[inline(always)]
    fn deref(&self) -> &T {
        self.current()
    }
}
