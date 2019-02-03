//! Types and Traits for working with asynchronous tasks.

pub use core::task::*;

#[cfg(all(target_has_atomic = "ptr", target_has_atomic = "cas"))]
pub use if_arc::*;

#[cfg(all(target_has_atomic = "ptr", target_has_atomic = "cas"))]
mod if_arc {
    use super::*;
    use core::marker::PhantomData;
    use core::mem;
    use core::ptr::{self, NonNull};
    use crate::sync::Arc;

    /// A way of waking up a specific task.
    ///
    /// Any task executor must provide a way of signaling that a task it owns
    /// is ready to be `poll`ed again. Executors do so by implementing this trait.
    pub trait Wake: Send + Sync {
        /// Indicates that the associated task is ready to make progress and should
        /// be `poll`ed.
        ///
        /// Executors generally maintain a queue of "ready" tasks; `wake` should place
        /// the associated task onto this queue.
        fn wake(arc_self: &Arc<Self>);

        /// Indicates that the associated task is ready to make progress and should
        /// be `poll`ed. This function is like `wake`, but can only be called from the
        /// thread on which this `Wake` was created.
        ///
        /// Executors generally maintain a queue of "ready" tasks; `wake_local` should place
        /// the associated task onto this queue.
        #[inline]
        unsafe fn wake_local(arc_self: &Arc<Self>) {
            Self::wake(arc_self);
        }
    }

    #[cfg(all(target_has_atomic = "ptr", target_has_atomic = "cas"))]
    struct ArcWrapped<T>(PhantomData<T>);

    unsafe impl<T: Wake + 'static> UnsafeWake for ArcWrapped<T> {
        #[inline]
        unsafe fn clone_raw(&self) -> Waker {
            let me: *const ArcWrapped<T> = self;
            let arc = (*(&me as *const *const ArcWrapped<T> as *const Arc<T>)).clone();
            Waker::from(arc)
        }

        #[inline]
        unsafe fn drop_raw(&self) {
            let mut me: *const ArcWrapped<T> = self;
            let me = &mut me as *mut *const ArcWrapped<T> as *mut Arc<T>;
            ptr::drop_in_place(me);
        }

        #[inline]
        unsafe fn wake(&self) {
            let me: *const ArcWrapped<T> = self;
            T::wake(&*(&me as *const *const ArcWrapped<T> as *const Arc<T>))
        }

        #[inline]
        unsafe fn wake_local(&self) {
            let me: *const ArcWrapped<T> = self;
            T::wake_local(&*(&me as *const *const ArcWrapped<T> as *const Arc<T>))
        }
    }

    impl<T> From<Arc<T>> for Waker
        where T: Wake + 'static,
    {
        fn from(rc: Arc<T>) -> Self {
            unsafe {
                let ptr = mem::transmute::<Arc<T>, NonNull<ArcWrapped<T>>>(rc);
                Waker::new(ptr)
            }
        }
    }

    /// Creates a `LocalWaker` from a local `wake`.
    ///
    /// This function requires that `wake` is "local" (created on the current thread).
    /// The resulting `LocalWaker` will call `wake.wake_local()` when awoken, and
    /// will call `wake.wake()` if awoken after being converted to a `Waker`.
    #[inline]
    pub unsafe fn local_waker<W: Wake + 'static>(wake: Arc<W>) -> LocalWaker {
        let ptr = mem::transmute::<Arc<W>, NonNull<ArcWrapped<W>>>(wake);
        LocalWaker::new(ptr)
    }

    struct NonLocalAsLocal<T>(ArcWrapped<T>);

    unsafe impl<T: Wake + 'static> UnsafeWake for NonLocalAsLocal<T> {
        #[inline]
        unsafe fn clone_raw(&self) -> Waker {
            self.0.clone_raw()
        }

        #[inline]
        unsafe fn drop_raw(&self) {
            self.0.drop_raw()
        }

        #[inline]
        unsafe fn wake(&self) {
            self.0.wake()
        }

        #[inline]
        unsafe fn wake_local(&self) {
            // Since we're nonlocal, we can't call wake_local
            self.0.wake()
        }
    }

    /// Creates a `LocalWaker` from a non-local `wake`.
    ///
    /// This function is similar to `local_waker`, but does not require that `wake`
    /// is local to the current thread. The resulting `LocalWaker` will call
    /// `wake.wake()` when awoken.
    #[inline]
    pub fn local_waker_from_nonlocal<W: Wake + 'static>(wake: Arc<W>) -> LocalWaker {
        unsafe {
            let ptr = mem::transmute::<Arc<W>, NonNull<NonLocalAsLocal<W>>>(wake);
            LocalWaker::new(ptr)
        }
    }
}
