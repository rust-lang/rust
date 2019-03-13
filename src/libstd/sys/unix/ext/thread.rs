//! Unix-specific extensions to primitives in the `std::thread` module.

#![stable(feature = "thread_extensions", since = "1.9.0")]

#[allow(deprecated)]
use crate::os::unix::raw::pthread_t;
use crate::sys_common::{AsInner, IntoInner};
use crate::thread::JoinHandle;

#[stable(feature = "thread_extensions", since = "1.9.0")]
#[allow(deprecated)]
pub type RawPthread = pthread_t;

/// Unix-specific extensions to [`thread::JoinHandle`].
///
/// [`thread::JoinHandle`]: ../../../../std/thread/struct.JoinHandle.html
#[stable(feature = "thread_extensions", since = "1.9.0")]
pub trait JoinHandleExt {
    /// Extracts the raw pthread_t without taking ownership
    #[stable(feature = "thread_extensions", since = "1.9.0")]
    fn as_pthread_t(&self) -> RawPthread;

    /// Consumes the thread, returning the raw pthread_t
    ///
    /// This function **transfers ownership** of the underlying pthread_t to
    /// the caller. Callers are then the unique owners of the pthread_t and
    /// must either detach or join the pthread_t once it's no longer needed.
    #[stable(feature = "thread_extensions", since = "1.9.0")]
    fn into_pthread_t(self) -> RawPthread;
}

#[stable(feature = "thread_extensions", since = "1.9.0")]
impl<T> JoinHandleExt for JoinHandle<T> {
    fn as_pthread_t(&self) -> RawPthread {
        self.as_inner().id() as RawPthread
    }

    fn into_pthread_t(self) -> RawPthread {
        self.into_inner().into_id() as RawPthread
    }
}
