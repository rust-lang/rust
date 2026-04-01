use super::id::ThreadId;
use super::main_thread;
use crate::alloc::System;
use crate::ffi::CStr;
use crate::fmt;
use crate::pin::Pin;
use crate::sync::Arc;
use crate::sys::sync::Parker;
use crate::time::Duration;

// This module ensures private fields are kept private, which is necessary to enforce the safety requirements.
mod thread_name_string {
    use crate::ffi::{CStr, CString};
    use crate::str;

    /// Like a `String` it's guaranteed UTF-8 and like a `CString` it's null terminated.
    pub(crate) struct ThreadNameString {
        inner: CString,
    }

    impl From<String> for ThreadNameString {
        fn from(s: String) -> Self {
            Self {
                inner: CString::new(s).expect("thread name may not contain interior null bytes"),
            }
        }
    }

    impl ThreadNameString {
        pub fn as_cstr(&self) -> &CStr {
            &self.inner
        }

        pub fn as_str(&self) -> &str {
            // SAFETY: `ThreadNameString` is guaranteed to be UTF-8.
            unsafe { str::from_utf8_unchecked(self.inner.to_bytes()) }
        }
    }
}

use thread_name_string::ThreadNameString;

/// The internal representation of a `Thread` handle
///
/// We explicitly set the alignment for our guarantee in Thread::into_raw. This
/// allows applications to stuff extra metadata bits into the alignment, which
/// can be rather useful when working with atomics.
#[repr(align(8))]
struct Inner {
    name: Option<ThreadNameString>,
    id: ThreadId,
    parker: Parker,
}

impl Inner {
    fn parker(self: Pin<&Self>) -> Pin<&Parker> {
        unsafe { Pin::map_unchecked(self, |inner| &inner.parker) }
    }
}

#[derive(Clone)]
#[stable(feature = "rust1", since = "1.0.0")]
/// A handle to a thread.
///
/// Threads are represented via the `Thread` type, which you can get in one of
/// two ways:
///
/// * By spawning a new thread, e.g., using the [`thread::spawn`]
///   function, and calling [`thread`] on the [`JoinHandle`].
/// * By requesting the current thread, using the [`thread::current`] function.
///
/// The [`thread::current`] function is available even for threads not spawned
/// by the APIs of this module.
///
/// There is usually no need to create a `Thread` struct yourself, one
/// should instead use a function like `spawn` to create new threads, see the
/// docs of [`Builder`] and [`spawn`] for more details.
///
/// [`thread::spawn`]: super::spawn
/// [`thread`]: super::JoinHandle::thread
/// [`JoinHandle`]: super::JoinHandle
/// [`thread::current`]: super::current::current
/// [`Builder`]: super::Builder
/// [`spawn`]: super::spawn
pub struct Thread {
    // We use the System allocator such that creating or dropping this handle
    // does not interfere with a potential Global allocator using thread-local
    // storage.
    inner: Pin<Arc<Inner, System>>,
}

impl Thread {
    pub(crate) fn new(id: ThreadId, name: Option<String>) -> Thread {
        let name = name.map(ThreadNameString::from);

        // We have to use `unsafe` here to construct the `Parker` in-place,
        // which is required for the UNIX implementation.
        //
        // SAFETY: We pin the Arc immediately after creation, so its address never
        // changes.
        let inner = unsafe {
            let mut arc = Arc::<Inner, _>::new_uninit_in(System);
            let ptr = Arc::get_mut_unchecked(&mut arc).as_mut_ptr();
            (&raw mut (*ptr).name).write(name);
            (&raw mut (*ptr).id).write(id);
            Parker::new_in_place(&raw mut (*ptr).parker);
            Pin::new_unchecked(arc.assume_init())
        };

        Thread { inner }
    }

    /// Like the public [`park`], but callable on any handle. This is used to
    /// allow parking in TLS destructors.
    ///
    /// # Safety
    /// May only be called from the thread to which this handle belongs.
    ///
    /// [`park`]: super::park
    pub(crate) unsafe fn park(&self) {
        unsafe { self.inner.as_ref().parker().park() }
    }

    /// Like the public [`park_timeout`], but callable on any handle. This is
    /// used to allow parking in TLS destructors.
    ///
    /// # Safety
    /// May only be called from the thread to which this handle belongs.
    ///
    /// [`park_timeout`]: super::park_timeout
    pub(crate) unsafe fn park_timeout(&self, dur: Duration) {
        unsafe { self.inner.as_ref().parker().park_timeout(dur) }
    }

    /// Atomically makes the handle's token available if it is not already.
    ///
    /// Every thread is equipped with some basic low-level blocking support, via
    /// the [`park`] function and the `unpark()` method. These can be used as a
    /// more CPU-efficient implementation of a spinlock.
    ///
    /// See the [park documentation] for more details.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::thread;
    /// use std::time::Duration;
    /// use std::sync::atomic::{AtomicBool, Ordering};
    ///
    /// static QUEUED: AtomicBool = AtomicBool::new(false);
    ///
    /// let parked_thread = thread::Builder::new()
    ///     .spawn(|| {
    ///         println!("Parking thread");
    ///         QUEUED.store(true, Ordering::Release);
    ///         thread::park();
    ///         println!("Thread unparked");
    ///     })
    ///     .unwrap();
    ///
    /// // Let some time pass for the thread to be spawned.
    /// thread::sleep(Duration::from_millis(10));
    ///
    /// // Wait until the other thread is queued.
    /// // This is crucial! It guarantees that the `unpark` below is not consumed
    /// // by some other code in the parked thread (e.g. inside `println!`).
    /// while !QUEUED.load(Ordering::Acquire) {
    ///     // Spinning is of course inefficient; in practice, this would more likely be
    ///     // a dequeue where we have no work to do if there's nobody queued.
    ///     std::hint::spin_loop();
    /// }
    ///
    /// println!("Unpark the thread");
    /// parked_thread.thread().unpark();
    ///
    /// parked_thread.join().unwrap();
    /// ```
    ///
    /// [`park`]: super::park
    /// [park documentation]: super::park
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn unpark(&self) {
        self.inner.as_ref().parker().unpark();
    }

    /// Gets the thread's unique identifier.
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
    #[stable(feature = "thread_id", since = "1.19.0")]
    #[must_use]
    pub fn id(&self) -> ThreadId {
        self.inner.id
    }

    /// Gets the thread's name.
    ///
    /// For more information about named threads, see
    /// [this module-level documentation][naming-threads].
    ///
    /// # Examples
    ///
    /// Threads by default have no name specified:
    ///
    /// ```
    /// use std::thread;
    ///
    /// let builder = thread::Builder::new();
    ///
    /// let handler = builder.spawn(|| {
    ///     assert!(thread::current().name().is_none());
    /// }).unwrap();
    ///
    /// handler.join().unwrap();
    /// ```
    ///
    /// Thread with a specified name:
    ///
    /// ```
    /// use std::thread;
    ///
    /// let builder = thread::Builder::new()
    ///     .name("foo".into());
    ///
    /// let handler = builder.spawn(|| {
    ///     assert_eq!(thread::current().name(), Some("foo"))
    /// }).unwrap();
    ///
    /// handler.join().unwrap();
    /// ```
    ///
    /// [naming-threads]: ./index.html#naming-threads
    #[stable(feature = "rust1", since = "1.0.0")]
    #[must_use]
    pub fn name(&self) -> Option<&str> {
        if let Some(name) = &self.inner.name {
            Some(name.as_str())
        } else if main_thread::get() == Some(self.inner.id) {
            Some("main")
        } else {
            None
        }
    }

    /// Consumes the `Thread`, returning a raw pointer.
    ///
    /// To avoid a memory leak the pointer must be converted
    /// back into a `Thread` using [`Thread::from_raw`]. The pointer is
    /// guaranteed to be aligned to at least 8 bytes.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(thread_raw)]
    ///
    /// use std::thread::{self, Thread};
    ///
    /// let thread = thread::current();
    /// let id = thread.id();
    /// let ptr = Thread::into_raw(thread);
    /// unsafe {
    ///     assert_eq!(Thread::from_raw(ptr).id(), id);
    /// }
    /// ```
    #[unstable(feature = "thread_raw", issue = "97523")]
    pub fn into_raw(self) -> *const () {
        // Safety: We only expose an opaque pointer, which maintains the `Pin` invariant.
        let inner = unsafe { Pin::into_inner_unchecked(self.inner) };
        Arc::into_raw_with_allocator(inner).0 as *const ()
    }

    /// Constructs a `Thread` from a raw pointer.
    ///
    /// The raw pointer must have been previously returned
    /// by a call to [`Thread::into_raw`].
    ///
    /// # Safety
    ///
    /// This function is unsafe because improper use may lead
    /// to memory unsafety, even if the returned `Thread` is never
    /// accessed.
    ///
    /// Creating a `Thread` from a pointer other than one returned
    /// from [`Thread::into_raw`] is **undefined behavior**.
    ///
    /// Calling this function twice on the same raw pointer can lead
    /// to a double-free if both `Thread` instances are dropped.
    #[unstable(feature = "thread_raw", issue = "97523")]
    pub unsafe fn from_raw(ptr: *const ()) -> Thread {
        // Safety: Upheld by caller.
        unsafe {
            Thread { inner: Pin::new_unchecked(Arc::from_raw_in(ptr as *const Inner, System)) }
        }
    }

    pub(crate) fn cname(&self) -> Option<&CStr> {
        if let Some(name) = &self.inner.name {
            Some(name.as_cstr())
        } else if main_thread::get() == Some(self.inner.id) {
            Some(c"main")
        } else {
            None
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl fmt::Debug for Thread {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Thread")
            .field("id", &self.id())
            .field("name", &self.name())
            .finish_non_exhaustive()
    }
}
