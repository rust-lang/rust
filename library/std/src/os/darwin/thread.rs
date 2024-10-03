//! Darwin-specific extensions to threads.
#![unstable(feature = "darwin_mtm", issue = "none")]

use crate::fmt;
use crate::marker::PhantomData;

/// A marker type for functionality only available on the main thread.
///
/// The main thread is a system-level property on Darwin platforms, and has extra capabilities not
/// available on other threads. This is usually relevant when using native GUI frameworks, where
/// most operations must be done on the main thread.
///
/// This type enables you to manage that capability. By design, it is neither [`Send`] nor [`Sync`],
/// and can only be created on the main thread, meaning that if you have an instance of this, you
/// are guaranteed to be on the main thread / have the "main-thread capability".
///
/// [The `main` function][main-functions] will run on the main thread. This type can also be used
/// with `#![no_main]` or other such cases where Rust is not defining the binary entry point.
///
/// See the following links for more information on main-thread-only APIs:
/// - [Are the Cocoa Frameworks Thread Safe?](https://developer.apple.com/library/archive/documentation/Cocoa/Conceptual/CocoaFundamentals/AddingBehaviortoaCocoaProgram/AddingBehaviorCocoa.html#//apple_ref/doc/uid/TP40002974-CH5-SW47)
/// - [About Threaded Programming](https://developer.apple.com/library/archive/documentation/Cocoa/Conceptual/Multithreading/AboutThreads/AboutThreads.html)
/// - [Thread Safety Summary](https://developer.apple.com/library/archive/documentation/Cocoa/Conceptual/Multithreading/ThreadSafetySummary/ThreadSafetySummary.html#//apple_ref/doc/uid/10000057i-CH12-SW1)
/// - [Technical Note TN2028 - Threading Architectures](https://developer.apple.com/library/archive/technotes/tn/tn2028.html#//apple_ref/doc/uid/DTS10003065)
/// - [Thread Management](https://developer.apple.com/library/archive/documentation/Cocoa/Conceptual/Multithreading/CreatingThreads/CreatingThreads.html)
/// - [Swift's `@MainActor`](https://developer.apple.com/documentation/swift/mainactor)
/// - [Main Thread Only APIs on OS X](https://www.dribin.org/dave/blog/archives/2009/02/01/main_thread_apis/)
/// - [Mike Ash' article on thread safety](https://www.mikeash.com/pyblog/friday-qa-2009-01-09.html)
///
/// [main-functions]: https://doc.rust-lang.org/reference/crates-and-source-files.html#main-functions
///
///
/// # Main Thread Checker
///
/// Xcode provides a tool called the ["Main Thread Checker"][mtc] which verifies that UI APIs are
/// being used from the correct thread. This is not as principled as `MainThreadMarker`, but is
/// helpful for catching mistakes.
///
/// You can use this tool on macOS by loading `libMainThreadChecker.dylib` into your process using
/// `DYLD_INSERT_LIBRARIES`:
///
/// ```console
/// DYLD_INSERT_LIBRARIES=/Applications/Xcode.app/Contents/Developer/usr/lib/libMainThreadChecker.dylib MTC_RESET_INSERT_LIBRARIES=0 cargo run
/// ```
///
/// If you're not running your binary through Cargo, you can omit
/// [`MTC_RESET_INSERT_LIBRARIES`][mtc-reset].
///
/// ```console
/// DYLD_INSERT_LIBRARIES=/Applications/Xcode.app/Contents/Developer/usr/lib/libMainThreadChecker.dylib target/debug/myapp
/// ```
///
/// If you're developing for iOS, you probably better off enabling the tool in Xcode's own UI.
///
/// See [this excellent blog post][mtc-cfg] for details on further configuration options.
///
/// [mtc]: https://developer.apple.com/documentation/xcode/diagnosing-memory-thread-and-crash-issues-early#Detect-improper-UI-updates-on-background-threads
/// [mtc-reset]: https://bryce.co/main-thread-checker-configuration/#mtc_reset_insert_libraries
/// [mtc-cfg]: https://bryce.co/main-thread-checker-configuration/
///
///
/// # Examples
///
/// Retrieve the main thread marker in different situations.
///
/// ```
/// #![feature(darwin_mtm)]
/// use std::os::darwin::thread::MainThreadMarker;
///
/// # // doc test explicitly uses `fn main` to show that that's where it counts.
/// fn main() {
///     // The thread that `fn main` runs on is the main thread.
///     assert!(MainThreadMarker::new().is_some());
///
///     // Subsequently spawned threads are not the main thread.
///     std::thread::spawn(|| {
///         assert!(MainThreadMarker::new().is_none());
///     }).join().unwrap();
/// }
/// ```
///
/// Create a static that is only usable on the main thread. This is similar to a thread-local, but
/// can be more efficient because it doesn't handle multiple threads.
///
/// ```
/// #![feature(sync_unsafe_cell)]
/// #![feature(darwin_mtm)]
/// use std::os::darwin::thread::MainThreadMarker;
/// use std::cell::SyncUnsafeCell;
///
/// static MAIN_THREAD_ONLY_VALUE: SyncUnsafeCell<i32> = SyncUnsafeCell::new(0);
///
/// fn set(value: i32, _mtm: MainThreadMarker) {
///     // SAFETY: We have an instance of `MainThreadMarker`, so we know that
///     // we're running on the main thread (and thus do not need any
///     // synchronization, since the only accesses to this value is from the
///     // main thread).
///     unsafe { *MAIN_THREAD_ONLY_VALUE.get() = value };
/// }
///
/// fn get(_mtm: MainThreadMarker) -> i32 {
///     // SAFETY: Same as above.
///     unsafe { *MAIN_THREAD_ONLY_VALUE.get() }
/// }
///
/// // Usage
/// fn main() {
///     let mtm = MainThreadMarker::new().expect("must be on the main thread");
///     set(42, mtm);
///     assert_eq!(get(mtm), 42);
/// }
/// ```
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
//              ^^^^ this is valid because it's still `!Send` and `!Sync`.
#[unstable(feature = "darwin_mtm", issue = "none")]
pub struct MainThreadMarker {
    // No lifetime information needed; the main thread is static and available throughout the entire
    // program!

    // Ensure `!Send` and `!Sync`.
    _priv: PhantomData<*mut ()>,
}

// Manually implementing these results in slightly better error messages.
#[unstable(feature = "darwin_mtm", issue = "none")]
impl !Send for MainThreadMarker {}
#[unstable(feature = "darwin_mtm", issue = "none")]
impl !Sync for MainThreadMarker {}

impl MainThreadMarker {
    /// Construct a new `MainThreadMarker`.
    ///
    /// Returns [`None`] if the current thread was not the main thread.
    ///
    ///
    /// # Example
    ///
    /// Check whether the current thread is the main thread.
    ///
    /// ```
    /// #![feature(darwin_mtm)]
    /// use std::os::darwin::thread::MainThreadMarker;
    ///
    /// if MainThreadMarker::new().is_some() {
    ///     // Is the main thread
    /// } else {
    ///     // Not the main thread
    /// }
    /// ```
    #[inline]
    #[doc(alias = "is_main_thread")]
    #[doc(alias = "pthread_main_np")]
    #[doc(alias = "isMainThread")]
    #[unstable(feature = "darwin_mtm", issue = "none")]
    pub fn new() -> Option<Self> {
        if is_main_thread() {
            // SAFETY: We just checked that we are running on the main thread.
            Some(unsafe { Self::new_unchecked() })
        } else {
            None
        }
    }

    /// Construct a new `MainThreadMarker` without first checking whether the current thread is
    /// the main one.
    ///
    ///
    /// # Safety
    ///
    /// The current thread must be the main thread.
    #[inline]
    #[unstable(feature = "darwin_mtm", issue = "none")]
    #[rustc_const_unstable(feature = "darwin_mtm", issue = "none")]
    pub const unsafe fn new_unchecked() -> Self {
        // SAFETY: Upheld by caller.
        //
        // We can't debug_assert that this actually is the main thread, both because this is
        // `const` (to allow usage in `static`s), and because users may sometimes want to create
        // this briefly, e.g. to access an API that in most cases requires the marker, but is safe
        // to use without in specific cases.
        Self { _priv: PhantomData }
    }
}

#[unstable(feature = "darwin_mtm", issue = "none")]
impl fmt::Debug for MainThreadMarker {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("MainThreadMarker").finish()
    }
}

// Implementation:

/// Whether the current thread is the main thread.
#[inline]
fn is_main_thread() -> bool {
    // In Objective-C you would use `+[NSThread isMainThread]`, but benchmarks have shown that
    // calling the underlying `pthread_main_np` directly is up to four times faster, so we use that
    // instead.
    //
    // `pthread_main_np` is also included via. libSystem, so that avoids linking Foundation.

    // SAFETY: Can be called from any thread.
    //
    // Apple's man page says:
    // > The pthread_main_np() function returns 1 if the calling thread is the initial thread, 0 if
    // > the calling thread is not the initial thread, and -1 if the thread's initialization has not
    // > yet completed.
    //
    // However, Apple's header says:
    // > Returns non-zero if the current thread is the main thread.
    //
    // So unclear if we should be doing a comparison against 1, or a negative comparison against 0?
    // To be safe, we compare against 1, though in reality, the current implementation can only ever
    // return 0 or 1:
    // https://github.com/apple-oss-distributions/libpthread/blob/libpthread-535/src/pthread.c#L1084-L1089
    unsafe { libc::pthread_main_np() == 1 }
}
