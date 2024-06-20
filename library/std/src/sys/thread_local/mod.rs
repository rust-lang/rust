#![unstable(feature = "thread_local_internals", reason = "should not be necessary", issue = "none")]
#![cfg_attr(test, allow(unused))]

// There are three thread-local implementations: "static", "fast", "OS".
// The "OS" thread local key type is accessed via platform-specific API calls and is slow, while the
// "fast" key type is accessed via code generated via LLVM, where TLS keys are set up by the linker.
// "static" is for single-threaded platforms where a global static is sufficient.

cfg_if::cfg_if! {
    if #[cfg(any(all(target_family = "wasm", not(target_feature = "atomics")), target_os = "uefi"))] {
        #[doc(hidden)]
        mod static_local;
        #[doc(hidden)]
        pub use static_local::{EagerStorage, LazyStorage, thread_local_inner};
    } else if #[cfg(target_thread_local)] {
        #[doc(hidden)]
        mod fast_local;
        #[doc(hidden)]
        pub use fast_local::{EagerStorage, LazyStorage, thread_local_inner};
    } else {
        #[doc(hidden)]
        mod os_local;
        #[doc(hidden)]
        pub use os_local::{Key, thread_local_inner};
    }
}

/// Run a callback in a scenario which must not unwind (such as a `extern "C"
/// fn` declared in a user crate). If the callback unwinds anyway, then
/// `rtabort` with a message about thread local panicking on drop.
#[inline]
#[allow(dead_code)]
fn abort_on_dtor_unwind(f: impl FnOnce()) {
    // Using a guard like this is lower cost.
    let guard = DtorUnwindGuard;
    f();
    core::mem::forget(guard);

    struct DtorUnwindGuard;
    impl Drop for DtorUnwindGuard {
        #[inline]
        fn drop(&mut self) {
            // This is not terribly descriptive, but it doesn't need to be as we'll
            // already have printed a panic message at this point.
            rtabort!("thread local panicked on drop");
        }
    }
}
