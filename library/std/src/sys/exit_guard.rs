cfg_select! {
    target_os = "linux" => {
        /// Mitigation for <https://github.com/rust-lang/rust/issues/126600>
        ///
        /// On glibc, `libc::exit` has been observed to not always be thread-safe.
        /// It is currently unclear whether that is a glibc bug or allowed by the standard.
        /// To mitigate this problem, we ensure that only one
        /// Rust thread calls `libc::exit` (or returns from `main`) by calling this function before
        /// calling `libc::exit` (or returning from `main`).
        ///
        /// Technically, this is not enough to ensure soundness, since other code directly calling
        /// `libc::exit` will still race with this.
        ///
        /// *This function does not itself call `libc::exit`.* This is so it can also be used
        /// to guard returning from `main`.
        ///
        /// This function will return only the first time it is called in a process.
        ///
        /// * If it is called again on the same thread as the first call, it will abort.
        /// * If it is called again on a different thread, it will wait in a loop
        ///   (waiting for the process to exit).
        #[cfg_attr(any(test, doctest), allow(dead_code))]
        pub(crate) fn unique_thread_exit() {
            use crate::ffi::c_int;
            use crate::ptr;
            use crate::sync::atomic::AtomicPtr;
            use crate::sync::atomic::Ordering::{Acquire, Relaxed};

            static EXITING_THREAD_ID: AtomicPtr<c_int> = AtomicPtr::new(ptr::null_mut());

            // We use the address of `errno` as a cheap and safe way to identify
            // threads. As the C standard mandates that `errno` must have thread
            // storage duration, we can rely on its address not changing over the
            // lifetime of the thread. Additionally, accesses to `errno` are
            // async-signal-safe, so this function is available in all imaginable
            // circumstances.
            let this_thread_id = crate::sys::os::errno_location();
            match EXITING_THREAD_ID.compare_exchange(ptr::null_mut(), this_thread_id, Acquire, Relaxed) {
                Ok(_) => {
                    // This is the first thread to call `unique_thread_exit`,
                    // and this is the first time it is called. Continue exiting.
                }
                Err(exiting_thread_id) if exiting_thread_id == this_thread_id => {
                    // This is the first thread to call `unique_thread_exit`,
                    // but this is the second time it is called.
                    // Abort the process.
                    core::panicking::panic_nounwind("std::process::exit called re-entrantly")
                }
                Err(_) => {
                    // This is not the first thread to call `unique_thread_exit`.
                    // Pause until the process exits.
                    loop {
                        // Safety: libc::pause is safe to call.
                        unsafe { libc::pause(); }
                    }
                }
            }
        }
    }
    _ => {
        /// Mitigation for <https://github.com/rust-lang/rust/issues/126600>
        ///
        /// Mitigation is ***NOT*** implemented on this platform, either because this platform
        /// is not affected, or because mitigation is not yet implemented for this platform.
        #[cfg_attr(any(test, doctest), allow(dead_code))]
        pub(crate) fn unique_thread_exit() {
            // Mitigation not required on platforms where `exit` is thread-safe.
        }
    }
}
