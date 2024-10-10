cfg_if::cfg_if! {
    if #[cfg(target_os = "linux")] {
        use crate::mem;
        use crate::sync::atomic::{AtomicUsize, Ordering};

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
            let this_thread_id = unsafe { libc::pthread_self() };

            const _: () = assert!(mem::size_of::<libc::pthread_t>() <= mem::size_of::<usize>());

            const NONE: usize = 0;
            static EXITING_THREAD_ID: AtomicUsize = AtomicUsize::new(NONE);
            match EXITING_THREAD_ID.compare_exchange(NONE, this_thread_id as usize, Ordering::Relaxed, Ordering::Relaxed).unwrap_or_else(|id| id) {
                NONE => {
                    // This is the first thread to call `unique_thread_exit`,
                    // and this is the first time it is called.
                    //
                    // We set `EXITING_THREAD_ID` to this thread's ID already
                    // and will return.
                }
                id if id == this_thread_id as usize => {
                    // This is the first thread to call `unique_thread_exit`,
                    // but this is the second time it is called.
                    //
                    // Abort the process.
                    core::panicking::panic_nounwind("std::process::exit called re-entrantly")
                }
                _ => {
                    // This is not the first thread to call `unique_thread_exit`.
                    // Pause until the process exits.
                    loop {
                        // Safety: libc::pause is safe to call.
                        unsafe { libc::pause(); }
                    }
                }
            }
        }
    } else {
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
