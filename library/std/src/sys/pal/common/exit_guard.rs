cfg_if::cfg_if! {
    if #[cfg(target_os = "linux")] {
        /// Mitigation for <https://github.com/rust-lang/rust/issues/126600>
        ///
        /// On `unix` (where `libc::exit` may not be thread-safe), ensure that only one Rust thread
        /// calls `libc::exit` (or returns from `main`) by calling this function before calling
        /// `libc::exit` (or returning from `main`).
        ///
        /// Technically not enough to ensure soundness, since other code directly calling
        /// libc::exit will still race with this.
        ///
        /// *This function does not itself call `libc::exit`.* This is so it can also be used
        /// to guard returning from `main`.
        ///
        /// This function will return only the first time it is called in a process.
        ///
        /// * If it is called again on the same thread as the first call, it will abort.
        /// * If it is called again on a different thread, it will `thread::park()` in a loop
        ///   (waiting for the process to exit).
        pub(crate) fn unique_thread_exit() {
            let this_thread_id = unsafe { libc::gettid() };
            debug_assert_ne!(this_thread_id, 0, "thread ID cannot be zero");
            #[cfg(target_has_atomic = "32")]
            {
                use crate::sync::atomic::{AtomicI32, Ordering};
                static EXITING_THREAD_ID: AtomicI32 = AtomicI32::new(0);
                match EXITING_THREAD_ID.compare_exchange(
                    0,
                    this_thread_id,
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                ) {
                    Ok(_zero) => {
                        // This is the first thread to call `unique_thread_exit`,
                        // and this is the first time it is called.
                        // Set EXITING_THREAD_ID to this thread's ID (done by the
                        // compare_exchange) and return.
                    }
                    Err(id) if id == this_thread_id => {
                        // This is the first thread to call `unique_thread_exit`,
                        // but this is the second time it is called.
                        // Abort the process.
                        core::panicking::panic_nounwind("std::process::exit called re-entrantly")
                    }
                    Err(_) => {
                        // This is not the first thread to call `unique_thread_exit`.
                        // Park until the process exits.
                        loop {
                            crate::thread::park();
                        }
                    }
                }
            }
            #[cfg(not(target_has_atomic = "32"))]
            {
                use crate::sync::{Mutex, PoisonError};
                static EXITING_THREAD_ID: Mutex<i32> = Mutex::new(0);
                let mut exiting_thread_id =
                    EXITING_THREAD_ID.lock().unwrap_or_else(PoisonError::into_inner);
                if *exiting_thread_id == 0 {
                    // This is the first thread to call `unique_thread_exit`,
                    // and this is the first time it is called.
                    // Set EXITING_THREAD_ID to this thread's ID and return.
                    *exiting_thread_id = this_thread_id;
                } else if *exiting_thread_id == this_thread_id {
                    // This is the first thread to call `unique_thread_exit`,
                    // but this is the second time it is called.
                    // Abort the process.
                    core::panicking::panic_nounwind("std::process::exit called re-entrantly")
                } else {
                    // This is not the first thread to call `unique_thread_exit`.
                    // Park until the process exits.
                    drop(exiting_thread_id);
                    loop {
                        crate::thread::park();
                    }
                }
            }
        }
    } else {
        /// Mitigation for <https://github.com/rust-lang/rust/issues/126600>
        ///
        /// Mitigation is ***NOT*** implemented on this platform, either because this platform
        /// is not affected, or because mitigation is not yet implemented for this platform.
        pub(crate) fn unique_thread_exit() {
            // Mitigation not required on platforms where `exit` is thread-safe.
        }
    }
}
