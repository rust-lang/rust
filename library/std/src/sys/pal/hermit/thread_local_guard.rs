#![cfg(target_thread_local)]
#![unstable(feature = "thread_local_internals", issue = "none")]

pub fn activate() {
    // run_dtors is always executed by the threading support.
}
