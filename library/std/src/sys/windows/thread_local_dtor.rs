//! Implements thread-local destructors that are not associated with any
//! particular data.

#![unstable(feature = "thread_local_internals", issue = "none")]
#![cfg(target_thread_local)]

// Using a per-thread list avoids the problems in synchronizing global state.
#[thread_local]
static mut DESTRUCTORS: Vec<(*mut u8, unsafe extern "C" fn(*mut u8))> = Vec::new();

pub unsafe fn register_dtor(t: *mut u8, dtor: unsafe extern "C" fn(*mut u8)) {
    DESTRUCTORS.push((t, dtor));
}

/// Runs destructors. This should not be called until thread exit.
pub unsafe fn run_keyless_dtors() {
    // Drop all the destructors.
    //
    // Note: While this is potentially an infinite loop, it *should* be
    // the case that this loop always terminates because we provide the
    // guarantee that a TLS key cannot be set after it is flagged for
    // destruction.
    while let Some((ptr, dtor)) = DESTRUCTORS.pop() {
        (dtor)(ptr);
    }
    // We're done so free the memory.
    DESTRUCTORS = Vec::new();
}
