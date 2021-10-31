//! Implements thread-local destructors that are not associated with any
//! particular data.

#![unstable(feature = "thread_local_internals", issue = "none")]
#![cfg(target_thread_local)]
use super::c;

// Using a per-thread list avoids the problems in synchronizing global state.
#[thread_local]
static mut DESTRUCTORS: Vec<(*mut u8, unsafe extern "C" fn(*mut u8))> = Vec::new();

pub unsafe fn register_dtor(t: *mut u8, dtor: unsafe extern "C" fn(*mut u8)) {
    DESTRUCTORS.push((t, dtor));
}

// See windows/thread_local_keys.rs for an explanation of this callback function.
// The short version is that all the function pointers in the `.CRT$XL*` array
// will be called whenever a thread or process starts or ends.

#[link_section = ".CRT$XLD"]
#[doc(hidden)]
#[used]
pub static TLS_CALLBACK: unsafe extern "system" fn(c::LPVOID, c::DWORD, c::LPVOID) = tls_callback;

unsafe extern "system" fn tls_callback(_: c::LPVOID, reason: c::DWORD, _: c::LPVOID) {
    if reason == c::DLL_THREAD_DETACH || reason == c::DLL_PROCESS_DETACH {
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
        DESTRUCTORS.shrink_to_fit();
    }
}
