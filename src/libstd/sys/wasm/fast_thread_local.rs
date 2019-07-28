#![unstable(feature = "thread_local_internals", issue = "0")]

pub unsafe fn register_dtor(_t: *mut u8, _dtor: unsafe extern fn(*mut u8)) {
    // FIXME: right now there is no concept of "thread exit", but this is likely
    // going to show up at some point in the form of an exported symbol that the
    // wasm runtime is oging to be expected to call. For now we basically just
    // ignore the arguments, but if such a function starts to exist it will
    // likely look like the OSX implementation in `unix/fast_thread_local.rs`
}
