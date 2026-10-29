// Tests that forced unwind through POF Rust frames wouldn't trigger our terminating guards.

#![no_main]

extern "C-unwind" {
    fn pthread_exit(v: *mut core::ffi::c_void) -> !;
}

unsafe extern "C" fn call_pthread_exit() {
    pthread_exit(core::ptr::null_mut());
}

#[no_mangle]
unsafe extern "C-unwind" fn main(_argc: core::ffi::c_int, _argv: *mut *mut core::ffi::c_char) {
    call_pthread_exit();
}
