//@ignore-target: windows # No pthreads on Windows
//@ignore-target: android # No pthread_{get,set}name_np on Android

// pthread_setname_np / pthread_getname_np on invalid handles causes segfaults on Linux,
// so it is safe to assume that it is UB.

fn main() {
    let invalid_thread = 0xdeadbeef;

    let mut buf = [0; 64];
    let _res = unsafe { libc::pthread_getname_np(invalid_thread, buf.as_mut_ptr(), buf.len()) };
    //~^ERROR: invalid pthread_t
}
