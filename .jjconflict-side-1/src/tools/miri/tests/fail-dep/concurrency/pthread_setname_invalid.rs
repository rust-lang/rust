//@ignore-target: windows # No pthreads on Windows
//@ignore-target: android # No pthread_{get,set}name_np on Android
//@ignore-target: apple # macOS has no pthread_setname_np for other threads

// pthread_setname_np / pthread_getname_np on invalid handles causes segfaults on Linux,
// so it is safe to assume that it is UB.

fn main() {
    let invalid_thread = 0xdeadbeef;

    let _res = unsafe { libc::pthread_setname_np(invalid_thread, [0].as_ptr()) };
    //~^ERROR: invalid pthread_t
}
