//@ignore-target: windows # No pthreads on Windows
//@run-native

use std::{process, ptr};

unsafe extern "C" fn dtor(_ptr: *mut libc::c_void) {
    process::abort();
}

// Returning from `main` does not call the destructor (consistent with the behavior on `exit`,
// which is tested in `pthread_key.rs`).
fn main() {
    unsafe {
        let dtor = dtor as unsafe extern "C" fn(*mut libc::c_void);

        let mut key: libc::pthread_key_t = 0;
        assert_eq!(libc::pthread_key_create(&mut key, Some(dtor)), 0);

        let r = libc::pthread_setspecific(key, ptr::without_provenance_mut(42));
        assert_eq!(r, 0);
    }
}
