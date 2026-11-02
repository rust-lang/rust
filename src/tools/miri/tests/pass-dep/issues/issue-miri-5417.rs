//@ignore-target: windows # No pthreads on Windows
use std::ffi::c_void;
use std::mem::MaybeUninit;

unsafe extern "C" {
    fn pthread_key_create(
        k: *mut libc::pthread_key_t,
        dtor: MaybeUninit<extern "C" fn(*mut c_void)>,
    ) -> i32;
}

extern "C" fn my_dtor(_: *mut c_void) {}

fn main() {
    unsafe {
        let mut key: libc::pthread_key_t = 0;
        assert_eq!(pthread_key_create(&mut key, MaybeUninit::new(my_dtor)), 0);
    }
}
