//@ignore-target: windows # No pthreads on Windows
//@normalize-stderr-test: "attempting to access \d+ bytes" -> "attempting to access N bytes"
use std::ffi::c_void;

unsafe extern "C" {
    fn pthread_key_create(k: *mut c_void, dtor: extern "C" fn(*mut c_void)) -> i32;
}

extern "C" fn my_dtor(_: *mut c_void) {}

pub fn main() {
    unsafe {
        let mut i: u8 = 32;
        pthread_key_create(&raw mut i as *mut c_void, my_dtor);
        //~^ ERROR: memory access failed
    }
}
