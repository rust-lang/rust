//@ run-pass
//@ ignore-sgx no libc

use std::ffi::CString;

mod mlibc {
    use std::ffi::{c_char, c_long, c_longlong};

    extern "C" {
        pub fn atol(x: *const c_char) -> c_long;
        pub fn atoll(x: *const c_char) -> c_longlong;
    }
}

fn atol(s: String) -> isize {
    let c = CString::new(s).unwrap();
    unsafe { mlibc::atol(c.as_ptr()) as isize }
}

fn atoll(s: String) -> i64 {
    let c = CString::new(s).unwrap();
    unsafe { mlibc::atoll(c.as_ptr()) as i64 }
}

pub fn main() {
    assert_eq!(atol("1024".to_string()) * 10, atol("10240".to_string()));
    assert_eq!(
        (atoll("11111111111111111".to_string()) * 10),
        atoll("111111111111111110".to_string())
    );
}
