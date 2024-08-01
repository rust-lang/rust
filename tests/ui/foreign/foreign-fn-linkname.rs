//@ run-pass
//@ ignore-sgx no libc

use std::ffi::CString;

mod mlibc {
    use std::ffi::c_char;

    extern "C" {
        #[link_name = "strlen"]
        pub fn my_strlen(str: *const c_char) -> usize;
    }
}

fn strlen(str: String) -> usize {
    // C string is terminated with a zero
    let s = CString::new(str).unwrap();
    unsafe { mlibc::my_strlen(s.as_ptr()) as usize }
}

pub fn main() {
    let len = strlen("Rust".to_string());
    assert_eq!(len, 4);
}
