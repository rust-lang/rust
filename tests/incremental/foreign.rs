// Test what happens we save incremental compilation state that makes
// use of foreign items. This used to ICE (#34991).
//@ revisions: rpass1

use std::ffi::CString;

mod mlibc {
    extern "C" {
        // strlen is provided either by an external library or compiler-builtins as a fallback
        pub fn strlen(x: *const std::ffi::c_char) -> usize;
    }
}

fn strlen(s: String) -> usize {
    let c = CString::new(s).unwrap();
    unsafe { mlibc::strlen(c.as_ptr()) }
}

pub fn main() {
    assert_eq!(strlen("1024".to_string()), strlen("2048".to_string()));
}
