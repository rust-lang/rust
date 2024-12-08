//@ check-pass

#![deny(dangling_pointers_from_temporaries)]

// The original code example comes from bindgen-produced code for emacs.
// Hence the name of the test.
// https://github.com/rust-lang/rust/pull/128985#issuecomment-2338951363

use std::ffi::{c_char, CString};

fn read(ptr: *const c_char) -> c_char {
    unsafe { ptr.read() }
}

fn main() {
    let fnptr: Option<fn(ptr: *const c_char) -> c_char> = Some(read);
    let x = fnptr.unwrap()(CString::new("foo").unwrap().as_ptr());
    assert_eq!(x as u8, b'f');
}
