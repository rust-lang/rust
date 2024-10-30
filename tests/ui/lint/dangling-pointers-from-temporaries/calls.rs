#![deny(dangling_pointers_from_temporaries)]

use std::ffi::{c_char, CString};

fn cstring() -> CString {
    CString::new("hello").unwrap()
}

fn consume(ptr: *const c_char) {
    let c = unsafe { ptr.read() };
    dbg!(c);
}

// None of these should trigger the lint.
fn ok() {
    consume(cstring().as_ptr());
    consume({ cstring() }.as_ptr());
    consume({ cstring().as_ptr() });
    consume(cstring().as_ptr().cast());
    consume({ cstring() }.as_ptr().cast());
    consume({ cstring().as_ptr() }.cast());
}

// All of these should trigger the lint.
fn not_ok() {
    {
        let ptr = cstring().as_ptr();
        //~^ ERROR a dangling pointer will be produced because the temporary `CString` will be dropped
        consume(ptr);
    }
    consume({
        let ptr = cstring().as_ptr();
        //~^ ERROR a dangling pointer will be produced because the temporary `CString` will be dropped
        ptr
    });
    consume({
        let s = cstring();
        s.as_ptr()
        //^ FIXME: should error
    });
    let _ptr: *const u8 = cstring().as_ptr().cast();
    //~^ ERROR a dangling pointer will be produced because the temporary `CString` will be dropped
    let _ptr: *const u8 = { cstring() }.as_ptr().cast();
    //~^ ERROR a dangling pointer will be produced because the temporary `CString` will be dropped
    let _ptr: *const u8 = { cstring().as_ptr() }.cast();
    //~^ ERROR a dangling pointer will be produced because the temporary `CString` will be dropped
}

fn main() {
    ok();
    not_ok();
}
