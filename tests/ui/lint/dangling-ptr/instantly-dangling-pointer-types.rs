#![deny(instantly_dangling_pointer)]

use std::cell::Cell;
use std::ffi::{CStr, CString};
use std::mem::MaybeUninit;

struct AsPtrFake;

impl AsPtrFake {
    fn as_ptr(&self) -> *const () {
        std::ptr::null()
    }
}

fn declval<T>() -> T {
    loop {}
}

fn main() {
    declval::<CString>().as_ptr(); //~ ERROR [instantly_dangling_pointer]
    declval::<String>().as_ptr(); //~ ERROR [instantly_dangling_pointer]
    declval::<Vec<u8>>().as_ptr(); //~ ERROR [instantly_dangling_pointer]
    declval::<Box<CString>>().as_ptr(); //~ ERROR [instantly_dangling_pointer]
    declval::<Box<[u8]>>().as_ptr(); //~ ERROR [instantly_dangling_pointer]
    declval::<Box<str>>().as_ptr(); //~ ERROR [instantly_dangling_pointer]
    declval::<Box<CStr>>().as_ptr(); //~ ERROR [instantly_dangling_pointer]
    declval::<[u8; 10]>().as_ptr(); //~ ERROR [instantly_dangling_pointer]
    declval::<Box<[u8; 10]>>().as_ptr(); //~ ERROR [instantly_dangling_pointer]
    declval::<Box<Vec<u8>>>().as_ptr(); //~ ERROR [instantly_dangling_pointer]
    declval::<Box<String>>().as_ptr(); //~ ERROR [instantly_dangling_pointer]
    declval::<Box<Box<Box<Box<[u8]>>>>>().as_ptr(); //~ ERROR [instantly_dangling_pointer]
    declval::<Cell<u8>>().as_ptr(); //~ ERROR [instantly_dangling_pointer]
    declval::<MaybeUninit<u8>>().as_ptr(); //~ ERROR [instantly_dangling_pointer]
    declval::<Vec<AsPtrFake>>().as_ptr(); //~ ERROR [instantly_dangling_pointer]
    declval::<Box<AsPtrFake>>().as_ptr();
    declval::<AsPtrFake>().as_ptr();
}
