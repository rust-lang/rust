#![deny(dangling_pointers_from_temporaries)]

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
    declval::<CString>().as_ptr();
    //~^ ERROR getting a pointer from a temporary `CString` will result in a dangling pointer
    declval::<String>().as_ptr();
    //~^ ERROR getting a pointer from a temporary `String` will result in a dangling pointer
    declval::<Vec<u8>>().as_ptr();
    //~^ ERROR getting a pointer from a temporary `Vec<u8>` will result in a dangling pointer
    declval::<Box<CString>>().as_ptr();
    //~^ ERROR getting a pointer from a temporary `Box<CString>` will result in a dangling pointer
    declval::<Box<[u8]>>().as_ptr();
    //~^ ERROR getting a pointer from a temporary `Box<[u8]>` will result in a dangling pointer
    declval::<Box<str>>().as_ptr();
    //~^ ERROR getting a pointer from a temporary `Box<str>` will result in a dangling pointer
    declval::<Box<CStr>>().as_ptr();
    //~^ ERROR getting a pointer from a temporary `Box<CStr>` will result in a dangling pointer
    declval::<[u8; 10]>().as_ptr();
    //~^ ERROR getting a pointer from a temporary `[u8; 10]` will result in a dangling pointer
    declval::<Box<[u8; 10]>>().as_ptr();
    //~^ ERROR getting a pointer from a temporary `Box<[u8; 10]>` will result in a dangling pointer
    declval::<Box<Vec<u8>>>().as_ptr();
    //~^ ERROR getting a pointer from a temporary `Box<Vec<u8>>` will result in a dangling pointer
    declval::<Box<String>>().as_ptr();
    //~^ ERROR getting a pointer from a temporary `Box<String>` will result in a dangling pointer
    declval::<Box<Box<Box<Box<[u8]>>>>>().as_ptr();
    //~^ ERROR getting a pointer from a temporary `Box<Box<Box<Box<[u8]>>>>` will result in a dangling pointer
    declval::<Cell<u8>>().as_ptr();
    //~^ ERROR getting a pointer from a temporary `Cell<u8>` will result in a dangling pointer
    declval::<MaybeUninit<u8>>().as_ptr();
    //~^ ERROR getting a pointer from a temporary `MaybeUninit<u8>` will result in a dangling pointer
    declval::<Vec<AsPtrFake>>().as_ptr();
    //~^ ERROR getting a pointer from a temporary `Vec<AsPtrFake>` will result in a dangling pointer
    declval::<Box<AsPtrFake>>().as_ptr();
    declval::<AsPtrFake>().as_ptr();
}
