#![deny(dangling_pointers_from_temporaries)]
#![feature(sync_unsafe_cell)]

use std::cell::{Cell, SyncUnsafeCell, UnsafeCell};
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
    //~^ ERROR a dangling pointer will be produced because the temporary `CString` will be dropped
    declval::<String>().as_ptr();
    //~^ ERROR a dangling pointer will be produced because the temporary `String` will be dropped
    declval::<Vec<u8>>().as_ptr();
    //~^ ERROR a dangling pointer will be produced because the temporary `Vec<u8>` will be dropped
    declval::<Box<CString>>().as_ptr();
    //~^ ERROR a dangling pointer will be produced because the temporary `Box<CString>` will be dropped
    declval::<Box<[u8]>>().as_ptr();
    //~^ ERROR a dangling pointer will be produced because the temporary `Box<[u8]>` will be dropped
    declval::<Box<str>>().as_ptr();
    //~^ ERROR a dangling pointer will be produced because the temporary `Box<str>` will be dropped
    declval::<Box<CStr>>().as_ptr();
    //~^ ERROR a dangling pointer will be produced because the temporary `Box<CStr>` will be dropped
    declval::<[u8; 10]>().as_ptr();
    //~^ ERROR a dangling pointer will be produced because the temporary `[u8; 10]` will be dropped
    declval::<Box<[u8; 10]>>().as_ptr();
    //~^ ERROR a dangling pointer will be produced because the temporary `Box<[u8; 10]>` will be dropped
    declval::<Box<Vec<u8>>>().as_ptr();
    //~^ ERROR a dangling pointer will be produced because the temporary `Box<Vec<u8>>` will be dropped
    declval::<Box<String>>().as_ptr();
    //~^ ERROR a dangling pointer will be produced because the temporary `Box<String>` will be dropped
    declval::<Box<Box<Box<Box<[u8]>>>>>().as_ptr();
    //~^ ERROR a dangling pointer will be produced because the temporary `Box<Box<Box<Box<[u8]>>>>` will be dropped
    declval::<Cell<u8>>().as_ptr();
    //~^ ERROR a dangling pointer will be produced because the temporary `Cell<u8>` will be dropped
    declval::<MaybeUninit<u8>>().as_ptr();
    //~^ ERROR a dangling pointer will be produced because the temporary `MaybeUninit<u8>` will be dropped
    declval::<Vec<AsPtrFake>>().as_ptr();
    //~^ ERROR a dangling pointer will be produced because the temporary `Vec<AsPtrFake>` will be dropped
    declval::<UnsafeCell<u8>>().get();
    //~^ ERROR a dangling pointer will be produced because the temporary `UnsafeCell<u8>` will be dropped
    declval::<SyncUnsafeCell<u8>>().get();
    //~^ ERROR a dangling pointer will be produced because the temporary `SyncUnsafeCell<u8>` will be dropped
    declval::<Box<AsPtrFake>>().as_ptr();
    declval::<AsPtrFake>().as_ptr();
}
