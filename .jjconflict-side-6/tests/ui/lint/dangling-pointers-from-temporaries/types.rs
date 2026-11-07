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
    //~^ ERROR dangling pointer because temporary `CString`
    declval::<String>().as_ptr();
    //~^ ERROR dangling pointer because temporary `String`
    declval::<Vec<u8>>().as_ptr();
    //~^ ERROR dangling pointer because temporary `Vec<u8>`
    declval::<Box<CString>>().as_ptr();
    //~^ ERROR dangling pointer because temporary `Box<CString>`
    declval::<Box<[u8]>>().as_ptr();
    //~^ ERROR dangling pointer because temporary `Box<[u8]>`
    declval::<Box<str>>().as_ptr();
    //~^ ERROR dangling pointer because temporary `Box<str>`
    declval::<Box<CStr>>().as_ptr();
    //~^ ERROR dangling pointer because temporary `Box<CStr>`
    declval::<[u8; 10]>().as_ptr();
    //~^ ERROR dangling pointer because temporary `[u8; 10]`
    declval::<Box<[u8; 10]>>().as_ptr();
    //~^ ERROR dangling pointer because temporary `Box<[u8; 10]>`
    declval::<Box<Vec<u8>>>().as_ptr();
    //~^ ERROR dangling pointer because temporary `Box<Vec<u8>>`
    declval::<Box<String>>().as_ptr();
    //~^ ERROR dangling pointer because temporary `Box<String>`
    declval::<Box<Box<Box<Box<[u8]>>>>>().as_ptr();
    //~^ ERROR dangling pointer because temporary `Box<Box<Box<Box<[u8]>>>>`
    declval::<Cell<u8>>().as_ptr();
    //~^ ERROR dangling pointer because temporary `Cell<u8>`
    declval::<MaybeUninit<u8>>().as_ptr();
    //~^ ERROR dangling pointer because temporary `MaybeUninit<u8>`
    declval::<Vec<AsPtrFake>>().as_ptr();
    //~^ ERROR dangling pointer because temporary `Vec<AsPtrFake>`
    declval::<UnsafeCell<u8>>().get();
    //~^ ERROR dangling pointer because temporary `UnsafeCell<u8>`
    declval::<SyncUnsafeCell<u8>>().get();
    //~^ ERROR dangling pointer because temporary `SyncUnsafeCell<u8>`
    declval::<Box<AsPtrFake>>().as_ptr();
    declval::<AsPtrFake>().as_ptr();
}
