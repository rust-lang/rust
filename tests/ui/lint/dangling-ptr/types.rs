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
    declval::<CString>().as_ptr(); //~ ERROR [dangling_pointers_from_temporaries]
    declval::<String>().as_ptr(); //~ ERROR [dangling_pointers_from_temporaries]
    declval::<Vec<u8>>().as_ptr(); //~ ERROR [dangling_pointers_from_temporaries]
    declval::<Box<CString>>().as_ptr(); //~ ERROR [dangling_pointers_from_temporaries]
    declval::<Box<[u8]>>().as_ptr(); //~ ERROR [dangling_pointers_from_temporaries]
    declval::<Box<str>>().as_ptr(); //~ ERROR [dangling_pointers_from_temporaries]
    declval::<Box<CStr>>().as_ptr(); //~ ERROR [dangling_pointers_from_temporaries]
    declval::<[u8; 10]>().as_ptr(); //~ ERROR [dangling_pointers_from_temporaries]
    declval::<Box<[u8; 10]>>().as_ptr(); //~ ERROR [dangling_pointers_from_temporaries]
    declval::<Box<Vec<u8>>>().as_ptr(); //~ ERROR [dangling_pointers_from_temporaries]
    declval::<Box<String>>().as_ptr(); //~ ERROR [dangling_pointers_from_temporaries]
    declval::<Box<Box<Box<Box<[u8]>>>>>().as_ptr(); //~ ERROR [dangling_pointers_from_temporaries]
    declval::<Cell<u8>>().as_ptr(); //~ ERROR [dangling_pointers_from_temporaries]
    declval::<MaybeUninit<u8>>().as_ptr(); //~ ERROR [dangling_pointers_from_temporaries]
    declval::<Vec<AsPtrFake>>().as_ptr(); //~ ERROR [dangling_pointers_from_temporaries]
    declval::<Box<AsPtrFake>>().as_ptr();
    declval::<AsPtrFake>().as_ptr();
}
