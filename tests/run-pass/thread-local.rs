// ignore-windows: No libc on Windows

#![feature(rustc_private)]
extern crate libc;

use std::mem;

pub type Key = libc::pthread_key_t;

static mut RECORD : usize = 0;
static mut KEYS : [Key; 2] = [0; 2];
static mut GLOBALS : [u64; 2] = [1, 0];

static mut CANNARY : *mut u64 = 0 as *mut _; // this serves as a cannary: if TLS dtors are not run properly, this will not get deallocated, making the test fail.

pub unsafe fn create(dtor: Option<unsafe extern fn(*mut u8)>) -> Key {
    let mut key = 0;
    assert_eq!(libc::pthread_key_create(&mut key, mem::transmute(dtor)), 0);
    key
}

pub unsafe fn set(key: Key, value: *mut u8) {
    let r = libc::pthread_setspecific(key, value as *mut _);
    assert_eq!(r, 0);
}

pub fn record(r: usize) {
    assert!(r < 10);
    unsafe { RECORD = RECORD*10 + r };
}

unsafe extern fn dtor(ptr: *mut u64) {
    assert!(CANNARY != 0 as *mut _); // make sure we do not get run too often
    let val = *ptr;

    let which_key = GLOBALS.iter().position(|global| global as *const _ == ptr).expect("Should find my global");
    record(which_key);

    if val > 0 {
        *ptr = val-1;
        set(KEYS[which_key], ptr as *mut _);
    }

    // Check if the records matches what we expect. If yes, clear the cannary.
    // If the record is wrong, the cannary will never get cleared, leading to a leak -> test fails.
    // If the record is incomplete (i.e., more dtor calls happen), the check at the beginning of this function will fail -> test fails.
    // The correct sequence is: First key 0, then key 1, then key 0.
    if RECORD == 0_1_0 {
        drop(Box::from_raw(CANNARY));
        CANNARY = 0 as *mut _;
    }
}

fn main() {
    unsafe {
        create(None); // check that the no-dtor case works

        // Initialize the keys we use to check destructor ordering
        for (key, global) in KEYS.iter_mut().zip(GLOBALS.iter_mut()) {
            *key = create(Some(mem::transmute(dtor as unsafe extern fn(*mut u64))));
            set(*key, global as *mut _ as *mut u8);
        }

        // Initialize cannary
        CANNARY = Box::into_raw(Box::new(0u64));
    }
}
