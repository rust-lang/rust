//@ignore-target: windows # No pthreads on Windows
//! Test that pthread_key destructors are run in the right order.
//! Note that these are *not* used by actual `thread_local!` on Linux! Those use
//! `destructors::register` from the stdlib instead. In Miri this ends up hitting
//! the fallback path in `guard::key::enable`, which uses a *single* pthread_key
//! to manage a thread-local list of dtors to call.

// FIXME(static_mut_refs): Do not allow `static_mut_refs` lint
#![allow(static_mut_refs)]

use std::{mem, ptr};

pub type Key = libc::pthread_key_t;

static mut RECORD: usize = 0;
static mut KEYS: [Key; 2] = [0; 2];
static mut GLOBALS: [u64; 2] = [1, 0];

static mut CANARY: *mut u64 = ptr::null_mut(); // this serves as a canary: if TLS dtors are not run properly, this will not get deallocated, making the test fail.

pub unsafe fn create(dtor: Option<unsafe extern "C" fn(*mut u8)>) -> Key {
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
    unsafe { RECORD = RECORD * 10 + r };
}

unsafe extern "C" fn dtor(ptr: *mut u64) {
    assert!(CANARY != ptr::null_mut()); // make sure we do not get run too often
    let val = *ptr;

    let which_key =
        GLOBALS.iter().position(|global| global as *const _ == ptr).expect("Should find my global");
    record(which_key);

    if val > 0 {
        *ptr = val - 1;
        set(KEYS[which_key], ptr as *mut _);
    }

    // Check if the records matches what we expect. If yes, clear the canary.
    // If the record is wrong, the canary will never get cleared, leading to a leak -> test fails.
    // If the record is incomplete (i.e., more dtor calls happen), the check at the beginning of this function will fail -> test fails.
    // The correct sequence is: First key 0, then key 1, then key 0.
    // Note that this relies on dtor order, which is not specified by POSIX, but seems to be
    // consistent between Miri and Linux currently (as of Aug 2022).
    if RECORD == 0_1_0 {
        drop(Box::from_raw(CANARY));
        CANARY = ptr::null_mut();
    }
}

fn main() {
    unsafe {
        create(None); // check that the no-dtor case works

        // Initialize the keys we use to check destructor ordering
        for (key, global) in KEYS.iter_mut().zip(GLOBALS.iter_mut()) {
            *key = create(Some(mem::transmute(dtor as unsafe extern "C" fn(*mut u64))));
            set(*key, global as *mut _ as *mut u8);
        }

        // Initialize canary
        CANARY = Box::into_raw(Box::new(0u64));
    }
}
