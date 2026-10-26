//@ignore-target: windows # No pthreads on Windows
//@run-native
//! Test that pthread_key destructors are run in the right order.

// We don't keep our refs any longer than necessary or allowed.
#![allow(static_mut_refs)]

use std::mem;

pub type Key = libc::pthread_key_t;

static mut KEYS: [Key; 2] = [0; 2];
static mut GLOBALS: [u8; 2] = [1, 0];

pub unsafe fn create(dtor: Option<unsafe extern "C" fn(*mut u8)>) -> Key {
    let mut key = 0;
    assert_eq!(libc::pthread_key_create(&mut key, mem::transmute(dtor)), 0);
    key
}

pub unsafe fn set(key: Key, value: *mut u8) {
    let r = libc::pthread_setspecific(key, value as *mut _);
    assert_eq!(r, 0);
}

unsafe extern "C" fn dtor(ptr: *mut u8) {
    let val = *ptr;

    let which_key =
        GLOBALS.iter().position(|global| global as *const _ == ptr).expect("Should find my global");
    eprintln!("dropping key {which_key}");

    if val > 0 {
        *ptr = val - 1;
        set(KEYS[which_key], ptr as *mut _);
    }
}

fn main() {
    let t = std::thread::spawn(|| unsafe {
        create(None); // check that the no-dtor case works

        // Initialize the keys we use to check destructor ordering
        for (key, global) in KEYS.iter_mut().zip(GLOBALS.iter_mut()) {
            *key = create(Some(dtor));
            set(*key, global as *mut _ as *mut u8);
        }
    });
    t.join().unwrap()
}
