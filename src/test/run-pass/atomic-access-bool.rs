#![allow(stable_features)]
#![feature(atomic_access)]
use std::sync::atomic::{AtomicBool, ATOMIC_BOOL_INIT};
use std::sync::atomic::Ordering::*;

static mut ATOMIC: AtomicBool = ATOMIC_BOOL_INIT;

fn main() {
    unsafe {
        assert_eq!(*ATOMIC.get_mut(), false);
        ATOMIC.store(true, SeqCst);
        assert_eq!(*ATOMIC.get_mut(), true);
        ATOMIC.fetch_or(false, SeqCst);
        assert_eq!(*ATOMIC.get_mut(), true);
        ATOMIC.fetch_and(false, SeqCst);
        assert_eq!(*ATOMIC.get_mut(), false);
        ATOMIC.fetch_nand(true, SeqCst);
        assert_eq!(*ATOMIC.get_mut(), true);
        ATOMIC.fetch_xor(true, SeqCst);
        assert_eq!(*ATOMIC.get_mut(), false);
    }
}
