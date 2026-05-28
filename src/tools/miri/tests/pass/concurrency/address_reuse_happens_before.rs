//! Regression test for <https://github.com/rust-lang/miri/issues/3450>:
//! When the address gets reused, there should be a happens-before relation.
//@compile-flags: -Zmiri-address-reuse-cross-thread-rate=1.0
#![feature(sync_unsafe_cell)]

use std::cell::SyncUnsafeCell;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering::Relaxed;
use std::thread;

static ADDR: AtomicUsize = AtomicUsize::new(0);
static VAL: SyncUnsafeCell<i32> = SyncUnsafeCell::new(0);

fn addr() -> usize {
    let alloc = Box::new(42);
    <*const i32>::addr(&*alloc)
}

fn thread1() {
    unsafe {
        VAL.get().write(42);
    }
    let alloc = addr();
    ADDR.store(alloc, Relaxed);
}

fn thread2() -> bool {
    // We try to get an allocation at the same address as the global `ADDR`. If we fail too often,
    // just bail. `main` will try again with a different allocation.
    for _ in 0..16 {
        let alloc = addr();
        let addr = ADDR.load(Relaxed);
        if alloc == addr {
            // We got a reuse!
            // If the new allocation is at the same address as the old one, there must be a
            // happens-before relationship between them. Therefore, we can read VAL without racing
            // and must observe the write above.
            let val = unsafe { VAL.get().read() };
            assert_eq!(val, 42);
            return true;
        }
    }

    false
}

fn main() {
    let mut success = false;
    while !success {
        let t1 = thread::spawn(thread1);
        let t2 = thread::spawn(thread2);
        t1.join().unwrap();
        success = t2.join().unwrap();

        // Reset everything.
        ADDR.store(0, Relaxed);
        unsafe {
            VAL.get().write(0);
        }
    }
}
