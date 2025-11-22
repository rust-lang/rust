//@ check-pass
//@ run-rustfix

#![allow(deprecated)]
#![allow(dead_code)]
#![feature(atomic_try_update)]

use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicU32, Ordering};

fn atomic_bool() {
    const A: AtomicBool = AtomicBool::new(false);

    let _a = A.store(true, Ordering::SeqCst);
    //~^ WARN mutation of an interior mutable `const` item with call to `store`

    let _a = A.swap(true, Ordering::SeqCst);
    //~^ WARN mutation of an interior mutable `const` item with call to `swap`

    let _a = A.compare_and_swap(false, true, Ordering::SeqCst);
    //~^ WARN mutation of an interior mutable `const` item with call to `compare_and_swap`

    let _a = A.compare_exchange(false, true, Ordering::SeqCst, Ordering::Relaxed);
    //~^ WARN mutation of an interior mutable `const` item with call to `compare_exchange`

    let _a = A.compare_exchange_weak(false, true, Ordering::SeqCst, Ordering::Relaxed);
    //~^ WARN mutation of an interior mutable `const` item with call to `compare_exchange_weak`

    let _a = A.fetch_and(true, Ordering::SeqCst);
    //~^ WARN mutation of an interior mutable `const` item with call to `fetch_and`

    let _a = A.fetch_nand(true, Ordering::SeqCst);
    //~^ WARN mutation of an interior mutable `const` item with call to `fetch_nand`

    let _a = A.fetch_or(true, Ordering::SeqCst);
    //~^ WARN mutation of an interior mutable `const` item with call to `fetch_or`

    let _a = A.fetch_xor(true, Ordering::SeqCst);
    //~^ WARN mutation of an interior mutable `const` item with call to `fetch_xor`

    let _a = A.fetch_not(Ordering::SeqCst);
    //~^ WARN mutation of an interior mutable `const` item with call to `fetch_not`

    let _a = A.fetch_update(Ordering::SeqCst, Ordering::Relaxed, |_| Some(true));
    //~^ WARN mutation of an interior mutable `const` item with call to `fetch_update`

    let _a = A.try_update(Ordering::SeqCst, Ordering::Relaxed, |_| Some(false));
    //~^ WARN mutation of an interior mutable `const` item with call to `try_update`

    let _a = A.update(Ordering::SeqCst, Ordering::Relaxed, |_| true);
    //~^ WARN mutation of an interior mutable `const` item with call to `update`
}

fn atomic_ptr() {
    const A: AtomicPtr<i32> = AtomicPtr::new(std::ptr::null_mut());

    let _a = A.store(std::ptr::null_mut(), Ordering::SeqCst);
    //~^ WARN mutation of an interior mutable `const` item with call to `store`

    let _a = A.swap(std::ptr::null_mut(), Ordering::SeqCst);
    //~^ WARN mutation of an interior mutable `const` item with call to `swap`

    let _a = A.compare_and_swap(std::ptr::null_mut(), std::ptr::null_mut(), Ordering::SeqCst);
    //~^ WARN mutation of an interior mutable `const` item with call to `compare_and_swap`

    let _a = A.compare_exchange(
        //~^ WARN mutation of an interior mutable `const` item with call to `compare_exchange`
        std::ptr::null_mut(),
        std::ptr::null_mut(),
        Ordering::SeqCst,
        Ordering::Relaxed,
    );

    let _a = A.compare_exchange_weak(
        //~^ WARN mutation of an interior mutable `const` item with call to `compare_exchange_weak`
        std::ptr::null_mut(),
        std::ptr::null_mut(),
        Ordering::SeqCst,
        Ordering::Relaxed,
    );

    let _a = A.fetch_update(Ordering::SeqCst, Ordering::Relaxed, |_| Some(std::ptr::null_mut()));
    //~^ WARN mutation of an interior mutable `const` item with call to `fetch_update`

    let _a = A.try_update(Ordering::SeqCst, Ordering::Relaxed, |_| Some(std::ptr::null_mut()));
    //~^ WARN mutation of an interior mutable `const` item with call to `try_update`

    let _a = A.update(Ordering::SeqCst, Ordering::Relaxed, |_| std::ptr::null_mut());
    //~^ WARN mutation of an interior mutable `const` item with call to `update`

    let _a = A.fetch_ptr_add(1, Ordering::SeqCst);
    //~^ WARN mutation of an interior mutable `const` item with call to `fetch_ptr_add`

    let _a = A.fetch_ptr_sub(1, Ordering::SeqCst);
    //~^ WARN mutation of an interior mutable `const` item with call to `fetch_ptr_sub`

    let _a = A.fetch_byte_add(1, Ordering::SeqCst);
    //~^ WARN mutation of an interior mutable `const` item with call to `fetch_byte_add`

    let _a = A.fetch_byte_sub(1, Ordering::SeqCst);
    //~^ WARN mutation of an interior mutable `const` item with call to `fetch_byte_sub`

    let _a = A.fetch_and(1, Ordering::SeqCst);
    //~^ WARN mutation of an interior mutable `const` item with call to `fetch_and`

    let _a = A.fetch_or(1, Ordering::SeqCst);
    //~^ WARN mutation of an interior mutable `const` item with call to `fetch_or`

    let _a = A.fetch_xor(1, Ordering::SeqCst);
    //~^ WARN mutation of an interior mutable `const` item with call to `fetch_xor`
}

fn atomic_u32() {
    const A: AtomicU32 = AtomicU32::new(0);

    let _a = A.store(1, Ordering::SeqCst);
    //~^ WARN mutation of an interior mutable `const` item with call to `store`

    let _a = A.swap(2, Ordering::SeqCst);
    //~^ WARN mutation of an interior mutable `const` item with call to `swap`

    let _a = A.compare_and_swap(2, 3, Ordering::SeqCst);
    //~^ WARN mutation of an interior mutable `const` item with call to `compare_and_swap`

    let _a = A.compare_exchange(3, 4, Ordering::SeqCst, Ordering::Relaxed);
    //~^ WARN mutation of an interior mutable `const` item with call to `compare_exchange`

    let _a = A.compare_exchange_weak(4, 5, Ordering::SeqCst, Ordering::Relaxed);
    //~^ WARN mutation of an interior mutable `const` item with call to `compare_exchange_weak`

    let _a = A.fetch_add(1, Ordering::SeqCst);
    //~^ WARN mutation of an interior mutable `const` item with call to `fetch_add`

    let _a = A.fetch_sub(1, Ordering::SeqCst);
    //~^ WARN mutation of an interior mutable `const` item with call to `fetch_sub`

    let _a = A.fetch_add(2, Ordering::SeqCst);
    //~^ WARN mutation of an interior mutable `const` item with call to `fetch_add`

    let _a = A.fetch_nand(1, Ordering::SeqCst);
    //~^ WARN mutation of an interior mutable `const` item with call to `fetch_nand`

    let _a = A.fetch_or(1, Ordering::SeqCst);
    //~^ WARN mutation of an interior mutable `const` item with call to `fetch_or`

    let _a = A.fetch_xor(1, Ordering::SeqCst);
    //~^ WARN mutation of an interior mutable `const` item with call to `fetch_xor`

    let _a = A.fetch_update(Ordering::SeqCst, Ordering::Relaxed, |_| Some(10));
    //~^ WARN mutation of an interior mutable `const` item with call to `fetch_update`

    let _a = A.try_update(Ordering::SeqCst, Ordering::Relaxed, |_| Some(11));
    //~^ WARN mutation of an interior mutable `const` item with call to `try_update`

    let _a = A.update(Ordering::SeqCst, Ordering::Relaxed, |_| 12);
    //~^ WARN mutation of an interior mutable `const` item with call to `update`

    let _a = A.fetch_max(20, Ordering::SeqCst);
    //~^ WARN mutation of an interior mutable `const` item with call to `fetch_max`

    let _a = A.fetch_min(5, Ordering::SeqCst);
    //~^ WARN mutation of an interior mutable `const` item with call to `fetch_min`
}

fn main() {}
