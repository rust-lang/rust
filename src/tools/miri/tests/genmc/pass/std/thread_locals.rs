//@compile-flags: -Zmiri-ignore-leaks -Zmiri-genmc -Zmiri-disable-stacked-borrows
//@normalize-stderr-test: "\n *= note: inside `std::.*" -> ""

use std::alloc::{Layout, alloc};
use std::cell::Cell;
use std::sync::atomic::AtomicPtr;
use std::sync::atomic::Ordering::*;

static X: AtomicPtr<u64> = AtomicPtr::new(std::ptr::null_mut());

thread_local! {
    static R: Cell<*mut u64> = Cell::new(std::ptr::null_mut());
}

pub unsafe fn malloc() -> *mut u64 {
    alloc(Layout::new::<u64>()) as *mut u64
}

fn main() {
    let handles = [
        std::thread::spawn(|| {
            R.set(unsafe { malloc() });
            let r_ptr = R.get();
            let _ = X.compare_exchange(std::ptr::null_mut(), r_ptr, SeqCst, SeqCst);
        }),
        std::thread::spawn(|| {
            R.set(unsafe { malloc() });
        }),
        std::thread::spawn(|| {
            R.set(unsafe { malloc() });
            let r_ptr = R.get();
            let _ = X.compare_exchange(std::ptr::null_mut(), r_ptr, SeqCst, SeqCst);
        }),
    ];
    handles.into_iter().for_each(|handle| handle.join().unwrap());
}
