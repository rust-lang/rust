// Tests that C++ exceptions can unwind through Rust code run destructors and
// are caught by catch_unwind. Also tests that Rust panics can unwind through
// C++ code.

// For linking libstdc++ on MinGW
#![cfg_attr(all(windows, target_env = "gnu"), feature(static_nobundle))]
#![feature(unwind_attributes)]

use std::panic::{catch_unwind, AssertUnwindSafe};

struct DropCheck<'a>(&'a mut bool);
impl<'a> Drop for DropCheck<'a> {
    fn drop(&mut self) {
        println!("DropCheck::drop");
        *self.0 = true;
    }
}

extern "C" {
    fn test_cxx_exception();

    #[unwind(allowed)]
    fn cxx_catch_callback(cb: extern "C" fn(), ok: *mut bool);
}

#[no_mangle]
#[unwind(allowed)]
extern "C" fn rust_catch_callback(cb: extern "C" fn(), rust_ok: &mut bool) {
    let _drop = DropCheck(rust_ok);
    cb();
    unreachable!("should have unwound instead of returned");
}

fn test_rust_panic() {
    #[unwind(allowed)]
    extern "C" fn callback() {
        println!("throwing rust panic");
        panic!(1234i32);
    }

    let mut dropped = false;
    let mut cxx_ok = false;
    let caught_unwind = catch_unwind(AssertUnwindSafe(|| {
        let _drop = DropCheck(&mut dropped);
        unsafe {
            cxx_catch_callback(callback, &mut cxx_ok);
        }
        unreachable!("should have unwound instead of returned");
    }));
    println!("caught rust panic");
    assert!(dropped);
    assert!(caught_unwind.is_err());
    let panic_obj = caught_unwind.unwrap_err();
    let panic_int = *panic_obj.downcast_ref::<i32>().unwrap();
    assert_eq!(panic_int, 1234);
    assert!(cxx_ok);
}

fn main() {
    unsafe { test_cxx_exception() };
    test_rust_panic();
}
