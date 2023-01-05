// Tests that C++ exceptions can unwind through Rust code run destructors and
// are caught by catch_unwind. Also tests that Rust panics can unwind through
// C++ code.

#![feature(c_unwind)]

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
}

extern "C-unwind" {
    fn cxx_catch_callback(cb: extern "C-unwind" fn(), ok: *mut bool);
}

#[no_mangle]
extern "C-unwind" fn rust_catch_callback(cb: extern "C-unwind" fn(), rust_ok: &mut bool) {
    let _drop = DropCheck(rust_ok);
    cb();
    unreachable!("should have unwound instead of returned");
}

fn test_rust_panic() {
    extern "C-unwind" fn callback() {
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
