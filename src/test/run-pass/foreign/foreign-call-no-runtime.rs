// run-pass
// ignore-emscripten no threads support

#![feature(rustc_private)]

extern crate libc;

use std::mem;
use std::thread;

#[link(name = "rust_test_helpers", kind = "static")]
extern {
    fn rust_dbg_call(cb: extern "C" fn(libc::uintptr_t),
                     data: libc::uintptr_t) -> libc::uintptr_t;
}

pub fn main() {
    unsafe {
        thread::spawn(move|| {
            let i: isize = 100;
            rust_dbg_call(callback_isize, mem::transmute(&i));
        }).join().unwrap();

        thread::spawn(move|| {
            let i: i32 = 100;
            rust_dbg_call(callback_i32, mem::transmute(&i));
        }).join().unwrap();

        thread::spawn(move|| {
            let i: i64 = 100;
            rust_dbg_call(callback_i64, mem::transmute(&i));
        }).join().unwrap();
    }
}

extern fn callback_isize(data: libc::uintptr_t) {
    unsafe {
        let data: *const isize = mem::transmute(data);
        assert_eq!(*data, 100);
    }
}

extern fn callback_i64(data: libc::uintptr_t) {
    unsafe {
        let data: *const i64 = mem::transmute(data);
        assert_eq!(*data, 100);
    }
}

extern fn callback_i32(data: libc::uintptr_t) {
    unsafe {
        let data: *const i32 = mem::transmute(data);
        assert_eq!(*data, 100);
    }
}
