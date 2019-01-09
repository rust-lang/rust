// run-pass
#![allow(unused_must_use)]
// ignore-emscripten no threads support

#![feature(rustc_private)]

extern crate libc;
use std::thread;

mod rustrt {
    extern crate libc;

    #[link(name = "rust_test_helpers", kind = "static")]
    extern {
        pub fn rust_dbg_call(cb: extern "C" fn(libc::uintptr_t) -> libc::uintptr_t,
                             data: libc::uintptr_t)
                             -> libc::uintptr_t;
    }
}

extern fn cb(data: libc::uintptr_t) -> libc::uintptr_t {
    if data == 1 {
        data
    } else {
        count(data - 1) + 1
    }
}

fn count(n: libc::uintptr_t) -> libc::uintptr_t {
    unsafe {
        println!("n = {}", n);
        rustrt::rust_dbg_call(cb, n)
    }
}

pub fn main() {
    // Make sure we're on a thread with small Rust stacks (main currently
    // has a large stack)
    thread::spawn(move|| {
        let result = count(1000);
        println!("result = {}", result);
        assert_eq!(result, 1000);
    }).join();
}
