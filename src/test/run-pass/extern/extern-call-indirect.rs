// run-pass
// ignore-wasm32-bare no libc to test ffi with

#![feature(rustc_private)]

extern crate libc;

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
        fact(data - 1) * data
    }
}

fn fact(n: libc::uintptr_t) -> libc::uintptr_t {
    unsafe {
        println!("n = {}", n);
        rustrt::rust_dbg_call(cb, n)
    }
}

pub fn main() {
    let result = fact(10);
    println!("result = {}", result);
    assert_eq!(result, 3628800);
}
