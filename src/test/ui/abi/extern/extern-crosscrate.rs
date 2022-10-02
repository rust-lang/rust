// run-pass
// aux-build:extern-crosscrate-source.rs
// ignore-wasm32-bare no libc to test ffi with

#![feature(rustc_private)]

extern crate externcallback;
extern crate libc;

fn fact(n: libc::uintptr_t) -> libc::uintptr_t {
    unsafe {
        println!("n = {}", n);
        externcallback::rustrt::rust_dbg_call(externcallback::cb, n)
    }
}

pub fn main() {
    let result = fact(10);
    println!("result = {}", result);
    assert_eq!(result, 3628800);
}
