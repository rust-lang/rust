//@ run-pass
//@ only-linux
//@ only-gnu
//@ compile-flags: -Zexport-executable-symbols
//@ edition: 2024

// Regression test for <https://github.com/rust-lang/rust/issues/101610>.

#![feature(rustc_private)]

extern crate libc;

#[unsafe(no_mangle)]
fn hack() -> u64 {
    998244353
}

fn main() {
    unsafe {
        let handle = libc::dlopen(std::ptr::null(), libc::RTLD_NOW);
        let ptr = libc::dlsym(handle, c"hack".as_ptr());
        let ptr: Option<unsafe fn() -> u64> = std::mem::transmute(ptr);
        if let Some(f) = ptr {
            assert!(f() == 998244353);
            println!("symbol `hack` is found successfully");
        } else {
            panic!("symbol `hack` is not found");
        }
    }
}
