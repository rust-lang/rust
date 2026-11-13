//! Regression test for https://github.com/rust-lang/rust/issues/114665.

//@ build-pass
//@ compile-flags: -Zmir-opt-level=0

#![feature(unboxed_closures)]
fn main() {
    unsafe { std::mem::transmute::<usize, extern "rust-call" fn()>(5); }
}
