//! Regression test for <https://github.com/rust-lang/rust/issues/76387>
//! Tests that LLVM doesn't miscompile this
//! See upstream fix: <https://reviews.llvm.org/D88529>.

//@ compile-flags: -C opt-level=3
//@ aux-build: llvm-miscompile-MarkValue-MaybeLive.rs
//@ run-pass

extern crate llvm_miscompile_MarkValue_MaybeLive;

use llvm_miscompile_MarkValue_MaybeLive::FatPtr;

fn print(data: &[u8]) {
    println!("{:#?}", data);
}

fn main() {
    let ptr = FatPtr::new(20);
    let data = unsafe { std::slice::from_raw_parts(ptr.as_ptr(), ptr.len()) };

    print(data);
}
