//! Regression test for https://github.com/rust-lang/rust/issues/29485
//! Exposed an LLVM bug that caused rustc to panic.
//@ run-pass
#![allow(unused_attributes)]
//@ aux-build:mut-ref-write-visible-after-unwind.rs
//@ needs-unwind
//@ needs-threads
//@ ignore-backends: gcc

#[feature(recover)]

extern crate mut_ref_write_visible_after_unwind as lib;

fn main() {
    let _ = std::thread::spawn(move || {
        lib::f(&mut lib::X(0), g);
    }).join();
}

fn g() {
    panic!();
}
