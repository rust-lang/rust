//! Regression test for std weakening a strong `gettid` reference under fat LTO (#154439).

//@ run-pass
//@ only-linux
//@ aux-build: strong-gettid-ref.rs
//@ no-prefer-dynamic
//@ compile-flags: -Copt-level=3 -Clto=fat -Ctarget-feature=+crt-static

extern crate strong_gettid_ref;

fn main() {
    // The main thread's TID equals its PID.
    assert_eq!(strong_gettid_ref::tid() as u32, std::process::id());
}
