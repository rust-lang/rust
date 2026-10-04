// Ensure that env::vars() does not panic if environ is null.
// Regression test for rust-lang/rust#53200
//@ run-pass

#![feature(rustc_private)]

// FIXME: more platforms?
#[cfg(target_os = "linux")]
fn main() {
    extern crate libc;
    unsafe { libc::clearenv(); }
    assert_eq!(std::env::vars().count(), 0);
}

#[cfg(not(target_os = "linux"))]
fn main() {}
