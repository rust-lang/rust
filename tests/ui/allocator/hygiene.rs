//@ run-pass
//@ no-prefer-dynamic
//@ aux-build:custom.rs
//@ aux-build:helper.rs

#![allow(nonstandard_style)]

extern crate custom;
extern crate helper;

use custom::A;
use std::sync::atomic::{AtomicUsize, Ordering};

#[allow(dead_code)]
struct u8;
#[allow(dead_code)]
struct usize;
#[allow(dead_code)]
static arg0: () = ();

#[global_allocator]
pub static GLOBAL: A = A(AtomicUsize::new(0));

fn main() {
    let n = GLOBAL.0.load(Ordering::SeqCst);
    let s = Box::new(0);
    helper::work_with(&s);
    assert_eq!(GLOBAL.0.load(Ordering::SeqCst), n + 1);
    drop(s);
    assert_eq!(GLOBAL.0.load(Ordering::SeqCst), n + 2);
}
