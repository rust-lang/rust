#![feature(lazy_cell)]
#![deny(suspicious_double_ref_op, noop_method_call)]

pub fn clone_on_double_ref() {
    let x = vec![1];
    let y = &&x;
    let z: &Vec<_> = y.clone();
    //~^ ERROR using `.clone()` on a double reference, which returns `&Vec<i32>`

    println!("{:p} {:p}", *y, z);
}

use std::sync::LazyLock;

pub static STRS: LazyLock<&str> = LazyLock::new(|| "First");

// https://github.com/rust-lang/rust-clippy/issues/9272
fn rust_clippy_issue_9272() {
    let str = STRS.clone();
    println!("{str}")
}

fn check(mut encoded: &[u8]) {
    let _ = &mut encoded.clone();
    //~^ ERROR call to `.clone()` on a reference in this situation does nothing
    let _ = &encoded.clone();
    //~^ ERROR call to `.clone()` on a reference in this situation does nothing
}

fn main() {}
