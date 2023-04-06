#![feature(once_cell)]
#![deny(clone_double_ref)]

pub fn clone_on_double_ref() {
    let x = vec![1];
    let y = &&x;
    let z: &Vec<_> = y.clone();
    //~^ ERROR using `clone` on a double-reference, which copies the reference of type `Vec<i32>`

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
    let _ = &encoded.clone();
}

fn main() {}
