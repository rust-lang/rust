// run-rustfix

#![allow(
    unused,
    clippy::redundant_clone,
    clippy::deref_addrof,
    clippy::no_effect,
    clippy::unnecessary_operation,
    clippy::vec_init_then_push
)]

use std::cell::RefCell;
use std::rc::{self, Rc};
use std::sync::{self, Arc};

fn main() {}

fn is_ascii(ch: char) -> bool {
    ch.is_ascii()
}

fn clone_on_copy() {
    42.clone();

    vec![1].clone(); // ok, not a Copy type
    Some(vec![1]).clone(); // ok, not a Copy type
    (&42).clone();

    let rc = RefCell::new(0);
    rc.borrow().clone();

    // Issue #4348
    let mut x = 43;
    let _ = &x.clone(); // ok, getting a ref
    'a'.clone().make_ascii_uppercase(); // ok, clone and then mutate
    is_ascii('z'.clone());

    // Issue #5436
    let mut vec = Vec::new();
    vec.push(42.clone());
}
