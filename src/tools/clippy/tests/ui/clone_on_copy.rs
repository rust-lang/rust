#![allow(
    unused,
    clippy::redundant_clone,
    clippy::deref_addrof,
    clippy::no_effect,
    clippy::unnecessary_operation,
    clippy::vec_init_then_push,
    clippy::toplevel_ref_arg,
    clippy::needless_borrow
)]

use std::cell::RefCell;
use std::rc::{self, Rc};
use std::sync::{self, Arc};

fn main() {}

fn is_ascii(ch: char) -> bool {
    ch.is_ascii()
}

fn clone_on_copy() -> Option<(i32)> {
    42.clone();
    //~^ clone_on_copy

    vec![1].clone(); // ok, not a Copy type
    Some(vec![1]).clone(); // ok, not a Copy type
    (&42).clone();
    //~^ clone_on_copy

    let rc = RefCell::new(0);
    rc.borrow().clone();
    //~^ clone_on_copy

    let x = 0u32;
    x.clone().rotate_left(1);
    //~^ clone_on_copy

    #[derive(Clone, Copy)]
    struct Foo;
    impl Foo {
        fn clone(&self) -> u32 {
            0
        }
    }
    Foo.clone(); // ok, this is not the clone trait

    macro_rules! m {
        ($e:expr) => {{ $e }};
    }
    m!(42).clone();
    //~^ clone_on_copy

    struct Wrap([u32; 2]);
    impl core::ops::Deref for Wrap {
        type Target = [u32; 2];
        fn deref(&self) -> &[u32; 2] {
            &self.0
        }
    }
    let x = Wrap([0, 0]);
    x.clone()[0];
    //~^ clone_on_copy

    let x = 42;
    let ref y = x.clone(); // ok, binds by reference
    let ref mut y = x.clone(); // ok, binds by reference

    // Issue #4348
    let mut x = 43;
    let _ = &x.clone(); // ok, getting a ref
    'a'.clone().make_ascii_uppercase(); // ok, clone and then mutate
    is_ascii('z'.clone());
    //~^ clone_on_copy

    // Issue #5436
    let mut vec = Vec::new();
    vec.push(42.clone());
    //~^ clone_on_copy

    //  Issue #9277
    let opt: &Option<i32> = &None;
    let value = opt.clone()?; // operator precedence needed (*opt)?
    //
    //~^^ clone_on_copy
    None
}
