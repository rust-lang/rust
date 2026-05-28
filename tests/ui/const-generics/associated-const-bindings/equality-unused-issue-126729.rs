//@ check-pass

#![feature(min_generic_const_args)]
#![allow(incomplete_features)]
#![deny(dead_code)]

trait Tr {
    type const I: i32;
}

impl Tr for () {
    type const I: i32 = 1;
}

fn foo() -> impl Tr<I = 1> {}

trait Tr2 {
    type const J: i32;
    type const K: i32;
}

impl Tr2 for () {
    type const J: i32 = 1;
    type const K: i32 = 1;
}

fn foo2() -> impl Tr2<J = 1, K = 1> {}

mod t {
    pub trait Tr3 {
        type     const L: i32;
    }

    impl Tr3 for () {
        type     const L: i32 = 1;
    }
}

fn foo3() -> impl t::Tr3<L = 1> {}

fn main() {
    foo();
    foo2();
    foo3();
}
