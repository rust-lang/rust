//@ check-pass

#![feature(associated_const_equality)]
#![deny(dead_code)]

trait Tr {
    const I: i32;
}

impl Tr for () {
    const I: i32 = 1;
}

fn foo() -> impl Tr<I = 1> {}

trait Tr2 {
    const J: i32;
    const K: i32;
}

impl Tr2 for () {
    const J: i32 = 1;
    const K: i32 = 1;
}

fn foo2() -> impl Tr2<J = 1, K = 1> {}

mod t {
    pub trait Tr3 {
        const L: i32;
    }

    impl Tr3 for () {
        const L: i32 = 1;
    }
}

fn foo3() -> impl t::Tr3<L = 1> {}

fn main() {
    foo();
    foo2();
    foo3();
}
