//@ check-pass

#![feature(min_generic_const_args)]
#![allow(incomplete_features)]
#![deny(dead_code)]

trait Tr {
    #[type_const]
    const I: i32;
}

impl Tr for () {
    #[type_const]
    const I: i32 = 1;
}

fn foo() -> impl Tr<I = 1> {}

trait Tr2 {
    #[type_const]
    const J: i32;
    #[type_const]
    const K: i32;
}

impl Tr2 for () {
    #[type_const]
    const J: i32 = 1;
    #[type_const]
    const K: i32 = 1;
}

fn foo2() -> impl Tr2<J = 1, K = 1> {}

mod t {
    pub trait Tr3 {
        #[type_const]
        const L: i32;
    }

    impl Tr3 for () {
        #[type_const]
        const L: i32 = 1;
    }
}

fn foo3() -> impl t::Tr3<L = 1> {}

fn main() {
    foo();
    foo2();
    foo3();
}
