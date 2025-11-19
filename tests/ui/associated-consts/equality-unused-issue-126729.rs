//@ check-pass

#![feature(associated_const_equality, min_generic_const_args)]
#![allow(incomplete_features)]
#![deny(dead_code)]

trait Tr {
    #[type_const]
    const I: i32;
}

impl Tr for () {
    #[type_const]
    const I: i32 = const { 1 };
}

fn foo() -> impl Tr<I = const { 1 }> {}

trait Tr2 {
    #[type_const]
    const J: i32;
    #[type_const]
    const K: i32;
}

impl Tr2 for () {
    #[type_const]
    const J: i32 = const { 1 };
    #[type_const]
    const K: i32 = const { 1 };
}

fn foo2() -> impl Tr2<J = const { 1 }, K = const { 1 }> {}

mod t {
    pub trait Tr3 {
        #[type_const]
        const L: i32;
    }

    impl Tr3 for () {
        #[type_const]
        const L: i32 = const { 1 };
    }
}

fn foo3() -> impl t::Tr3<L = const { 1 }> {}

fn main() {
    foo();
    foo2();
    foo3();
}
