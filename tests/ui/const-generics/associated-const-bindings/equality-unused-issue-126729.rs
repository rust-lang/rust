//@ check-pass

#![feature(min_generic_const_args)]
#![allow(incomplete_features)]
#![deny(dead_code)]

trait Tr {
    #[rustc_always_gca]
    const I: i32;
}

impl Tr for () {
    const I: i32 = core::direct_const_arg!(1);
}

fn foo() -> impl Tr<I = 1> {}

trait Tr2 {
    #[rustc_always_gca]
    const J: i32;
    #[rustc_always_gca]
    const K: i32;
}

impl Tr2 for () {
    const J: i32 = core::direct_const_arg!(1);
    const K: i32 = core::direct_const_arg!(1);
}

fn foo2() -> impl Tr2<J = 1, K = 1> {}

mod t {
    pub trait Tr3 {
        #[rustc_always_gca]
        const L: i32;
    }

    impl Tr3 for () {
        const L: i32 = core::direct_const_arg!(1);
    }
}

fn foo3() -> impl t::Tr3<L = 1> {}

fn main() {
    foo();
    foo2();
    foo3();
}
