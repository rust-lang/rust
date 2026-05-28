//@aux-build:proc_macro_derive.rs
#![allow(clippy::clone_on_copy, unused)]
#![allow(clippy::assigning_clones)]
#![no_main]

extern crate proc_macros;
use proc_macros::inline_macros;

struct A(u32);

impl Clone for A {
    fn clone(&self) -> Self {
        //~^ non_canonical_clone_impl
        Self(self.0)
    }

    fn clone_from(&mut self, source: &Self) {
        //~^ non_canonical_clone_impl
        source.clone();
        *self = source.clone();
    }
}

impl Copy for A {}

// do not lint

struct B(u32);

impl Clone for B {
    fn clone(&self) -> Self {
        *self
    }
}

impl Copy for B {}

// do not lint derived (clone's implementation is `*self` here anyway)
#[derive(Clone, Copy)]
struct C(u32);

// do not lint derived (fr this time)
struct D(u32);

#[automatically_derived]
impl Clone for D {
    fn clone(&self) -> Self {
        Self(self.0)
    }

    fn clone_from(&mut self, source: &Self) {
        source.clone();
        *self = source.clone();
    }
}

impl Copy for D {}

// do not lint if clone is not manually implemented
struct E(u32);

#[automatically_derived]
impl Clone for E {
    fn clone(&self) -> Self {
        Self(self.0)
    }

    fn clone_from(&mut self, source: &Self) {
        source.clone();
        *self = source.clone();
    }
}

impl Copy for E {}

// lint since clone is not derived

#[derive(Copy)]
struct F(u32);

impl Clone for F {
    fn clone(&self) -> Self {
        //~^ non_canonical_clone_impl
        Self(self.0)
    }

    fn clone_from(&mut self, source: &Self) {
        //~^ non_canonical_clone_impl
        source.clone();
        *self = source.clone();
    }
}

// do not lint since copy has more restrictive bounds
#[derive(Eq, PartialEq)]
struct Uwu<A: Copy>(A);

impl<A: Copy> Clone for Uwu<A> {
    fn clone(&self) -> Self {
        Self(self.0)
    }

    fn clone_from(&mut self, source: &Self) {
        source.clone();
        *self = source.clone();
    }
}

impl<A: std::fmt::Debug + Copy + Clone> Copy for Uwu<A> {}

#[inline_macros]
mod issue12788 {
    use proc_macros::{external, with_span};

    // lint non-external macro
    inline!(
        #[derive(Copy)]
        pub struct A;

        impl Clone for A {
            fn clone(&self) -> Self {
                //~^ non_canonical_clone_impl
                todo!()
            }
        }
    );

    // do not lint external macros
    external!(
        #[derive(Copy)]
        pub struct B;

        impl Clone for B {
            fn clone(&self) -> Self {
                todo!()
            }
        }
    );

    // do not lint proc macros
    #[derive(proc_macro_derive::NonCanonicalClone)]
    pub struct C;

    with_span!(
        span

        #[derive(Copy)]
        struct D;
        impl Clone for D {
            fn clone(&self) -> Self {
                todo!()
            }
        }
    );
}

struct N(u32);

impl Clone for N {
    fn clone(&self) -> Self {
        //~^ non_canonical_clone_impl
        { *self }
    }
}

impl Copy for N {}

/// Test for corner cases with `implicit_return` enabled
mod with_implicit_return {
    #![warn(clippy::implicit_return)]
    #![allow(clippy::needless_return)]

    // Don't lint `return *self` under `implicit_return`
    struct G(u32);

    impl Clone for G {
        fn clone(&self) -> Self {
            return *self;
        }
    }

    impl Copy for G {}

    struct H(u32);

    impl Clone for H {
        fn clone(&self) -> Self {
            //~^ non_canonical_clone_impl
            {
                return *self;
            }
        }
    }

    impl Copy for H {}
}
