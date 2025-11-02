//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
//@ check-fail

// Next solver revision included because of trait-system-refactor-initiative#234.
// If we end up in a query cycle, it should be okay as long as results are the same.

#![feature(const_trait_impl)]
#![feature(c_variadic)]
#![feature(fn_delegation)]
#![allow(incomplete_features)]

mod opaque {
    trait Trait {}
    impl Trait for () {}

    mod to_reuse {
        use super::Trait;

        pub fn opaque_ret() -> impl Trait { () }
    }

    trait ToReuse {
        fn opaque_ret() -> impl Trait { () }
    }

    // FIXME: Inherited `impl Trait`s create query cycles when used inside trait impls.
    impl ToReuse for u8 {
        reuse to_reuse::opaque_ret; //~ ERROR cycle detected when computing type
    }
    impl ToReuse for u16 {
        reuse ToReuse::opaque_ret; //~ ERROR cycle detected when computing type
    }
}

mod recursive {
    mod to_reuse1 {
        pub mod to_reuse2 {
            pub fn foo() {}
        }

        pub reuse to_reuse2::foo;
    }

    reuse to_reuse1::foo;
    //~^ ERROR recursive delegation is not supported yet
}

mod effects {
    #[const_trait]
    trait Trait {
        fn foo();
    }

    reuse Trait::foo;
    //~^ ERROR type annotations needed
}

fn main() {}
