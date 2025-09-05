#![allow(dead_code)]
#![feature(const_trait_impl)]
#![feature(const_default)]
#![feature(derive_const)]

mod issue15493 {
    #[derive(Copy, Clone)]
    #[repr(transparent)]
    struct Foo(u64);

    impl const Default for Foo {
        //~^ derivable_impls
        fn default() -> Self {
            Self(0)
        }
    }

    #[derive(Copy, Clone)]
    enum Bar {
        A,
        B,
    }

    impl const Default for Bar {
        //~^ derivable_impls
        fn default() -> Self {
            Bar::A
        }
    }
}

fn main() {}
