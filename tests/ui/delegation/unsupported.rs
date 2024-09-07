#![feature(const_trait_impl)]
#![feature(c_variadic)]
#![feature(effects)]
#![feature(fn_delegation)]
#![allow(incomplete_features)]

mod opaque {
    trait Trait {}
    impl Trait for () {}

    mod to_reuse {
        use super::Trait;

        pub fn opaque_ret() -> impl Trait { unimplemented!() }
        //~^ warn: this function depends on never type fallback being `()`
        //~| warn: this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!
    }

    trait ToReuse {
        fn opaque_ret() -> impl Trait { unimplemented!() }
        //~^ warn: this function depends on never type fallback being `()`
        //~| warn: this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!
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
    //~^ ERROR delegation to a function with effect parameter is not supported yet
}

fn main() {}
