#![feature(c_variadic)]
#![feature(fn_delegation)]
#![allow(incomplete_features)]

mod generics {
    trait Trait0 {
        fn bar(&self) {}
    }

    trait Trait1<T> {
        fn foo(&self) {}
        reuse Trait0::bar;
        //~^ ERROR early bound generics are only supported for trait implementations and free functions
    }

    trait Trait2 {
        reuse Trait1::foo;
        //~^ ERROR early bound generics are only supported for trait implementations and free functions
    }

    struct S;
    impl S {
        reuse Trait1::foo;
        //~^ ERROR early bound generics are only supported for trait implementations and free functions
    }
}

mod opaque {
    trait Trait {}
    impl Trait for () {}

    mod to_reuse {
        use super::Trait;

        pub fn opaque_ret() -> impl Trait { unimplemented!() }
    }

    trait ToReuse {
        fn opaque_ret() -> impl Trait { unimplemented!() }
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

fn main() {}
