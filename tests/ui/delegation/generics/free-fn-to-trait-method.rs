#![feature(fn_delegation)]
#![allow(incomplete_features)]

mod default_param {
    pub trait Trait<T = u32> {
        fn foo(&self, _: T) {}
    }

    impl<T> Trait<T> for u8 {}
}

mod types_and_lifetimes {
    pub trait Trait<'a, T> {
        fn foo<'b: 'b>(&'a self, x: &'b T) {
            loop {}
        }
    }
    impl<'a, T> Trait<'a, T> for u8 {}
}

mod bounds {
    pub trait Trait<T> {
        fn foo<U: Clone>(&self, t: T, u: U) where T: Copy {}
    }

    impl<T> Trait<T> for u8 {}
}

mod generic_arguments {
    trait Trait<T> {
        fn foo<U>(&self, _: U, _: T) {}
    }

    impl<T> Trait<T> for u8 {}

    reuse Trait::<_>::foo::<i32> as generic_arguments1;
    //~^ ERROR mismatched types
    reuse <u8 as Trait<_>>::foo as generic_arguments2;
    //~^ ERROR mismatched types
    reuse <_ as Trait<_>>::foo as generic_arguments3; // OK
}

reuse default_param::Trait::foo as default_param;
reuse types_and_lifetimes::Trait::foo as types_and_lifetimes;
reuse bounds::Trait::foo as bounds;

fn main() {
    default_param(&0u8, "hello world"); // OK, default params are not substituted
    types_and_lifetimes::<'static, 'static, _, _>(&0u8, &0u16); // OK, lifetimes go first

    struct S;
    struct U;
    bounds(&0u8, S, U);
    //~^ ERROR the trait bound `S: Copy` is not satisfied
    //~| ERROR the trait bound `U: Clone` is not satisfied
}
