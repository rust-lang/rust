//@ run-pass
#![feature(fn_delegation)]
#![allow(incomplete_features)]

mod types {
    pub trait Trait<T> {
        fn foo<U>(&self, x: U, y: T) -> (T, U) {(y, x)}
    }
    impl<T> Trait<T> for u8 {}
}

mod types_and_lifetimes {
    pub trait Trait<'a, T> {
        fn foo<'b, U>(&self, _: &'b U, _: &'a T) -> bool {
            true
        }
    }
    impl<'a, T> Trait<'a, T> for u8 {}
}

reuse types::Trait::foo as types;
reuse types_and_lifetimes::Trait::foo as types_and_lifetimes;

fn main() {
    assert_eq!(types(&2, "str", 1), (1, "str"));

    struct T;
    struct U;
    assert_eq!(types_and_lifetimes::<u8, T, U>(&1, &U, &T), true);
}
