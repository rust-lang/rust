// edition:2018
#![warn(clippy::wrong_self_convention)]
#![warn(clippy::wrong_pub_self_convention)]
#![allow(dead_code)]

fn main() {}

mod issue6983 {
    pub struct Thing;
    pub trait Trait {
        fn to_thing(&self) -> Thing;
    }

    impl Trait for u8 {
        // don't trigger, e.g. `ToString` from `std` requires `&self`
        fn to_thing(&self) -> Thing {
            Thing
        }
    }

    trait ToU64 {
        fn to_u64(self) -> u64;
    }

    struct FooNoCopy;
    // trigger lint
    impl ToU64 for FooNoCopy {
        fn to_u64(self) -> u64 {
            2
        }
    }
}

mod issue7032 {
    trait Foo {
        fn from_usize(x: usize) -> Self;
    }
    // don't trigger
    impl Foo for usize {
        fn from_usize(x: usize) -> Self {
            x
        }
    }
}
