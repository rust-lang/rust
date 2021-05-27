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
    // don't trigger
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

mod issue7179 {
    pub struct S(i32);

    impl S {
        // don't trigger (`s` is not `self`)
        pub fn from_be(s: Self) -> Self {
            S(i32::from_be(s.0))
        }

        // lint
        pub fn from_be_self(self) -> Self {
            S(i32::from_be(self.0))
        }
    }

    trait T {
        // don't trigger (`s` is not `self`)
        fn from_be(s: Self) -> Self;
        // lint
        fn from_be_self(self) -> Self;
    }

    trait Foo: Sized {
        fn as_byte_slice(slice: &[Self]) -> &[u8];
    }
}
