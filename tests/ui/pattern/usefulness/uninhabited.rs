//@ check-pass
//@ aux-build:empty.rs
//
// This tests plays with matching and uninhabited types. This also serves as a test for the
// `Ty::is_inhabited_from` function.
#![feature(never_type)]
#![feature(never_type_fallback)]
#![deny(unreachable_patterns)]

macro_rules! assert_empty {
    ($ty:ty) => {
        const _: () = {
            fn assert_empty(x: $ty) {
                match x {}
                match Some(x) {
                    None => {}
                }
            }
        };
    };
}
macro_rules! assert_non_empty {
    ($ty:ty) => {
        const _: () = {
            fn assert_non_empty(x: $ty) {
                match x {
                    _ => {}
                }
                match Some(x) {
                    None => {}
                    Some(_) => {}
                }
            }
        };
    };
}

extern crate empty;
assert_empty!(empty::EmptyForeignEnum);
assert_empty!(empty::VisiblyUninhabitedForeignStruct);
assert_non_empty!(empty::SecretlyUninhabitedForeignStruct);

enum Void {}
assert_empty!(Void);

enum Enum2 {
    Foo(Void),
    Bar(!),
}
assert_empty!(Enum2);

enum Enum3 {
    Foo(Void),
    Bar {
        x: u64,
        y: !,
    },
}
assert_empty!(Enum3);

enum Enum4 {
    Foo(u64),
    Bar(!),
}
assert_non_empty!(Enum4);

struct Struct1(empty::EmptyForeignEnum);
assert_empty!(Struct1);

struct Struct2 {
    x: u64,
    y: !,
}
assert_empty!(Struct2);

union Union {
    foo: !,
}
assert_non_empty!(Union);

assert_empty!((!, String));

assert_non_empty!(&'static !);
assert_non_empty!(&'static Struct1);
assert_non_empty!(&'static &'static &'static !);

assert_empty!([!; 1]);
assert_empty!([Void; 2]);
assert_non_empty!([!; 0]);
assert_non_empty!(&'static [!]);

mod visibility {
    /// This struct can only be seen to be inhabited in modules `b`, `c` or `d`, because otherwise
    /// the uninhabitedness of both `SecretlyUninhabited` structs is hidden.
    struct SometimesEmptyStruct {
        x: a::b::SecretlyUninhabited,
        y: c::AlsoSecretlyUninhabited,
    }

    /// This enum can only be seen to be inhabited in module `d`.
    enum SometimesEmptyEnum {
        X(c::AlsoSecretlyUninhabited),
        Y(c::d::VerySecretlyUninhabited),
    }

    mod a {
        use super::*;
        pub mod b {
            use super::*;
            pub struct SecretlyUninhabited {
                _priv: !,
            }
            assert_empty!(SometimesEmptyStruct);
        }

        assert_non_empty!(SometimesEmptyStruct);
        assert_non_empty!(SometimesEmptyEnum);
    }

    mod c {
        use super::*;
        pub struct AlsoSecretlyUninhabited {
            _priv: crate::Struct1,
        }
        assert_empty!(SometimesEmptyStruct);
        assert_non_empty!(SometimesEmptyEnum);

        pub mod d {
            use super::*;
            pub struct VerySecretlyUninhabited {
                _priv: !,
            }
            assert_empty!(SometimesEmptyStruct);
            assert_empty!(SometimesEmptyEnum);
        }
    }

    assert_non_empty!(SometimesEmptyStruct);
    assert_non_empty!(SometimesEmptyEnum);
}

fn main() {}
