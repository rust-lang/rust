#![warn(clippy::empty_structs_with_brackets)]
#![allow(dead_code)]

pub struct MyEmptyStruct {} // should trigger lint
//~^ empty_structs_with_brackets
struct MyEmptyTupleStruct(); // should trigger lint
//~^ empty_structs_with_brackets

// should not trigger lint
struct MyCfgStruct {
    #[cfg(feature = "thisisneverenabled")]
    field: u8,
}

// should not trigger lint
struct MyCfgTupleStruct(#[cfg(feature = "thisisneverenabled")] u8);

// should not trigger lint
struct MyStruct {
    field: u8,
}
struct MyTupleStruct(usize, String); // should not trigger lint
struct MySingleTupleStruct(usize); // should not trigger lint
struct MyUnitLikeStruct; // should not trigger lint

macro_rules! empty_struct {
    ($s:ident) => {
        struct $s {}
    };
}

empty_struct!(FromMacro);

fn main() {}

mod issue15349 {
    trait Bar<T> {}
    impl<T> Bar<T> for [u8; 7] {}

    struct Foo<const N: usize> {}
    //~^ empty_structs_with_brackets
    impl<const N: usize> Foo<N>
    where
        [u8; N]: Bar<[(); N]>,
    {
        fn foo() {}
    }
}
