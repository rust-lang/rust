// Some types whose length depends on the target pointer length will be dumped.
//@ revisions: bit32 bit64
//@[bit32] only-32bit
//@[bit64] only-64bit
//@ run-pass
//@ check-run-results

#![feature(type_info)]
#![allow(dead_code)]

use std::mem::type_info::Type;

struct Foo {
    a: u32,
}

#[non_exhaustive]
struct NonExhaustiveStruct {
    a: u32,
}

struct TupleStruct(u32, u64);

enum Bar {
    Some(u32),
    None,
    #[non_exhaustive]
    Foomp {
        a: (),
        b: &'static str,
    },
}

struct Generics<'a, A, B, const C: u64> {
    a: A,
    b: B,
    l: &'a (),
}

struct Unsized {
    x: u16,
    s: str,
}

macro_rules! dump_types {
    ($($ty:ty),+ $(,)?) => {
        $(println!("{:#?}", const { Type::of::<$ty>() });)+
    };
}

fn main() {
    dump_types! {
        (u8, u8, ()),
        [u8; 2],
        i8, i32, i64, i128, isize,
        u8, u32, u64, u128, usize,
        Foo, Bar, NonExhaustiveStruct, TupleStruct, Generics<i32, u32, 1>,
        &Unsized, &str, &[u8],
        str, [u8],
        &u8, &mut u8,
    }
}
