//@ run-pass
//! Test using `#[splat]` on method tuple arguments (with receivers).

#![allow(incomplete_features)]
#![feature(splat)]

struct Foo;

impl Foo {
    fn tuple_2(&self, #[splat] (a, _b): (u32, i8)) -> u32 {
        a
    }

    fn tuple_4(&self, #[splat] (a, _b, _c, _d): (u32, i8, (), f32)) -> u32 {
        a
    }
}

struct TupleStruct(u32, i8); //~ WARN: fields `0` and `1` are never read

impl TupleStruct {
    fn tuple_2(&self, #[splat] (a, _b): (u32, i8)) -> u32 {
        a
    }
}

fn main() {
    // FIXME(splat): should splatted functions be callable with tupled and un-tupled arguments?
    // Add a tupled test for each call if they are.
    //foo.tuple_2((1, 2));

    let foo = Foo;
    foo.tuple_2(1u32, 2i8);
    foo.tuple_4(1u32, 2i8, (), 3f32);

    let tuple_struct = TupleStruct(1u32, 2i8);
    tuple_struct.tuple_2(1u32, 2i8);
}
