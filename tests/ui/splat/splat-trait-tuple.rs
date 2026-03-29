//@ run-pass
//! Test using `#[splat]` on trait assoc function/method tuple arguments.

#![allow(incomplete_features)]
#![feature(splat)]

trait FooTrait {
    fn tuple_1_trait(#[splat] _: (u32,));

    fn tuple_2_trait(&self, #[splat] _: (u32, f32));
}

struct Foo;

impl FooTrait for Foo {
    // Currently, splat attributes on impls must match traits. This provides better UX.
    fn tuple_1_trait(#[splat] _: (u32,)) {}

    fn tuple_2_trait(&self, #[splat] _: (u32, f32)) {}
}

struct TupleStruct(u32, i8); //~ WARN fields `0` and `1` are never read

impl FooTrait for TupleStruct {
    fn tuple_1_trait(#[splat] _: (u32,)) {}

    fn tuple_2_trait(&self, #[splat] _: (u32, f32)) {}
}

fn main() {
    // FIXME(splat): should splatted functions be callable with tupled and un-tupled arguments?
    // Add a tupled test for each call if they are.
    //Foo::tuple_1_trait((1u32,));

    let foo = Foo;
    Foo::tuple_1_trait(1u32);
    foo.tuple_2_trait(1, 3.5);

    let tuple_struct = TupleStruct(1u32, 2i8);
    TupleStruct::tuple_1_trait(1u32);
    tuple_struct.tuple_2_trait(1, 3.5)
}
