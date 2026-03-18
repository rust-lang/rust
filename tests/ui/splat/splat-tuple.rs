//@ dont-check-compiler-stderr
//@ dont-check-failure-status
//@ dont-require-annotations: ERROR
// FIXME(splat): ^change the actual types during typeck so MIR doesn't ICE.

//! Test using `#[splat]` on tuple function arguments.

#![allow(incomplete_features)]
#![feature(splat)]
#![feature(tuple_trait)]

fn tuple_args(#[splat] (a, b): (u32, i8)) {}

trait FooTrait {
    fn tuple_1_trait(#[splat] _: (u32,));

    // Ambiguous case, self could be a tuple or a non-tuple.
    fn tuple_4_trait(#[splat] &self, _: (u32, i8, (), f32)) {}
}

struct Foo;

impl Foo {
    fn tuple_1(#[splat] (a,): (u32,)) {}

    fn tuple_2(&self, #[splat] (a, b): (u32, i8)) -> u32 {
        a
    }

    fn tuple_3(#[splat] (a, b, c): (u32, i32, i8)) {}

    fn tuple_4(&self, #[splat] (a, b, c, d): (u32, i8, (), f32)) -> u32 {
        a
    }
}

impl FooTrait for Foo {
    // FIXME(splat): should splat attributes be inherited from traits?
    // Can splat attributes be added on impls?
    fn tuple_1_trait(#[splat] _: (u32,)) {}
}

struct TupleStruct(u32, i8);

impl TupleStruct {
    // FIXME(splat): tuple structs should error until we have a specific use case for them.
    fn tuple_2(#[splat] &self, (a, b): (u32, i8)) -> u32 {
        a
    }
}

impl FooTrait for TupleStruct {
    fn tuple_1_trait(#[splat] _: (u32,)) {}
}

extern "C" {
    // FIXME(splat): tuple layouts are unspecified, so this should error.
    #[expect(improper_ctypes)]
    fn bar_2(#[splat] _: (u32, i8));
}

// FIXME(splat): multiple splats in a fn should error.
fn multisplat_bad_2(#[splat] (a, b): (u32, i8), #[splat] (c, d): (u32, i8)) {}

// FIXME(splat): non-terminal splat attributes should error, until we have a specific use case for
// them.
fn splat_non_terminal_bad(#[splat] (a, b): (u32, i8), (c, d): (u32, i8)) {}

fn splat_generic_tuple<T: std::marker::Tuple>(#[splat] t: T) {}

fn main() {
    // FIXME(splat): actually modify the argument list during typeck, to avoid "broken MIR" errors
    tuple_args(1, 2); //~ ERROR: broken MIR in
    // FIXME(splat): should splatted functions be callable with tupled and un-tupled arguments?
    // Add a tupled test for each call if they are.
    //tuple_args((1, 2));
    tuple_args(1u32, 2i8); //~ ERROR: broken MIR in

    let foo = Foo;
    Foo::tuple_1(1u32); //~ ERROR: broken MIR in
    foo.tuple_2(1u32, 2i8); //~ ERROR: broken MIR in
    Foo::tuple_3(1u32, 2i32, 3i8); //~ ERROR broken MIR
    foo.tuple_4(1u32, 2i8, (), 3f32);

    Foo::tuple_1_trait(1u32);
    // FIXME(splat): this should error because `self` is splatted, but `Foo` is not a tuple.
    foo.tuple_4_trait((1u32, 2i8, (), 3f32));

    let tuple_struct = TupleStruct(1u32, 2i8);
    tuple_struct.tuple_2((1u32, 2i8));

    TupleStruct::tuple_1_trait(1u32);
    // FIXME(splat): this should error because `self` is splatted, but `TupleStruct` is not a tuple.
    tuple_struct.tuple_4_trait((1u32, 2i8, (), 3f32));

    // FIXME(splat): generic tuple trait implementers should work without explicit tuple type
    // parameters.
    splat_generic_tuple::<(u32, i8)>(1u32, 2i8);

    splat_generic_tuple::<()>();
}
