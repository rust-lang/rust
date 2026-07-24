//! Test that using `#[arg_splat]` on non-tuple function arguments is an error.

#![allow(incomplete_features)]
#![feature(arg_splat)]
#![expect(unused)]

fn primitive_arg(#[arg_splat] x: u32) {} //~ ERROR cannot use splat attribute; the splatted argument type must be a tuple or unit, not a u32

enum NotATuple {
    A(u32),
    B(i8),
}

fn enum_arg(#[arg_splat] y: NotATuple) {} //~ ERROR cannot use splat attribute; the splatted argument type must be a tuple or unit, not a NotATuple

trait FooTrait {
    fn tuple_1(#[arg_splat] _: (u32,)); //~ NOTE type in trait

    // Ambiguous case, self could be a tuple or a non-tuple
    fn tuple_4(#[arg_splat] self, _: (u32, i8, (), f32));
}

struct Foo;

fn struct_arg(#[arg_splat] z: Foo) {} //~ ERROR cannot use splat attribute; the splatted argument type must be a tuple or unit, not a Foo

impl Foo {
    fn tuple_2_self(
        // FIXME(splat): ERROR cannot use splat attribute; the splatted argument type must be a...
        #[arg_splat] self,
        (a, b): (u32, i8),
    ) -> u32 {
        a
    }
}

impl FooTrait for Foo {
    fn tuple_1(_: (u32,)) {}
    //~^ ERROR method `tuple_1` has an incompatible type for trait
    //~| NOTE expected fn with arg 0 splatted, found fn with no splatted arg
    //~| NOTE expected signature `fn(#[arg_splat] (_,))`
    //~| NOTE found signature `fn((_,))`

    fn tuple_4(
        // FIXME(splat): ERROR cannot use splat attribute; the splatted argument type must be a...
        #[arg_splat] self,
        _: (u32, i8, (), f32),
    ) {
    }
}

struct TupleStruct(u32, i8);

fn tuple_struct_arg(#[arg_splat] z: TupleStruct) {} //~ ERROR cannot use splat attribute; the splatted argument type must be a tuple or unit, not a TupleStruct

impl TupleStruct {
    fn tuple_2(
        #[arg_splat] self, // FIXME(splat): ERROR `#[arg_splat]` attribute must be used on a tuple
        (a, b): (f32, f64),
    ) -> f32 {
        a
    }
}

impl FooTrait for TupleStruct {
    fn tuple_1(#[arg_splat] _: (u32,)) {}

    fn tuple_4(
        #[arg_splat] self, // FIXME(splat): ERROR `#[arg_splat]` attribute must be used on a tuple
        _: (u32, i8, (), f32),
    ) {
    }
}

fn main() {
    // FIXME(splat): is it enough for just the definitions/callees to error,
    // or should the callers also error?
    primitive_arg(1u32);
    enum_arg(NotATuple::A(1u32));

    let foo = Foo;
    struct_arg(foo);
    foo.tuple_2_self((1u32, 2i8));

    let tuple_struct = TupleStruct(1u32, 2i8);
    tuple_struct_arg(tuple_struct);
    tuple_struct.tuple_2((1f32, 2f64));
    TupleStruct::tuple_1(1u32);
    tuple_struct.tuple_4((1u32, 2i8, (), 3f32));
}
