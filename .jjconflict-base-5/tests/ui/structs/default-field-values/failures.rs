//@ dont-require-annotations: HELP

#![feature(default_field_values)]

#[derive(Debug)]
pub struct S;

#[derive(Debug, Default)]
pub struct Foo {
    pub bar: S = S,
    pub baz: i32 = 42 + 3,
}

#[derive(Debug, Default)]
pub struct Bar {
    pub bar: S, //~ ERROR the trait bound `S: Default` is not satisfied
    pub baz: i32 = 42 + 3,
}

#[derive(Default)]
pub struct Qux<const C: i32> {
    bar: S = Self::S,
    baz: i32 = foo(),
    bat: i32 = <Qux<{ C }> as T>::K,
    bay: i32 = C,
}

pub struct Rak(i32 = 42); //~ ERROR default fields are not supported in tuple structs

impl<const C: i32> Qux<C> {
    const S: S = S;
}

trait T {
    const K: i32;
}

impl<const C: i32> T for Qux<C> {
    const K: i32 = 2;
}

const fn foo() -> i32 {
    42
}

#[derive(Debug, Default)]
enum E {
    #[default]
    Variant {} //~ ERROR the `#[default]` attribute may only be used on unit enum variants
}

union U
{
    x: i32 = 1,  //~ ERROR unions cannot have default field values
    y: f32 = 2., //~ ERROR unions cannot have default field values
}

fn main () {
    let _ = Foo { .. }; // ok
    let _ = Foo::default(); // ok
    let _ = Bar { .. }; //~ ERROR missing field
    let _ = Bar { baz: 0, .. }; //~ ERROR missing field
    let _ = Bar::default(); // silenced
    let _ = Bar { bar: S, .. }; // ok
    let _ = Qux::<4> { .. };
    let _ = Rak(..); //~ ERROR E0308
    //~^ HELP you might have meant to use `..` to skip providing
    let _ = Rak(0, ..); //~ ERROR E0061
    //~^ HELP you might have meant to use `..` to skip providing
    let _ = Rak(.., 0); //~ ERROR E0061
    //~^ HELP you might have meant to use `..` to skip providing
    let _ = Rak { .. }; // ok
}
