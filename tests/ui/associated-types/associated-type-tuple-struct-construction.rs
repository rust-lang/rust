// Users cannot yet construct structs through associated types
// in both expressions and patterns

#![feature(more_qualified_paths)]

fn main() {
    let <Foo as A>::Assoc(n) = <Foo as A>::Assoc(2);
    //~^ ERROR expected method or associated constant, found associated type
    //~| ERROR expected method or associated constant, found associated type
    assert!(n == 2);
}

struct TupleStruct(i8);

struct Foo;


trait A {
    type Assoc;
}

impl A for Foo {
    type Assoc = TupleStruct;
}
