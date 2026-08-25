#![feature(more_qualified_paths)]

fn main() {
    // destructure through a qualified path
    let <Foo as A>::Assoc { br } = StructStruct { br: 2 };
}

struct StructStruct {
    br: i8,
}

struct Foo;

trait A {
    type Assoc;
}

impl A for Foo {
    type Assoc = StructStruct;
}
