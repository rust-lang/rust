//! This test used to ICE #121613
//! Using a generic parameter where there are none expected
//! caused an ICE, hiding the important later errors.

#![feature(more_qualified_paths)]

fn main() {
    let _ = <Foo as A>::Assoc { br: 2 };

    let <E>::V(..) = E::V(|a, b| a.cmp(b));
    //~^ ERROR: multiple applicable items in scope
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

enum E {
    V(u8),
}
