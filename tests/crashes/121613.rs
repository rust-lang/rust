//@ known-bug: #121613
fn main() {
    let _ = <Foo as A>::Assoc { br: 2 };

    let <E>::V(..) = E::V(|a, b| a.cmp(b));
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
