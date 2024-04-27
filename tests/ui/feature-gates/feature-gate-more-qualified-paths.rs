fn main() {
    // destructure through a qualified path
    let <Foo as A>::Assoc { br } = StructStruct { br: 2 };
    //~^ ERROR usage of qualified paths in this context is experimental
    let _ = <Foo as A>::Assoc { br: 2 };
    //~^ ERROR usage of qualified paths in this context is experimental
    let <E>::V(..) = E::V(0);
    //~^ ERROR usage of qualified paths in this context is experimental
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
    V(u8)
}
