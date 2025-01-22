#![feature(default_field_values)]
#![allow(dead_code)]

mod a {
    pub struct Foo {
        pub something: i32 = 2,
    }

    pub struct Bar {
        pub foo: Foo = Foo { something: 10 }, //~ ERROR default field overrides that field's type's default
    }
}

mod b {
    pub enum Foo {
        X {
            something: i32 = 2,
        }
    }

    pub enum Bar {
        Y {
            foo: Foo = Foo::X { something: 10 }, //~ ERROR default field overrides that field's type's default
        }
    }
}

fn main() {
    let x = a::Bar { .. };
    let y = a::Foo { .. };
    assert_eq!(x.foo.something, y.something);
    let x = b::Bar::Y { .. };
    let y = b::Foo::X { .. };
    match (x, y) {
        (b::Bar::Y { foo: b::Foo::X { something: a} }, b::Foo::X { something:b }) if a == b=> {}
        _ => panic!(),
    }
}
