#![feature(type_alias_impl_trait)]

fn main() {}

type Foo = impl std::fmt::Display;

fn bar() {
    pub fn foo() -> Foo {
        "foo"
        //~^ ERROR: opaque type constrained in nested item
    }
}

fn baz() -> Foo {
    pub fn foo() -> Foo {
        "foo"
        //~^ ERROR: opaque type constrained in nested item
    }
    "baz"
}

struct Bak {
    x: [u8; {
        fn blob() -> Foo {
            "blob"
            //~^ ERROR: opaque type constrained in nested item
        }
        5
    }],
}
