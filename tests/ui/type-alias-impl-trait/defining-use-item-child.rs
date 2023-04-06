#![feature(type_alias_impl_trait)]

fn main() {}

type Foo = impl std::fmt::Display;

#[defines(Foo)]
fn bar() {
    pub fn foo() -> Foo {
        "foo"
        //~^ ERROR: mismatched types
    }
}

#[defines(Foo)]
fn baz() -> Foo {
    pub fn foo() -> Foo {
        "foo"
        //~^ ERROR: mismatched types
    }
    "baz"
}

#[defines(Foo)]
struct Bak {
    x: [u8; {
        fn blob() -> Foo {
            "blob"
            //~^ ERROR: mismatched types
        }
        5
    }],
}
