#![feature(type_alias_impl_trait)]

fn main() {}

type Foo = impl std::fmt::Display;

#[defines(Foo)]
fn bar() {
    pub fn foo() -> Foo {
        "foo"
        //~^ ERROR: cannot register hidden type without a `#[defines
    }
}

#[defines(Foo)]
fn baz() -> Foo {
    pub fn foo() -> Foo {
        "foo"
        //~^ ERROR: cannot register hidden type without a `#[defines
    }
    "baz"
}

#[defines(Foo)]
struct Bak {
    x: [u8; {
        fn blob() -> Foo {
            "blob"
            //~^ ERROR: cannot register hidden type without a `#[defines
        }
        5
    }],
}
