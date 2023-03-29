#![feature(type_alias_impl_trait)]

// check-pass

// FIXME(type_alias_impl_trait): only `baz` and its nested `foo` should compile.

fn main() {}

type Foo = impl std::fmt::Display;

fn bar() {
    #[defines(Foo)]
    pub fn foo() -> Foo {
        "foo"
    }
}

#[defines(Foo)]
fn baz() -> Foo {
    #[defines(Foo)]
    pub fn foo() -> Foo {
        "foo"
    }
    "baz"
}

struct Bak {
    x: [u8; {
        #[defines(Foo)]
        fn blob() -> Foo {
            "blob"
        }
        5
    }],
}
