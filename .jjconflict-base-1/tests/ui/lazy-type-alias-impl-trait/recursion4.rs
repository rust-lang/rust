#![feature(type_alias_impl_trait)]

type Foo = impl std::fmt::Debug;

#[define_opaque(Foo)]
fn foo(b: bool) -> Foo {
    if b {
        return vec![];
    }
    let mut x = foo(false);
    x = std::iter::empty().collect(); //~ ERROR `Foo` cannot be built from an iterator
    vec![]
}

fn bar(b: bool) -> impl std::fmt::Debug {
    if b {
        return vec![];
    }
    let mut x = bar(false);
    x = std::iter::empty().collect(); //~ ERROR `impl Debug` cannot be built from an iterator
    vec![]
}

fn main() {}
