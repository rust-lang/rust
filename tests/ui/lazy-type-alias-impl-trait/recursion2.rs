#![feature(type_alias_impl_trait)]

//@ check-pass

type Foo = impl std::fmt::Debug;

#[define_opaque(Foo)]
fn foo(b: bool) -> Foo {
    if b {
        return vec![];
    }
    let x: Vec<i32> = foo(false);
    std::iter::empty().collect()
}

fn bar(b: bool) -> impl std::fmt::Debug {
    if b {
        return vec![];
    }
    let x: Vec<i32> = bar(false);
    std::iter::empty().collect()
}

fn main() {}
