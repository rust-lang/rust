#![feature(type_alias_impl_trait)]

//@ check-pass

type Foo = impl std::fmt::Debug;

#[define_opaque(Foo)]
fn foo(b: bool) -> Foo {
    if b {
        return 42;
    }
    let x: u32 = foo(false);
    99
}

fn bar(b: bool) -> impl std::fmt::Debug {
    if b {
        return 42;
    }
    let x: u32 = bar(false);
    99
}

fn main() {}
