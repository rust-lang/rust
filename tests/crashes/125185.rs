//@ known-bug: rust-lang/rust#125185
//@ compile-flags: -Zvalidate-mir

#![feature(type_alias_impl_trait)]

type Foo = impl Send;

struct A;

#[define_opaques(Foo)]
const fn foo() -> Foo {
    value()
}

const VALUE: Foo = foo();

#[define_opaques(Foo)]
fn test(foo: Foo, f: impl for<'b> FnMut()) {
    match VALUE {
        0 | 0 => {}

        _ => (),
    }
}

fn main() {}
