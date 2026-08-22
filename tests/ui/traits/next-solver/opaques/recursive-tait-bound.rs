//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
//@ check-pass

#![feature(type_alias_impl_trait)]

pub type Foo = impl PartialEq<(Foo, i32)>;

#[define_opaque(Foo)]
fn foo() -> Foo {
    Bar
}

struct Bar;

impl PartialEq<(Foo, i32)> for Bar {
    fn eq(&self, _: &(Foo, i32)) -> bool {
        true
    }
}

fn main() {}
