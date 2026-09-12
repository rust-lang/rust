//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
//@ check-pass

#![feature(type_alias_impl_trait)]

pub type Foo<T> = impl PartialEq<(Foo<T>, T)>;

struct Bar<T>(T);

impl<T> PartialEq<(Foo<T>, T)> for Bar<T> {
    fn eq(&self, _: &(Foo<T>, T)) -> bool {
        true
    }
}

#[define_opaque(Foo)]
fn foo<T>(value: T) -> Foo<T> {
    Bar(value)
}

fn main() {}
