// Regression test for #57188

// check-pass

#![feature(type_alias_impl_trait)]

struct Baz<'a> {
    source: &'a str,
}

trait Foo<'a> {
    type T: Iterator<Item = Baz<'a>> + 'a;
    fn foo(source: &'a str) -> Self::T;
}

struct Bar;
impl<'a> Foo<'a> for Bar {
    type T = impl Iterator<Item = Baz<'a>> + 'a;
    fn foo(source: &'a str) -> Self::T {
        std::iter::once(Baz { source })
    }
}

fn main() {}
