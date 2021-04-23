// Regression test for #57188

// check-pass

// revisions: min_tait full_tait
#![feature(min_type_alias_impl_trait)]
#![cfg_attr(full_tait, feature(type_alias_impl_trait))]
//[full_tait]~^ WARN incomplete

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
