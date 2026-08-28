//@ compile-flags: -Znext-solver
//@ check-pass

#![feature(type_alias_impl_trait)]

// Checking the opaque's recursive item bound requires normalizing `Foo`
// to its provisional hidden type while selecting the `PartialEq` impl.
// This ensures we do not handle recursive TAIT bounds by suppressing
// normalization of the recursive opaque.

pub type Foo = impl PartialEq<(Foo, i32)>;

#[define_opaque(Foo)]
fn foo() -> Foo {
    Bar
}

struct Bar;

impl PartialEq<(Bar, i32)> for Bar {
    fn eq(&self, _: &(Bar, i32)) -> bool {
        true
    }
}

fn main() {}
