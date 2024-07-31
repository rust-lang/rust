//@ check-pass

#![feature(type_alias_impl_trait)]
#![allow(dead_code)]

type Foo = impl Iterator<Item = impl Send>;

#[defines(Foo)]
fn make_foo() -> Foo {
    vec![1, 2].into_iter()
}

type Bar = impl Send;
type Baz = impl Iterator<Item = Bar>;

// TODO: require `Bar` in this list, too (stop walking through type aliases in `opaque_types_defined_by`)
#[defines(Baz)]
fn make_baz() -> Baz {
    vec!["1", "2"].into_iter()
}

fn main() {}
