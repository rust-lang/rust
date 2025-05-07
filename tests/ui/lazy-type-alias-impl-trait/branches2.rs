#![feature(type_alias_impl_trait)]

//@ check-pass

type Foo = impl std::iter::FromIterator<i32> + PartialEq<Vec<i32>> + std::fmt::Debug;

#[define_opaque(Foo)]
fn foo(b: bool) -> Foo {
    if b { vec![42_i32] } else { std::iter::empty().collect() }
}

fn bar(b: bool) -> impl PartialEq<Vec<i32>> + std::fmt::Debug {
    if b { vec![42_i32] } else { std::iter::empty().collect() }
}

fn main() {
    assert_eq!(foo(true), vec![42]);
    assert_eq!(foo(false), vec![]);
    assert_eq!(bar(true), vec![42]);
    assert_eq!(bar(false), vec![]);
}
