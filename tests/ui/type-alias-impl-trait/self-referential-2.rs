//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
//@ check-pass
#![feature(type_alias_impl_trait)]

type Foo = impl std::fmt::Debug;
type Bar = impl PartialEq<Foo>;

fn bar() -> Bar {
    42_i32
}

fn main() {}
