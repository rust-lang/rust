//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
//@[next] check-pass
#![feature(type_alias_impl_trait)]

type Foo = impl std::fmt::Debug;
type Bar = impl PartialEq<Foo>;

#[define_opaque(Bar)]
fn bar() -> Bar {
    42_i32 //[current]~^ ERROR can't compare `i32` with `Foo`
}

fn main() {}
