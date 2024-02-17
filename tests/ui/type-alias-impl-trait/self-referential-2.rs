//@ revisions: current next
//@[next] compile-flags: -Znext-solver
//@[next] check-pass
#![feature(type_alias_impl_trait)]

type Foo = impl std::fmt::Debug;
type Bar = impl PartialEq<Foo>;

fn bar() -> Bar {
    42_i32 //[current]~^ ERROR can't compare `i32` with `Foo`
}

fn main() {}
