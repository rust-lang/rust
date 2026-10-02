//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
//@[next] check-pass
#![feature(type_alias_impl_trait)]

type Foo = impl std::fmt::Debug;

#[define_opaque(Foo)]
fn foo(b: bool) -> Foo {
    if b { vec![42_i32] } else { std::iter::empty().collect() }
}

type Bar = impl std::fmt::Debug;

#[define_opaque(Bar)]
fn bar(b: bool) -> Bar {
    let x: Bar = if b {
        vec![42_i32]
    } else {
        std::iter::empty().collect()
        //[current]~^ ERROR  a value of type `Bar` cannot be built from an iterator over elements of type `_`
    };
    x
}

fn main() {}
