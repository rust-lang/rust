//@ revisions: current next
//@[next] compile-flags: -Znext-solver
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[current] known-bug: #100041
//@[current] check-pass
// FIXME(inherent_associated_types): This should fail.

#![feature(inherent_associated_types)]
#![allow(incomplete_features)]

struct Foo;

impl Foo {
    type Bar<T> = ();
}

fn main() -> Foo::Bar::<Vec<[u32]>> {}
//[next]~^ ERROR the size for values of type `[u32]` cannot be known at compilation time
