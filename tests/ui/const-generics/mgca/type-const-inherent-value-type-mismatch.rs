// Regression test for https://github.com/rust-lang/rust/issues/152962

//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
//@ compile-flags: -Zvalidate-mir

#![feature(min_generic_const_args)]
#![expect(incomplete_features)]

struct Struct;

impl Struct {
    type const N: usize = "this isn't a usize";
    //~^ ERROR the constant `"this isn't a usize"` is not of type `usize`
}

fn f() -> [u8; const { Struct::N }] {}
//~^ ERROR mismatched types [E0308]
//[next]~| ERROR type annotations needed

fn main() {}
