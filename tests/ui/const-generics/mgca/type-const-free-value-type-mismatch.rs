#![feature(min_generic_const_args)]

//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
//@ compile-flags: -Zvalidate-mir

type const N: usize = "this isn't a usize";
//~^ ERROR the constant `"this isn't a usize"` is not of type `usize`

fn f() -> [u8; const { N }] {}
//[current]~^ ERROR mismatched types [E0308]
//[next]~^^ ERROR type annotations needed
//[next]~| ERROR mismatched types [E0308]

fn main() {}
