//@ edition:2015
//@ run-rustfix
// FIXME(fmease): Add historical context.

#![allow(unused)] // for rustfix

#[derive(Clone, Copy)]
struct S;

trait T {
    fn foo((x, y): (i32, i32));
    //~^ ERROR parameters can't have complex patterns in associated functions in traits in Rust 2015
    fn bar((x, y): (i32, i32)) {}
    //~^ ERROR parameters can't have complex patterns in associated functions in traits in Rust 2015
    fn method(S { .. }: S) {}
    //~^ ERROR parameters can't have complex patterns in associated functions in traits in Rust 2015
    fn f(&ident: &S) {}
    //~^ ERROR parameters can't have complex patterns in associated functions in traits in Rust 2015
    fn g(&&ident: &&S) {}
    //~^ ERROR parameters can't have complex patterns in associated functions in traits in Rust 2015
    fn h(mut ident: S) {} // ok
}

fn main() {}
