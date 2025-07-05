//@ run-rustfix

#![allow(unused)] // for rustfix

#[derive(Clone, Copy)]
struct S;

trait T {
    fn foo((x, y): (i32, i32)); //~ ERROR patterns aren't allowed in trait methods in the 2015 edition

    fn bar((x, y): (i32, i32)) {} //~ ERROR patterns aren't allowed in trait methods in the 2015 edition
    fn method(S { .. }: S) {} //~ ERROR patterns aren't allowed in trait methods in the 2015 edition

    fn f(&ident: &S) {} // ok
    fn g(&&ident: &&S) {} // ok
    fn h(mut ident: S) {} // ok
}

fn main() {}
