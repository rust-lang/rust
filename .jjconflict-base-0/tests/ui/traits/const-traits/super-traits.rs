//@ check-pass
//@ compile-flags: -Znext-solver
#![feature(const_trait_impl)]

const trait Foo {
    fn a(&self);
}

const trait Bar: [const] Foo {}

struct S;
const impl Foo for S {
    fn a(&self) {}
}

const impl Bar for S {}

const fn foo<T: [const] Bar>(t: &T) {
    t.a();
}

const _: () = foo(&S);

fn main() {}
