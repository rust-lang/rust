//@ compile-flags: -Znext-solver
//@ check-pass

#![feature(const_trait_impl, const_destruct)]

fn foo(_: impl std::marker::Destruct) {}

struct MyAdt;

fn main() {
    foo(1);
    foo(MyAdt);
}
