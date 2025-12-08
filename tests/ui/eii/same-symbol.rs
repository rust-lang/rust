//@ run-pass
//@ check-run-results
//@ ignore-backends: gcc
#![feature(extern_item_impls)]

pub mod a {
    #[eii(foo)]
    pub fn foo();
}

pub mod b {
    #[eii(foo)]
    pub fn foo();
}

#[a::foo]
fn a_foo_impl() {
    println!("foo1");
}

#[b::foo]
fn b_foo_impl() {
    println!("foo2");
}

fn main() {
    a::foo();
    b::foo();
}
