// In this fn, the type `F` is a function that takes a reference to a
// struct and returns another reference with the same lifetime.
//
// Meanwhile, the bare fn `foo` takes a reference to a struct with
// *ANY* lifetime and returns a reference with the 'static lifetime.
// This can safely be considered to be an instance of `F` because all
// lifetimes are sublifetimes of 'static.
//
//@ check-pass

#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(non_snake_case)]

struct S;

// Given 'cx, return 'cx
type F = for<'cx> fn(&'cx S) -> &'cx S;
fn want_F(f: F) {}

// Given anything, return 'static
type G = for<'cx> fn(&'cx S) -> &'static S;
fn want_G(f: G) {}

// Should meet both.
fn foo(x: &S) -> &'static S {
    panic!()
}

// Should meet both.
fn bar<'a, 'b>(x: &'a S) -> &'b S {
    panic!()
}

// Meets F, but not G.
fn baz(x: &S) -> &S {
    panic!()
}

fn supply_F() {
    want_F(foo);

    want_F(bar);

    want_F(baz);
}

pub fn main() {}
