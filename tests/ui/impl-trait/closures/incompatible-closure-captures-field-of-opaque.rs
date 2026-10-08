//@ revisions: old next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
//@[old] failure-status: 101
//@[old] dont-check-compiler-stderr
//@[old] known-bug: #155497
//@[next] check-pass
//@ compile-flags: -Wrust-2021-incompatible-closure-captures
//@ edition: 2018

#![feature(type_alias_impl_trait)]

// This ICE'd with the old solver the nested closure ended up accessing
// the fields of an opaque type. This caused the lint to then encounter
// an opaque type as the base type, resulting in an ICE.
//
// Regression test for #155497.

struct Foo(i32);

fn main() {
    type T = impl Sized;
    let foo: T = Foo(1);
    let x = move || {
        let Foo(x) = foo;
    };
}
