//@ revisions: old next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
//@[next] check-pass
//@[old] failure-status: 101
//@[old] dont-check-compiler-stderr
//@[old] known-bug: #125553
//@ edition:2021

#![feature(type_alias_impl_trait)]

#[derive(Copy, Clone)]
struct Foo(i32);

// This ICEd during MIR building with the old solver. It's fixed by normalizing
// opaque types with the new solver.

fn main() {
    type T = impl Copy;
    let foo: T = Foo(1);
    let x = move || {
        let derive = move || {
            let Foo(a) = foo;
        };
    };
}
