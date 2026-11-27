//@ compile-flags: -Znext-solver
//@ check-pass

// Previously, when building MIR body, we were using typing env
// `non_body_analysis` which is wrong since we're indeed in a body.
// It caused opaque types in defining body to be not revealed.
// This caused opaque types in the defining body not to be revealed.

#![feature(type_alias_impl_trait)]
fn main() {
    struct Foo(U);
    type U = impl Copy;
    let foo: _ = Foo(());
    let Foo(()) = foo;
}
