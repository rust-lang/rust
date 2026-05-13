//@ compile-flags: -Znext-solver=globally
//@ failure-status: 1

//~? ERROR rustdoc does not support generating auto-trait impls for display with `-Znext-solver=globally`

// Regression test for https://github.com/rust-lang/rust/issues/156487.
struct A;
fn main() {}
