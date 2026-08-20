//@ aux-build:impls-cycle-90817.rs
//@ check-pass
//! Regression test for <https://github.com/rust-lang/rust/issues/90817>.
//! Checking whether these impls specialize one another used to hit a query
//! cycle (E0391) when the impls live in another crate.

extern crate impls_cycle_90817;

fn main() {
    impls_cycle_90817::reduce(impls_cycle_90817::Higher);
    impls_cycle_90817::combine(());
}
