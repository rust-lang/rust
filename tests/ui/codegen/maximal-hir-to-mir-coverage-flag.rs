//! Test that -Z maximal-hir-to-mir-coverage flag is accepted.
//!
//! Original PR: https://github.com/rust-lang/rust/pull/105286

//@ compile-flags: -Zmaximal-hir-to-mir-coverage
//@ run-pass

fn main() {
    let x = 1;
    let y = x + 1;
    println!("{y}");
}
