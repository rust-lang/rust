//! Ensure `-Clink-dead-code=true` with `generic_const_exprs` and
//! `min_generic_const_args` doesn't ICE in mono item collection.
//!
//! Regression test for <https://github.com/rust-lang/rust/issues/149035>.

//@ build-pass
//@ compile-flags: -Clink-dead-code=true

#![feature(min_generic_const_args, generic_const_exprs)]

type const L: usize = 4;
trait Print<const N: usize> {
    fn print() -> usize {
        N
    }
}
struct Printer;
impl Print<L> for Printer {}

fn main() {}
