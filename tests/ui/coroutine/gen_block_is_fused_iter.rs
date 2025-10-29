//@ revisions: next old
//@ edition: 2024
//@[next] compile-flags: -Znext-solver
//@ check-pass
#![feature(gen_blocks)]

use std::iter::FusedIterator;

fn foo() -> impl FusedIterator {
    gen { 42.yield }
}

fn bar() -> impl FusedIterator<Item = u16> {
    gen { 42.yield }
}

fn baz() -> impl FusedIterator + Iterator<Item = i64> {
    gen { 42.yield }
}

fn main() {}
