//@ revisions: next old
//@ edition: 2024
//@[next] compile-flags: -Znext-solver
//@ check-pass
#![feature(gen_blocks)]

fn foo() -> impl Iterator<Item = u32> {
    gen { 42.yield }
}

fn bar() -> impl Iterator<Item = i64> {
    gen { 42.yield }
}

fn baz() -> impl Iterator<Item = i32> {
    gen { 42.yield }
}

fn main() {}
