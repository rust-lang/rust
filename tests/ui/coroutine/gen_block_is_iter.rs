// revisions: next old
//compile-flags: --edition 2024 -Zunstable-options
//[next] compile-flags: -Ztrait-solver=next
// check-pass
#![feature(coroutines)]

fn foo() -> impl Iterator<Item = u32> {
    gen { yield 42 }
}

fn bar() -> impl Iterator<Item = i64> {
    gen { yield 42 }
}

fn baz() -> impl Iterator<Item = i32> {
    gen { yield 42 }
}

fn main() {}
