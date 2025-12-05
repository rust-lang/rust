// Regression test for <https://github.com/rust-lang/rust/issues/144826>.
//@ check-pass

#![feature(explicit_tail_calls)]
#![expect(incomplete_features)]

fn foo(x: fn(&i32)) {
    become bar(x);
}

fn bar(_: fn(&i32)) {}

fn main() {}
