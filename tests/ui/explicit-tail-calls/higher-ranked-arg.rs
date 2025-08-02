// Regression test for <https://github.com/rust-lang/rust/issues/144826>.
//@ check-pass

#![feature(explicit_tail_calls)]
//~^ WARN the feature `explicit_tail_calls` is incomplete

fn foo(x: fn(&i32)) {
    become bar(x);
}

fn bar(_: fn(&i32)) {}

fn main() {}
