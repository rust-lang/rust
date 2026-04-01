// Regression test for <https://github.com/rust-lang/rust/issues/120600>
//
//@ edition: 2024
//@ check-pass

#![feature(never_type)]

fn ice(a: !) {
    a == a;
}

fn main() {}
