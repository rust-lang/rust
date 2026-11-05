// RFC 3791: `-Zcrate-attr` applies before the crate's own attributes, so `warn` here wins.
//@ check-pass
//@ compile-flags: -Zcrate-attr=deny(unused)

#![warn(unused)]

fn foo() {} //~ WARN function `foo` is never used

fn main() {}
