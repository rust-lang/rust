// Regression test for <https://github.com/rust-lang/rust/issues/165690>.
//
//@ edition: 2021..
#![feature(forced_keywords)]

fn foo(S k#self) {}
//~^ ERROR expected one of `:`, `@`, or `|`, found keyword `k#self`

fn bar(S k#not_a_keyword) {}
//~^ ERROR expected one of `:`, `@`, or `|`, found `k#not_a_keyword`

fn main() {}
