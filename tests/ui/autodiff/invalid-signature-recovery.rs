//! Regression test for <https://github.com/rust-lang/rust/issues/160329>.
//! Parser recovery left the function with more parameters than activities,
//! and expansion used to ICE instead of stopping at the count error.

#![feature(autodiff)]

#[core::autodiff::autodiff_forward(fd_inner, Dual)]
//~^ ERROR expected 2 activities, but found 1
fn f(_x: struct S<B, T>) {}
//~^ ERROR expected type, found keyword `struct`
//~| ERROR expected one of `:`, `@`, or `|`, found `>`

fn main() {}
