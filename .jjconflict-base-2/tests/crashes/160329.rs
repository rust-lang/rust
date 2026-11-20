//@ known-bug: #160329
#![feature(autodiff)]
#[core::autodiff::autodiff_forward(fd_inner, Dual)]
fn f(_x: struct S<B, T>) {}
