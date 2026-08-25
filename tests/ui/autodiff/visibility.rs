//@ ignore-enzyme
//@ revisions: std_autodiff no_std_autodiff
//@ only-nightly
//@[no_std_autodiff] check-pass
//@ proc-macro: my_macro.rs
#![crate_type = "lib"]
#![feature(autodiff)]

#[cfg(std_autodiff)]
use std::autodiff::autodiff_forward;
extern crate my_macro;
use my_macro::autodiff_forward; // bring `autodiff_forward` in scope

#[autodiff_forward(dfoo)]
//[std_autodiff]~^^^ ERROR the name `autodiff_forward` is defined multiple times
fn foo() {}
