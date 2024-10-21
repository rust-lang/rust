//@ ignore-enzyme
//@ revisions: std_autodiff no_std_autodiff
//@[no_std_autodiff] check-pass
//@ aux-build: my_macro.rs
#![crate_type = "lib"]
#![feature(autodiff)]

#[cfg(std_autodiff)]
use std::autodiff::autodiff;

extern crate my_macro;
use my_macro::autodiff; // bring `autodiff` in scope

#[autodiff]
//[std_autodiff]~^^^ ERROR the name `autodiff` is defined multiple times
//[std_autodiff]~^^ ERROR this rustc version does not support autodiff
fn foo() {}
