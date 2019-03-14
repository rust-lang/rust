// aux-build:llvm-pass-plugin.rs
// ignore-stage1

#![feature(plugin)]
#![plugin(llvm_pass_plugin)]

pub fn main() { }
