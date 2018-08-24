// aux-build:llvm_pass_plugin.rs
// ignore-stage1

#![feature(plugin)]
#![plugin(llvm_pass_plugin)]

pub fn main() { }
