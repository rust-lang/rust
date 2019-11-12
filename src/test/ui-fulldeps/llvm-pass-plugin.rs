// run-pass
// aux-build:llvm-pass-plugin.rs
// ignore-stage1

#![feature(plugin)]
#![plugin(llvm_pass_plugin)] //~ WARNING compiler plugins are deprecated

pub fn main() { }
