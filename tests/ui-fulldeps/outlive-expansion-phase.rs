// run-pass
// aux-build:outlive-expansion-phase.rs
// ignore-stage1

#![feature(plugin)]
#![plugin(outlive_expansion_phase)] //~ WARNING compiler plugins are deprecated

pub fn main() {}
