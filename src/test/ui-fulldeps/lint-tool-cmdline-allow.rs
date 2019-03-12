// run-pass
// aux-build:lint-tool-test.rs
// ignore-stage1
// compile-flags: -A test-lint

#![feature(plugin)]
#![warn(unused)]
#![plugin(lint_tool_test)]

fn lintme() {}

pub fn main() {}
