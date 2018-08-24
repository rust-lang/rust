// run-pass
// aux-build:lint_tool_test.rs
// ignore-stage1
#![feature(plugin)]
#![feature(tool_lints)]
#![plugin(lint_tool_test)]
#![allow(dead_code)]

fn lintme() { } //~ WARNING item is named 'lintme'

#[allow(clippy::test_lint)]
pub fn main() {
    fn lintme() { }
}
