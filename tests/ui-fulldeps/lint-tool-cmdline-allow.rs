// check-pass
// aux-build:lint-tool-test.rs
// ignore-stage1
// compile-flags: -A test-lint

#![feature(plugin)]
#![plugin(lint_tool_test)] //~ WARNING compiler plugins are deprecated

fn lintme() {}
//~^ WARNING item is named 'lintme' [clippy::test_lint]

pub fn main() {}
