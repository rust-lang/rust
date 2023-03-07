// check-pass
// aux-build:lint-plugin-test.rs
// ignore-stage1
// compile-flags: -A test-lint

#![feature(plugin)]
#![plugin(lint_plugin_test)] //~ WARNING compiler plugins are deprecated

fn lintme() { }

pub fn main() {
}
