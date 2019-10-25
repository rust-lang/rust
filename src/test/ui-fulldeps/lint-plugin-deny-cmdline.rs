// aux-build:lint-plugin-test.rs
// ignore-stage1
// compile-flags: -D test-lint

#![feature(plugin)]
#![plugin(lint_plugin_test)]

fn lintme() { } //~ ERROR item is named 'lintme'

pub fn main() {
    lintme();
}
