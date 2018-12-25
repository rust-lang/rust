// aux-build:lint_plugin_test.rs
// ignore-stage1

#![feature(plugin)]
#![plugin(lint_plugin_test)]
#![deny(test_lint)]

fn lintme() { } //~ ERROR item is named 'lintme'

pub fn main() {
    lintme();
}
