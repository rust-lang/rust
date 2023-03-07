// aux-build:lint-plugin-test.rs
// ignore-stage1

#![feature(plugin)]
#![plugin(lint_plugin_test)]
//~^ WARN use of deprecated attribute `plugin`
#![deny(test_lint)]

fn lintme() { } //~ ERROR item is named 'lintme'

pub fn main() {
    lintme();
}
