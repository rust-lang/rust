// aux-build:lint-group-plugin-test.rs
// ignore-stage1
// compile-flags: -D lint-me

#![feature(plugin)]
#![plugin(lint_group_plugin_test)]

fn lintme() { } //~ ERROR item is named 'lintme'

fn pleaselintme() { } //~ ERROR item is named 'pleaselintme'

pub fn main() {
    lintme();
    pleaselintme();
}
