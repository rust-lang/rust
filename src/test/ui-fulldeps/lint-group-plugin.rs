// run-pass
// aux-build:lint-group-plugin-test.rs
// ignore-stage1

#![feature(plugin)]
#![plugin(lint_group_plugin_test)]
#![allow(dead_code)]

fn lintme() { } //~ WARNING item is named 'lintme'
fn pleaselintme() { } //~ WARNING item is named 'pleaselintme'

#[allow(lint_me)]
pub fn main() {
    fn lintme() { }

    fn pleaselintme() { }
}
