// run-pass
// aux-build:lint-plugin-test.rs
// ignore-stage1
// compile-flags: -Z extra-plugins=lint_plugin_test

#![allow(dead_code)]

fn lintme() { } //~ WARNING item is named 'lintme'

#[allow(test_lint)]
pub fn main() {
    fn lintme() { }
}
