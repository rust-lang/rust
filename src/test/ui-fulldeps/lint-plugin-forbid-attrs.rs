// aux-build:lint-plugin-test.rs
// ignore-stage1

#![feature(plugin)]
#![plugin(lint_plugin_test)]
#![forbid(test_lint)]

fn lintme() { } //~ ERROR item is named 'lintme'

#[allow(test_lint)]
//~^ ERROR allow(test_lint) overruled by outer forbid(test_lint)
pub fn main() {
    lintme();
}
