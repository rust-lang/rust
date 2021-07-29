// check-pass

#![feature(lint_reasons)]

#![expect(this_lint_does_not_exist)]
//~^ WARNING unknown lint: `this_lint_does_not_exist` [unknown_lints]
fn main() {}
