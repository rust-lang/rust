//@ check-pass
//@ aux-build:external-macro-issue-148114.rs

// This test ensures we do not trigger the lint on external macros
// ref. <https://github.com/rust-lang/rust/issues/148114>

#![deny(for_loops_over_fallibles)]

extern crate external_macro_issue_148114 as dep;

fn main() {
    let _name = Some(1);
    dep::do_loop!(_name);
}
