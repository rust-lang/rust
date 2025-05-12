//@ aux-build:issue-75982.rs
//@ check-pass

// Regression test for issue #75982
// Tests that don't ICE when invoking a foreign macro
// that occurs inside a module with a weird parent.

extern crate issue_75982;

fn main() {
    issue_75982::first_macro!();
    issue_75982::second_macro!();
}
