//@ proc-macro: issue-83510.rs
//@ ignore-backends: gcc

extern crate issue_83510;

issue_83510::dance_like_you_want_to_ice!();
//~^ ERROR: cannot find type `Foo` in this scope
//~| ERROR: expected trait, found struct `Box`
//~| ERROR: cannot find trait `Baz` in this scope
//~| ERROR: inherent associated types are unstable

fn main() {}
