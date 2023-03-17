// aux-build: issue-83510.rs

extern crate issue_83510;

issue_83510::dance_like_you_want_to_ice!();
//~^ ERROR: cannot find type `Foo`
//~| ERROR: expected trait, found struct `Box`
//~| ERROR: cannot find trait `Baz`
//~| ERROR: inherent associated types are unstable

fn main() {}
