#![warn(clippy::ignore_without_reason)]

fn main() {}

#[test]
fn unignored_test() {}

#[test]
#[ignore = "Some good reason"]
fn ignored_with_reason() {}

#[test]
#[ignore]
fn ignored_without_reason() {}
