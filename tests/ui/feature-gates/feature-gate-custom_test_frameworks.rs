#![test_runner(main)] //~ ERROR custom test frameworks are an unstable feature

#[test_case] //~ ERROR custom test frameworks are an unstable feature
fn f() {}

fn main() {}
