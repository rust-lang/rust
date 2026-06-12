// Hint: expected_specific_argument
#![feature(cfg_version)]
#[cfg(does_not_exist())]
fn test() {}

fn main() {}
