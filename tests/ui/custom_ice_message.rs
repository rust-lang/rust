// exec-env:RUST_BACKTRACE=0
// normalize-stderr-test: "Clippy version: .*" -> "Clippy version: foo"
// normalize-stderr-test: "internal_lints.rs.*" -> "internal_lints.rs:1:1"

#![deny(clippy::internal)]

fn should_trigger_an_ice_in_clippy() {}

fn main() {}
