// rustc-env:RUST_BACKTRACE=0
// normalize-stderr-test: "Clippy version: .*" -> "Clippy version: foo"
// normalize-stderr-test: "internal_lints.rs:\d*:\d*" -> "internal_lints.rs"
// normalize-stderr-test: "', .*clippy_lints" -> "', clippy_lints"

#![deny(clippy::internal)]

fn it_looks_like_you_are_trying_to_kill_clippy() {}

fn main() {}
