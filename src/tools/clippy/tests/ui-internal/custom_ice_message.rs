//@rustc-env:RUST_BACKTRACE=0
//@normalize-stderr-test: "Clippy version: .*" -> "Clippy version: foo"
//@normalize-stderr-test: "produce_ice.rs:\d*:\d*" -> "produce_ice.rs"
//@normalize-stderr-test: "', .*clippy_lints" -> "', clippy_lints"
//@normalize-stderr-test: "'rustc'" -> "'<unnamed>'"
//@normalize-stderr-test: "rustc 1\.\d+.* running on .*" -> "rustc <version> running on <target>"
//@normalize-stderr-test: "(?ms)query stack during panic:\n.*end of query stack\n" -> ""

#![deny(clippy::internal)]
#![allow(clippy::missing_clippy_version_attribute)]

fn it_looks_like_you_are_trying_to_kill_clippy() {}

fn main() {}
