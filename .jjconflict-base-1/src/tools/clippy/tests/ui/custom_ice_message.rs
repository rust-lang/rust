//@rustc-env:RUST_BACKTRACE=0
//@normalize-stderr-test: "Clippy version: .*" -> "Clippy version: foo"
//@normalize-stderr-test: "custom_ice_message.rs:\d*:\d*" -> "custom_ice_message.rs"
//@normalize-stderr-test: "rustc 1\.\d+.* running on .*" -> "rustc <version> running on <target>"
//@normalize-stderr-test: "(?ms)query stack during panic:\n.*end of query stack\n" -> ""
//@normalize-stderr-test: "note: delayed at .*" -> "note: delayed at <location>"
//@normalize-stderr-test: "note: compiler flags: .*" -> "note: compiler flags: <flags>"

#![feature(rustc_attrs)]

#[rustc_delayed_bug_from_inside_query]
fn main() {}
//~^ ice: delayed bug triggered by
