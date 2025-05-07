//@ compile-flags:--test --test-args --test-threads=1
//@ failure-status: 101
//@ normalize-stdout: "tests/rustdoc-ui/doctest" -> "$$DIR"
//@ normalize-stdout: "finished in \d+\.\d+s" -> "finished in $$TIME"
//@ normalize-stdout: ".rs:\d+:\d+" -> ".rs:$$LINE:$$COL"

//! ```
#![doc = "#![all\
ow(unused)]"]
//! ```
//!
//! ```
#![doc = r#"#![all\
ow(unused)]"#]
//! ```
