// error-pattern: no documentation found
// normalize-stderr-test: "nightly|beta|1\.[0-9][0-9]\.[0-9]" -> "$$CHANNEL"
#![deny(rustdoc::missing_crate_level_docs)]
//^~ NOTE defined here

pub fn foo() {}
