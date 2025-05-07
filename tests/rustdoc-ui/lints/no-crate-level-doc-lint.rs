//@ normalize-stderr: "nightly|beta|1\.[0-9][0-9]\.[0-9]" -> "$$CHANNEL"
#![deny(rustdoc::missing_crate_level_docs)]
//^~ NOTE defined here

pub fn foo() {}

//~? ERROR no documentation found for this crate's top-level module
