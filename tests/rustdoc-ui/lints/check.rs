//@ check-pass
//@ compile-flags: -Z unstable-options --check
//@ normalize-stderr: "nightly|beta|1\.[0-9][0-9]\.[0-9]" -> "$$CHANNEL"

#![feature(rustdoc_missing_doc_code_examples)]
//~^ WARN

#![warn(missing_docs)]
#![warn(rustdoc::missing_doc_code_examples)]
#![warn(rustdoc::all)]

pub fn foo() {}
//~^ WARN
//~^^ WARN

//~? WARN no documentation found for this crate's top-level module
