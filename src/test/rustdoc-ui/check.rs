// check-pass
// compile-flags: -Z unstable-options --check
// normalize-stderr-test: "nightly|beta|1\.[0-9][0-9]\.[0-9]" -> "$$CHANNEL"

#![warn(missing_docs)]
//~^ WARN
//~^^ WARN
#![warn(rustdoc::all)]

pub fn foo() {}
//~^ WARN
//~^^ WARN
