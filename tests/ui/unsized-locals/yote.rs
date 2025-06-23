//@ normalize-stderr: "you are using [0-9]+\.[0-9]+\.[0-9]+(-[a-zA-Z0-9.]+)?( \([^)]*\))?" -> "you are using $$RUSTC_VERSION"

#![feature(unsized_locals)] //~ERROR feature has been removed
#![crate_type = "lib"]
