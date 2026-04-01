// regression test for https://github.com/rust-lang/rust/issues/10656
#![deny(missing_docs)]
//~^ ERROR missing documentation for the crate
#![crate_type = "lib"]
