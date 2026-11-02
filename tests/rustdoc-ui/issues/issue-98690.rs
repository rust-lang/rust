//@ compile-flags: --test --persist-doctests /../../ -Z unstable-options
//@ failure-status: 101
//@ only-linux
//@ normalize-stdout: "finished in \d+\.\d+s" -> "finished in $$TIME"
//@ normalize-stdout: ".rs:\d+:\d+" -> ".rs:$$LINE:$$COL"

#![crate_name = "foo"]

//! ```rust
//! use foo::dummy;
//! dummy();
//! ```
