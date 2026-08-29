//@ compile-flags:--test
//@ normalize-stdout: "finished in \d+\.\d+s" -> "finished in $$TIME"
//@ check-pass

//! ```
//! #![crate_name="asdf"]
//!
//! println!("yo");
//! ```
