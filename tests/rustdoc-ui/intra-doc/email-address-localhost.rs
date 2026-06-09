//@ normalize-stderr: "nightly|beta|1\.[0-9][0-9]\.[0-9]" -> "$$CHANNEL"
//@ check-pass
#![deny(warnings)]

//! Email me at <hello@localhost>.

//! This should *not* warn: <hello@example.com>.
