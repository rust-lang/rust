//@ check-pass

// Ensure `rustdoc::all` only affects stable lints. See #106289.

#![deny(unknown_lints)]
#![allow(rustdoc::all)]
