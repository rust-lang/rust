// Ensure we validate `#![crate_name]` on print requests and reject macro calls inside of it.
// See also <https://github.com/rust-lang/rust/issues/122001>.

//@ compile-flags: --print=crate-name
#![crate_name = concat!("wrapped")] //~ ERROR attribute value must be a literal
