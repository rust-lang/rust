// Ensure that we validate *all* `#![crate_name]`s on print requests, not just the first,
// and that we reject macro calls inside of them.
// See also <https://github.com/rust-lang/rust/issues/122001>.

//@ compile-flags: --print=file-names
#![crate_name = "this_one_is_okay"]
#![crate_name = concat!("this_one_is_not")] //~ ERROR attribute value must be a literal
