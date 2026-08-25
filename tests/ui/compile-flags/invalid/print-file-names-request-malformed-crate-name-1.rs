// Ensure we validate `#![crate_name]` on print requests.

//@ compile-flags: --print=file-names
#![crate_name]  //~ ERROR malformed `crate_name` attribute input
