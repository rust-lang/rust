//@ compile-flags: --force-warn pub_use_of_private_extern_crate
//@ check-pass

extern crate core;
pub use core as reexported_core;
//~^ warning: extern crate `core` is private
//~| warning: this was previously accepted by the compiler

fn main() {}
