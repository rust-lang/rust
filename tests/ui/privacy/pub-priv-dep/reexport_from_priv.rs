//@ aux-crate:priv:reexport=reexport.rs
//@ compile-flags: -Zunstable-options
//@ check-pass

// Checks the behavior of a reexported item from a private dependency.

#![crate_type = "lib"]
#![deny(exported_private_dependencies)]

extern crate reexport;

// FIXME: This should trigger.
pub fn leaks_priv() -> reexport::Shared {
    reexport::Shared
}
