//@ aux-crate:priv:shared=shared.rs
//@ aux-crate:reexport=reexport.rs
//@ compile-flags: -Zunstable-options
//@ check-pass

// Resolve the public first hop before resolving the same crate as a private direct dependency.
// The direct private spelling must still be diagnosed regardless of load order.

#![crate_type = "lib"]
#![deny(exported_private_dependencies)]

extern crate reexport;
extern crate shared;

pub fn leaks_direct_name() -> shared::Shared {
    shared::Shared
}

pub fn leaks_reexported_name() -> reexport::Shared {
    reexport::Shared
}
