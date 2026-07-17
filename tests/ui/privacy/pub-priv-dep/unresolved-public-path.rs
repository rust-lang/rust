//@ aux-crate:priv:shared=shared.rs
//@ aux-crate:reexport=reexport.rs
//@ compile-flags: -Zunstable-options

// `reexport` provides a public path to `shared`, but this occurrence explicitly selects the
// private `shared` root. An unused public path cannot change another path's provenance.

#![crate_type = "lib"]
#![deny(exported_private_dependencies)]

extern crate shared;

pub fn leaks_private() -> shared::Shared {
    //~^ ERROR type `Shared` from private dependency 'shared' in public interface
    shared::Shared
}
