//@ aux-crate:priv:reexport=reexport.rs
//@ compile-flags: -Zunstable-options

// Checks the behavior of a reexported item from a private dependency.

#![crate_type = "lib"]
#![deny(exported_private_dependencies)]

extern crate reexport;
use reexport::Shared;

pub fn leaks_priv() -> reexport::Shared {
    //~^ ERROR type `Shared` from private dependency 'reexport' in public interface
    reexport::Shared
}

pub fn leaks_priv_use() -> Shared {
    //~^ ERROR type `Shared` from private dependency 'reexport' in public interface
    Shared
}
