//@ aux-crate:priv:shared=shared.rs
//@ aux-crate:reexport=reexport.rs
//@ compile-flags: -Zunstable-options
//@ check-pass

// A shared dependency, where a private dependency reexports a public dependency.
//
//         shared_both_private
//                  /\
//       (PRIVATE) /  | (PRIVATE)
//                /   |
//        reexport    |
//                \   |
//        (public) \  /
//                  \/
//                shared

#![crate_type = "lib"]
#![deny(exported_private_dependencies)]

extern crate shared;
extern crate reexport;

// FIXME: This should trigger.
pub fn leaks_priv() -> shared::Shared {
    shared::Shared
}

// FIXME: This should trigger.
pub fn leaks_priv_reexport() -> reexport::Shared {
    reexport::Shared
}
