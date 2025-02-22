//@ aux-crate:priv:shared=shared.rs
//@ aux-crate:priv:reexport=reexport.rs
//@ compile-flags: -Zunstable-options

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

pub fn leaks_priv() -> shared::Shared {
//~^ ERROR type `Shared` from private dependency 'shared' in public interface
    shared::Shared
}

pub fn leaks_priv_reexport() -> reexport::Shared {
//~^ ERROR type `Shared` from private dependency 'shared' in public interface
    reexport::Shared
}
