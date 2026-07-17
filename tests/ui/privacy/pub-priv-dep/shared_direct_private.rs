//@ aux-crate:priv:shared=shared.rs
//@ aux-crate:reexport=reexport.rs
//@ compile-flags: -Zunstable-options

// A shared dependency, where the public side reexports the same item as a
// direct private dependency.
//
//          shared_direct_private
//                  /\
//        (public) /  | (PRIVATE)
//                /   |
//        reexport    |
//                \   |
//        (public) \  /
//                  \/
//                shared
//

#![crate_type = "lib"]
#![deny(exported_private_dependencies)]

extern crate shared;
extern crate reexport;

pub fn leaks_priv() -> shared::Shared {
    //~^ ERROR type `Shared` from private dependency 'shared' in public interface
    shared::Shared
}

pub fn leaks_pub() -> reexport::Shared {
    reexport::Shared
}
