//@ aux-crate:priv:shared=shared.rs
//@ aux-crate:reexport=reexport.rs
//@ compile-flags: -Zunstable-options
//@ check-pass

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

// FIXME: Should this trigger?
//
// One could make an argument that I said I want "reexport" to be public, and
// since "reexport" says "shared_direct_private" is public, then it should
// transitively be public for me. However, as written, this is explicitly
// referring to a dependency that is marked "private", which I think is
// confusing.
pub fn leaks_priv() -> shared::Shared {
    shared::Shared
}

pub fn leaks_pub() -> reexport::Shared {
    reexport::Shared
}
