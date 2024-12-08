//@ aux-crate:priv:shared=shared.rs
//@ aux-crate:priv:indirect1=indirect1.rs
//@ compile-flags: -Zunstable-options
//@ check-pass

// A shared dependency, where it is only indirectly public.
//
//            shared_indirect
//                  /\
//       (PRIVATE) /  | (PRIVATE)
//                /   |
//     indirect1 |    |
//     (PRIVATE) |    |
//     indirect2 |    |
//                \   |
//        (public) \  /
//                  \/
//                shared

#![crate_type = "lib"]
#![deny(exported_private_dependencies)]

extern crate shared;
extern crate indirect1;

// FIXME: This should trigger.
pub fn leaks_priv() -> shared::Shared {
    shared::Shared
}
