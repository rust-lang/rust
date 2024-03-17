//@ aux-crate:priv:shared=shared.rs
//@ aux-crate:priv:b=b.rs
//@ compile-flags: -Zunstable-options
//@ check-pass

// A shared dependency, where it is only indirectly public.
//
//            shared_indirect
//                  /\
//       (PRIVATE) /  | (PRIVATE)
//                /   |
//          b    |    |
//     (PRIVATE) |    |
//          c    |    |
//                \   |
//        (public) \  /
//                  \/
//                shared

#![crate_type = "lib"]
#![deny(exported_private_dependencies)]

extern crate shared;
extern crate b;

// FIXME: This should trigger.
pub fn leaks_priv() -> shared::Shared {
    shared::Shared
}
