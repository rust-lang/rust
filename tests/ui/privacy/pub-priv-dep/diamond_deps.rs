//@ aux-crate:priv:diamond_priv_dep=diamond_priv_dep.rs
//@ aux-crate:diamond_pub_dep=diamond_pub_dep.rs
//@ compile-flags: -Zunstable-options

// A diamond dependency:
//
//           diamond_reepxort
//                  /\
//        (public) /  \ (PRIVATE)
//                /    \
//   diamond_pub_dep  diamond_priv_dep
//                \    /
//        (public) \  /  (public)
//                  \/
//                shared
//
// Where the pub and private crates reexport something from the shared crate.
//
// Checks the behavior when the same shared item appears in the public API,
// depending on whether it comes from the public side or the private side.
//
// NOTE: compiletest does not support deduplicating shared dependencies.
// However, it should work well enough for this test, the only downside is
// that diamond_shared gets built twice.

#![crate_type = "lib"]
#![deny(exported_private_dependencies)]

extern crate diamond_priv_dep;
extern crate diamond_pub_dep;

// FIXME: This should trigger.
pub fn leaks_priv() -> diamond_priv_dep::Shared {
    diamond_priv_dep::Shared
}

pub fn leaks_pub() -> diamond_pub_dep::Shared {
    diamond_pub_dep::Shared
}

pub struct PrivInStruct {
    pub f: diamond_priv_dep::SharedInType
//~^ ERROR type `diamond_priv_dep::SharedInType` from private dependency 'diamond_priv_dep' in public interface
}

pub struct PubInStruct {
    pub f: diamond_pub_dep::SharedInType
}
