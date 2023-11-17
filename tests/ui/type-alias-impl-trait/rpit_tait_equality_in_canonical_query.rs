//! This tries to prove the APIT's bounds in a canonical query,
//! which doesn't know anything about the defining scope of either
//! opaque type and thus makes a random choice as to which opaque type
//! becomes the hidden type of the other. When we leave the canonical
//! query, we attempt to actually check the defining anchor, but now we
//! have a situation where the RPIT gets constrained outside its anchor.

// revisions: current next
//[next] compile-flags: -Ztrait-solver=next
//[next] check-pass

//[current] known-bug: #108498
//[current] failure-status: 101
//[current] normalize-stderr-test: "DefId\(.*?\]::" -> "DefId("
//[current] normalize-stderr-test: "(?m)note: .*$" -> ""
//[current] normalize-stderr-test: "(?m)^ *\d+: .*\n" -> ""
//[current] normalize-stderr-test: "(?m)^ *at .*\n" -> ""

#![feature(type_alias_impl_trait)]

mod helper {
    pub type Opaque = impl Sized;

    pub fn get_rpit() -> impl Clone {}

    fn test() -> Opaque {
        super::query(get_rpit);
        get_rpit()
    }
}

use helper::*;

fn query(_: impl FnOnce() -> Opaque) {}

fn main() {}
