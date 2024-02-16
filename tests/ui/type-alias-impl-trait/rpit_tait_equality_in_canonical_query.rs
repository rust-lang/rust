//! This tries to prove the APIT's bounds in a canonical query,
//! which doesn't know anything about the defining scope of either
//! opaque type and thus makes a random choice as to which opaque type
//! becomes the hidden type of the other. When we leave the canonical
//! query, we attempt to actually check the defining anchor, but now we
//! have a situation where the RPIT gets constrained outside its anchor.

//@ revisions: current next
//@[next] compile-flags: -Znext-solver
//@[next] check-pass

//@[current] known-bug: #108498
//@[current] failure-status: 101
//@[current] normalize-stderr-test: "DefId\(.*?\]::" -> "DefId("
//@[current] normalize-stderr-test: "(?m)note: we would appreciate a bug report.*\n\n" -> ""
//@[current] normalize-stderr-test: "(?m)note: rustc.*running on.*\n\n" -> ""
//@[current] normalize-stderr-test: "(?m)note: compiler flags.*\n\n" -> ""
//@[current] normalize-stderr-test: "(?m)note: delayed at.*$" -> ""
//@[current] normalize-stderr-test: "(?m)^ *\d+: .*\n" -> ""
//@[current] normalize-stderr-test: "(?m)^ *at .*\n" -> ""

#![feature(type_alias_impl_trait)]

type Opaque = impl Sized;

fn get_rpit() -> impl Clone {}

fn query(_: impl FnOnce() -> Opaque) {}

fn test() -> Opaque {
    query(get_rpit);
    get_rpit()
}

fn main() {}
