// rust-lang/rust#57979 : the initial support for `impl Trait` didn't
// properly check syntax hidden behind an associated type projection,
// but it did catch *some cases*. This is checking that we continue to
// properly emit errors for those.
//
// issue-57979-nested-impl-trait-in-assoc-proj.rs shows the main case
// that we were previously failing to catch.

struct Deeper<T>(T);

pub trait Foo<T> { }
pub trait Bar { }
pub trait Quux { type Assoc; }
pub fn demo(_: impl Quux<Assoc=Deeper<impl Foo<impl Bar>>>) { }
//~^ ERROR nested `impl Trait` is not allowed

fn main() { }
