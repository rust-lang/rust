//! Regression test for ICE #139556

//@ revisions: old next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
//@[next] ignore-test: trait-system-refactor-initiative#242

#![feature(type_alias_impl_trait)]

trait T {}

type Alias<'a> = impl T;

struct S;
impl<'a> T for &'a S {}

#[define_opaque(Alias)]
fn with_positive(fun: impl Fn(Alias<'_>)) {
//~^ WARN: function cannot return without recursing
    with_positive(|&n| ());
    //~^ ERROR: cannot move out of a shared reference
}

fn main() {}
