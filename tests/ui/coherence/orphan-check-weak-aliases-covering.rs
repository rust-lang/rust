// Weak aliases cover type parameters if they normalize to a (local) type that covers them.

//@ check-pass
//@ revisions: classic next
//@[next] compile-flags: -Znext-solver

//@ aux-crate:foreign=parametrized-trait.rs
//@ edition:2021

#![feature(lazy_type_alias)]
#![allow(incomplete_features)]

type Alias<T> = LocalWrapper<T>;

struct Local;
struct LocalWrapper<T>(T);

impl<T> foreign::Trait1<Local, T> for Alias<T> {}

fn main() {}
