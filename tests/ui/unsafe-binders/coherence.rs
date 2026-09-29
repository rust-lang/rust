//@ compile-flags: -Znext-solver

// Lifetime names on unsafe binders should be ignored for coherence. Impls of
// foreign traits for unsafe binders should be rejected. And, inherent impls
// on unsafe binders should be rejected.

#![feature(unsafe_binders)]
#![allow(incomplete_features, dead_code)]

struct Local;

trait Tr {}
impl Tr for unsafe<'a> &'a Local {}
impl Tr for unsafe<'b> &'b Local {}
//~^ ERROR conflicting implementations of trait `Tr` for type `unsafe<'a> &'a Local`
impl Tr for &'static Local {}
impl<'x> Tr for unsafe<> &'x Local {}
//~^ WARN conflicting implementations of trait `Tr` for type `unsafe<'a> &'a Local`
//~| WARN the behavior may change in a future release

impl Clone for unsafe<'a> &'a Local {
    //~^ ERROR only traits defined in the current crate can be implemented for arbitrary types
    fn clone(&self) -> Self {
        todo!()
    }
}

impl unsafe<'a> &'a Local {}
//~^ ERROR cannot define inherent `impl` for primitive types

fn main() {}
