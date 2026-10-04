//@ revisions: classic coherence next
//@[classic] compile-flags: -Znext-solver=no
//@[coherence] compile-flags: -Znext-solver=coherence
//@[next] compile-flags: -Znext-solver=globally

#![feature(impl_restriction)]

impl(crate) trait Restricted {}
trait Unrestricted {}

trait RestrictedCase {}
impl<T: Restricted> RestrictedCase for T {}
impl<T> RestrictedCase for &T {}

trait UnrestrictedCase {}
impl<T: Unrestricted> UnrestrictedCase for T {}
impl<T> UnrestrictedCase for &T {}
//~^ ERROR conflicting implementations of trait `UnrestrictedCase` for type `&_`

fn main() {}
