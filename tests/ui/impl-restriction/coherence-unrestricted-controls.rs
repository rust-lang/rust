//@ check-pass
//@ revisions: classic next
//@[classic] compile-flags: -Znext-solver=no
//@[next] compile-flags: -Znext-solver=globally
//@ aux-build: coherence-impl-restriction.rs

#![feature(impl_restriction)]

extern crate coherence_impl_restriction as upstream;

struct Thing;

// Unrestricted upstream traits still allow negative reasoning here.
trait Direct {}
impl<T: upstream::Unrestricted> Direct for T {}
impl Direct for Thing {}

trait Fundamental {}
impl<T: upstream::Unrestricted> Fundamental for T {}
impl Fundamental for &Thing {}

trait HigherRanked {}
impl<T> HigherRanked for T where for<'a> &'a T: upstream::Unrestricted {}
impl HigherRanked for Thing {}

// Local restrictions cannot introduce unknown upstream impls.
impl(crate) trait LocalRestricted {}
trait Local {}
impl<T: LocalRestricted> Local for T {}
impl Local for Thing {}

mod nested {
    pub impl(self) trait RestrictedHere {}
}
trait LocallyScoped {}
impl<T: nested::RestrictedHere> LocallyScoped for T {}
impl LocallyScoped for Thing {}

// An impossible supertrait still rules out the restricted impl.
trait ImpossibleSupertrait {}
impl<T: upstream::RestrictedWithSupertrait> ImpossibleSupertrait for T {}
impl ImpossibleSupertrait for Thing {}

// An unrestricted subtrait keeps the usual coherence rules.
trait Indirect {}
impl<T: upstream::OpenWithRestrictedSupertrait> Indirect for T {}
impl Indirect for Thing {}

// Fundamental traits keep their existing coherence rules.
trait OpenFundamental {}
impl<T: upstream::FundamentalOpen> OpenFundamental for T {}
impl OpenFundamental for Thing {}

trait RestrictedFundamental {}
impl<T: upstream::FundamentalRestricted> RestrictedFundamental for T {}
impl RestrictedFundamental for Thing {}

// Known impls still work in ordinary trait selection.
fn require_known_restricted<T: upstream::KnownRestricted>() {}
fn require_supertrait<T: upstream::RestrictedWithSupertrait>() {}

fn main() {
    require_known_restricted::<u8>();
    require_supertrait::<u16>();
}
