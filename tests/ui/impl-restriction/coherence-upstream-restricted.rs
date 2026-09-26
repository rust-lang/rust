//@ revisions: classic coherence next
//@[classic] compile-flags: -Znext-solver=no
//@[coherence] compile-flags: -Znext-solver=coherence
//@[next] compile-flags: -Znext-solver=globally
//@ aux-build: coherence-impl-restriction.rs

// The upstream crate can still add impls for these restricted traits.

extern crate coherence_impl_restriction as upstream;

struct Thing;

trait Direct {}
impl<T: upstream::Restricted> Direct for T {}
impl Direct for Thing {}
//~^ ERROR conflicting implementations of trait `Direct` for type `Thing`

trait Fundamental {}
impl<T: upstream::Restricted> Fundamental for T {}
impl Fundamental for &Thing {}
//~^ ERROR conflicting implementations of trait `Fundamental` for type `&Thing`

trait HigherRanked {}
impl<T> HigherRanked for T where for<'a> &'a T: upstream::Restricted {}
impl HigherRanked for Thing {}
//~^ ERROR conflicting implementations of trait `HigherRanked` for type `Thing`

// All these scopes still allow upstream impls.
trait InCrate {}
impl<T: upstream::InCrateRestricted> InCrate for T {}
impl InCrate for Thing {}
//~^ ERROR conflicting implementations of trait `InCrate` for type `Thing`

trait InModule {}
impl<T: upstream::nested::InModuleRestricted> InModule for T {}
impl InModule for Thing {}
//~^ ERROR conflicting implementations of trait `InModule` for type `Thing`

trait InParent {}
impl<T: upstream::nested::InParentRestricted> InParent for T {}
impl InParent for Thing {}
//~^ ERROR conflicting implementations of trait `InParent` for type `Thing`

fn main() {}
