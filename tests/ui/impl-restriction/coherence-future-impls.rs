//@ revisions: classic coherence next
//@[classic] compile-flags: -Znext-solver=no
//@[coherence] compile-flags: -Znext-solver=coherence
//@[next] compile-flags: -Znext-solver=globally
//@ aux-build: coherence-future-impls.rs

extern crate coherence_future_impls as upstream;

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

// The coherence opt-in is independent of impl restrictions.
struct DirectImpl;
impl upstream::Unrestricted for DirectImpl {}

struct Other;
trait UnrestrictedMarked {}
impl<T: upstream::Unrestricted> UnrestrictedMarked for T {}
impl UnrestrictedMarked for Other {}
//~^ ERROR conflicting implementations of trait `UnrestrictedMarked` for type `Other`

// An impl restriction alone does not reserve arbitrary future upstream impls.
struct Control;
trait UnmarkedRestricted {}
impl<T: upstream::UnmarkedRestricted> UnmarkedRestricted for T {}
impl UnmarkedRestricted for Control {}

fn main() {}
