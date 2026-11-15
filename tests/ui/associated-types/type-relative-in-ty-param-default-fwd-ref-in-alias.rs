// Demonstrate that we don't detect / care about implicit forward references to parameters if they
// vanish "early enough" (i.e., during HIR ty lowering). Unchecked type aliases get expanded during
// HIR ty lowering and our check for implicit forward references is done on the lowered type where
// the `Discard<N>` has already vanished, so we only ever get to see `<T as Trait<()>>::Type` which
// doesn't contain any forward references.
//
// Checked type aliases on the other hand don't vanish during HIR ty lowering but lower to free
// alias types making them visible to our check.
//
// FIXME(checked_type_aliases): This discrepancy is unfortunate. Either expand away free alias types
// before passing it to the aforementioned check or accept breakage once/if retained type aliases
// become the default.
//
//@ revisions: unchecked checked
//@[unchecked] check-pass
#![cfg_attr(checked, feature(checked_type_aliases))]

struct Def0<T: Trait<Discard<N>>, U = T::Type, const N: usize = 0>(T, U);
//[checked]~^ ERROR type parameter default references generic parameter `N` before it is declared
struct Use0(Def0<()>);

type Discard<const N: usize> = ();

trait Trait<T> { type Type; }
impl<T, U> Trait<U> for T { type Type = (T, U); }

fn main() {}
