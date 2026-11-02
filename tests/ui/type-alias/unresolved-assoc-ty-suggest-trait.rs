// Ensure that we don't suggest *type alias bounds* for **unchecked** type aliases.
// issue: <https://github.com/rust-lang/rust/issues/125789>

//@ revisions: unchecked checked
#![cfg_attr(checked, feature(checked_type_aliases))]

trait Trait { type Assoc; }

type AssocOf<T> = T::Assoc; //~ ERROR associated type `Assoc` not found for `T`
//[unchecked]~^ HELP consider fully qualifying the associated type
//[checked]~| HELP consider restricting type parameter `T`

type AssokOf<T> = T::Assok; //~ ERROR associated type `Assok` not found for `T`
//[unchecked]~^ HELP consider fully qualifying and renaming the associated type
//[checked]~| HELP consider restricting type parameter `T`
//[checked]~| HELP and changing the associated type name

trait Parametrized<'a, T, const N: usize> {
    type Proj;
}

type ProjOf<T> = T::Proj; //~ ERROR associated type `Proj` not found for `T`
//[unchecked]~^ HELP consider fully qualifying the associated type
//[checked]~| HELP consider restricting type parameter `T`

fn main() {}
