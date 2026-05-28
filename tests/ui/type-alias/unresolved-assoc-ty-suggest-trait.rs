// Ensure that we don't suggest *type alias bounds* for **eager** type aliases.
// issue: rust-lang/rust#125789

//@ revisions: eager lazy
#![cfg_attr(lazy, feature(lazy_type_alias), allow(incomplete_features))]

trait Trait { type Assoc; }

type AssocOf<T> = T::Assoc; //~ ERROR associated type `Assoc` not found for `T`
//[eager]~^ HELP consider fully qualifying the associated type
//[lazy]~| HELP consider restricting type parameter `T`

type AssokOf<T> = T::Assok; //~ ERROR associated type `Assok` not found for `T`
//[eager]~^ HELP consider fully qualifying and renaming the associated type
//[lazy]~| HELP consider restricting type parameter `T`
//[lazy]~| HELP and changing the associated type name

trait Parametrized<'a, T, const N: usize> {
    type Proj;
}

type ProjOf<T> = T::Proj; //~ ERROR associated type `Proj` not found for `T`
//[eager]~^ HELP consider fully qualifying the associated type
//[lazy]~| HELP consider restricting type parameter `T`

fn main() {}
