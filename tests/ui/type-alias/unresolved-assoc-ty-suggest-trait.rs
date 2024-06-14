// Ensure that we don't suggest *type alias bounds* for **eager** type aliases.
// issue: rust-lang/rust#125789

//@ revisions: eager lazy
#![cfg_attr(lazy, feature(lazy_type_alias), allow(incomplete_features))]

// FIXME(fmease): Suggest a full trait ref (with placeholders) instead of just a trait name.

trait Trait<T> { type Assoc; }

type AssocOf<T> = T::Assoc; //~ ERROR associated type `Assoc` not found for `T`
//[eager]~^ HELP consider fully qualifying the associated type
//[lazy]~| HELP consider restricting type parameter `T`

type AssokOf<T> = T::Assok; //~ ERROR associated type `Assok` not found for `T`
//[eager]~^ HELP consider fully qualifying and renaming the associated type
//[lazy]~| HELP consider restricting type parameter `T`
//[lazy]~| HELP and changing the associated type name

fn main() {}
