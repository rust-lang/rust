//@ check-pass
// The test for the ICE 6539: https://github.com/rust-lang/rust-clippy/issues/6539.
// The cause is that `zero_sized_map_values` used `layout_of` with types from type aliases,
// which is essentially the same as the ICE 4968.
// Note that only type aliases with associated types caused the crash this time,
// not others such as trait impls.

use std::collections::{BTreeMap, HashMap};

pub trait Trait {
    type Assoc;
}

type TypeAlias<T> = HashMap<(), <T as Trait>::Assoc>;
type TypeAlias2<T> = BTreeMap<(), <T as Trait>::Assoc>;

fn main() {}
