#![feature(generic_associated_types)]
//~^ WARNING the feature `generic_associated_types` is incomplete

use std::ops::Deref;

// FIXME(#44265): "lifetime arguments are not allowed for this type" errors will be addressed in a
// follow-up PR.

trait Iterable {
    type Item<'a>;
    type Iter<'a>: Iterator<Item = Self::Item<'a>>
    //~^ ERROR lifetime arguments are not allowed for this type [E0109]
        + Deref<Target = Self::Item<'b>>;
    //~^ ERROR undeclared lifetime
    //~| ERROR lifetime arguments are not allowed for this type [E0109]

    fn iter<'a>(&'a self) -> Self::Iter<'undeclared>;
    //~^ ERROR undeclared lifetime
    //~| ERROR lifetime arguments are not allowed for this type [E0109]
}

fn main() {}
