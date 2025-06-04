struct Foo {}

impl<const UNUSED: usize> Drop for Foo {}
//~^ ERROR: `Drop` impl requires `the constant `_` has type `usize``
//~| ERROR: the const parameter `UNUSED` is not constrained by the impl trait, self type, or predicates
//~| ERROR: missing: `drop`

fn main() {}
