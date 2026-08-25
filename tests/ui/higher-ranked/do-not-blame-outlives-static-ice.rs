//@ compile-flags: -Zdeduplicate-diagnostics=yes

// Regression test for #146467.
trait Trait { type Assoc; }

impl Trait for fn(&()) { type Assoc = (); }

fn f(_: for<'a> fn(<fn(&'a ()) as Trait>::Assoc)) {}
//~^ ERROR implementation of `Trait` is not general enough
//~| ERROR higher-ranked subtype error

fn main() {}
