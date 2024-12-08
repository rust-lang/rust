// Regression test for ICE #130012
// Checks that we do not ICE while reporting
// lifetime mistmatch error

trait Fun {
    type Assoc;
}

trait MyTrait: for<'a> Fun<Assoc = &'a ()> {}
//~^ ERROR binding for associated type `Assoc` references lifetime `'a`, which does not appear in the trait input types
//~| ERROR binding for associated type `Assoc` references lifetime `'a`, which does not appear in the trait input types
//~| ERROR binding for associated type `Assoc` references lifetime `'a`, which does not appear in the trait input types

impl<F: for<'b> Fun<Assoc = &'b ()>> MyTrait for F {}
//~^ ERROR binding for associated type `Assoc` references lifetime `'b`, which does not appear in the trait input types
//~| ERROR mismatched types

fn main() {}
