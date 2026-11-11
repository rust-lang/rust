// Regression test for #110899.
// When looking for the elided lifetime for `wrap`,
// we must not consider the lifetimes in `bar` as candidates.

fn wrap(self: Wrap<{ fn bar(&self) {} }>) -> &() {
    //~^ ERROR `self` parameter is only allowed in associated functions
    //~| ERROR `self` parameter is only allowed in associated functions
    //~| ERROR missing lifetime specifier
    //~| ERROR cannot find type `Wrap` in this scope
    &()
}

fn main() {}
