// Regression test for #102800
//
// Here we are generating higher-ranked region constraints when normalizing and relating closure
// input types. Previously this was an ICE in the error path because we didn't register enough
// diagnostic information to render the higher-ranked subtyping error.

//@ check-fail

trait Trait {
    type Ty;
}

impl Trait for &'static () {
    type Ty = ();
}

fn main() {
    let _: for<'a> fn(<&'a () as Trait>::Ty) = |_| {};
    //~^ ERROR implementation of `Trait` is not general enough
    //~| ERROR implementation of `Trait` is not general enough
}
