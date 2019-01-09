// Check that we get an error when you use `<Self as Get>::Value` in
// the trait definition but `Self` does not, in fact, implement `Get`.
//
// See also associated-types-no-suitable-supertrait.rs, which checks
// that we see the same error when making this mistake on an impl
// rather than the default method impl.
//
// See also run-pass/associated-types-projection-to-unrelated-trait.rs,
// which checks that the trait interface itself is not considered an
// error as long as all impls satisfy the constraint.

trait Get {
    type Value;
}

trait Other {
    fn uhoh<U:Get>(&self, foo: U, bar: <Self as Get>::Value) {}
    //~^ ERROR the trait bound `Self: Get` is not satisfied
}

fn main() { }
