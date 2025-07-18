// Make sure we don't issue *two* error messages for the trait predicate *and* host predicate.

#![feature(const_trait_impl)]

#[const_trait]
trait Trait {
  type Out;
}

const fn needs_const<T: [const] Trait>(_: &T) {}

const IN_CONST: () = {
  needs_const(&());
  //~^ ERROR the trait bound `(): Trait` is not satisfied
};

const fn conditionally_const() {
  needs_const(&());
  //~^ ERROR the trait bound `(): Trait` is not satisfied
}

fn main() {}
