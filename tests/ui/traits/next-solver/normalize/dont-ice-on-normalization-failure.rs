//@compile-flags: -Znext-solver

// Regression test for #151308

#![feature(checked_type_aliases)]
trait Trait {
    type Associated;
}

trait Generic<T> {}

type TraitObject = dyn Generic<<i32 as Trait>::Associated>;
//~^ ERROR: the trait bound `i32: Trait` is not satisfied

struct Wrap(TraitObject);
//~^ ERROR: type mismatch resolving `TraitObject == _`

fn cast(x: *mut Wrap) {
    x as *mut Wrap;
}

fn main() {}
