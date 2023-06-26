#![feature(non_lifetime_binders)]
//~^ WARN the feature `non_lifetime_binders` is incomplete

trait Trait<Input> {
    type Assoc;
}

fn uwu(_: impl for<T> Trait<(), Assoc = impl Trait<T>>) {}
//~^ ERROR `impl Trait` can only mention type parameters from an fn or impl

fn main() {}
