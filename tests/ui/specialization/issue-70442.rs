#![feature(specialization)] //~ WARN the feature `specialization` is incomplete

//@ check-pass

trait Trait {
    type Assoc;
}

impl<T> Trait for T {
    default type Assoc = bool;
}

// This impl inherits the `Assoc` definition from above and "locks it in", or finalizes it, making
// child impls unable to further specialize it. However, since the specialization graph didn't
// correctly track this, we would refuse to project `Assoc` from this impl, even though that should
// happen for items that are final.
impl Trait for () {}

fn foo<X: Trait<Assoc=bool>>() {}

fn main() {
    foo::<()>();  // `<() as Trait>::Assoc` is normalized to `bool` correctly
}
