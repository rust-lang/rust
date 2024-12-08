//@ check-pass


trait Trait {
    type Type;

    // Check that we're adjusting bound vars correctly when installing the default
    // method projection assumptions.
    fn method(&self) -> impl Trait<Type = impl Sized + '_>;
}

trait Trait2 {
    type Type;

    fn method(&self) -> impl Trait2<Type = impl Trait2<Type = impl Sized + '_> + '_>;
}

fn main() {}
