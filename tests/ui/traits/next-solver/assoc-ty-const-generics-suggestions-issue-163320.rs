// The const-parameter environment fix must retain applicable associated-type
// suggestions without suggesting an impl for an incompatible projection.

//@ compile-flags: -Znext-solver=globally
//@ check-fail

trait Identity {
    type Out;
}

impl Identity for () {
    type Out = (u8, u8);
}

trait Different {
    type Out;
}

impl Different for () {
    type Out = (u8, u16);
}

trait HasItem {
    type Item;
}

impl<T> HasItem for (T, T) {
    type Item = ();
}

type Positive = <<() as Identity>::Out>::Item;
//~^ ERROR ambiguous associated type

type Negative = <<() as Different>::Out>::Item;
//~^ ERROR ambiguous associated type

fn main() {}
