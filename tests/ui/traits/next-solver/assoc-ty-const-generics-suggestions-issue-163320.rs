// Item lowering should not need trait solving for suggestions, while body
// type checking should retain applicable suggestions.

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

fn body_positive() {
    let _: <<() as Identity>::Out>::Item;
    //~^ ERROR ambiguous associated type
}

fn body_negative() {
    let _: <<() as Different>::Out>::Item;
    //~^ ERROR ambiguous associated type
}

// Repeated impl parameters must not produce an incompatible suggestion.
type RepeatedParam = <(u8, u16)>::Item;
//~^ ERROR ambiguous associated type

fn main() {}
