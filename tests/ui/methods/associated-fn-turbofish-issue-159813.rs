//@ run-rustfix

// The associated-function suggestion must use turbofish syntax for a generic ADT.

trait Kind {
    fn kind();
}

impl<T> Kind for Option<T> {
    fn kind() {}
}

fn main() {
    let _value = Some(2_i32);
    _value.kind();
    //~^ ERROR no method named `kind` found for enum `Option<T>`
}
