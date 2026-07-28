// The associated-function suggestion must use turbofish syntax for a generic ADT.

trait Kind {
    fn kind();
}

impl<T> Kind for Option<T> {
    fn kind() {}
}

fn main() {
    let value = Some(2_i32);
    value.kind();
    //~^ ERROR no method named `kind` found for enum `Option<T>`
}
