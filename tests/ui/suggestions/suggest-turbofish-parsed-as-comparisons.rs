struct Many<A, B, C, D> {
    a: A,
    b: B,
    c: C,
    d: D,
}

impl<A, B, C, D> Many<A, B, C, D> {
    fn new() -> Self {
        todo!()
    }
}

fn bar<T>(_: Many<T, T, T, T>) {}

fn main() {
    let _ = bar(Many<i32, Many<(), i32, (), ()>, i32, i32>::new());
    //~^ ERROR expected expression, found `,`
    //~| HELP use `::<...>` instead of `<...>`
}
