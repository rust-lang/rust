// Verify that the `where` clause suggestion is in the correct place
// Previously, the suggestion to add `where` clause was placed inside the derive
// like `#[derive(Clone where Inner<T>: Clone)]`
// instead of `struct Outer<T>(Inner<T>) where Inner<T>: Clone`

#![crate_type = "lib"]

struct Inner<T>(T);
//~^ HELP consider annotating `Inner<T>` with `#[derive(Clone)]`
impl Clone for Inner<()> {
    fn clone(&self) -> Self { todo!() }
}

#[derive(Clone)]
struct Outer<T>(Inner<T>);
//~^ ERROR the trait bound `Inner<T>: Clone` is not satisfied [E0277]
//~| HELP consider introducing a `where` clause
