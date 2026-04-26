//@ revisions: current next
//@[next] compile-flags: -Znext-solver
//@ ignore-compare-mode-next-solver (explicit revisions)

// A regression test for https://github.com/rust-lang/rust/issues/151329.
// Ensures we do not trigger an ICE when normalization fails for a
// projection on a trait object, even if the projection has the same
// trait id as the object's bound.

trait Foo {
    type V;
}

trait Callback<T: Foo>: Fn(&T, &T::V) {}

struct Bar<T: Foo + ?Sized> {
    callback: Box<dyn Callback<T>>,
}

impl<T: Foo> Bar<dyn Callback<T>> {
    //~^ ERROR: the trait bound `(dyn Callback<T> + 'static): Foo` is not satisfied
    fn event(&self) {
        //~^ ERROR: the trait bound `(dyn Callback<T> + 'static): Foo` is not satisfied
        (self.callback)(any(), any());
        //~^ ERROR: the trait bound `(dyn Callback<T> + 'static): Foo` is not satisfied
        //~| ERROR: expected function
    }
}

fn any() {}

fn main() {}
