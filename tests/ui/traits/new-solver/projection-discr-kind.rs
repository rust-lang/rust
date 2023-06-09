// compile-flags: -Ztrait-solver=next

// Check that `<T::Assoc as DiscriminantKind>::Discriminant` doesn't normalize
// to itself and cause overflow/ambiguity.

trait Foo {
    type Assoc;
}

trait Bar {}
fn needs_bar(_: impl Bar) {}

fn foo<T: Foo>(x: T::Assoc) {
    needs_bar(std::mem::discriminant(&x));
    //~^ ERROR the trait bound `Discriminant<<T as Foo>::Assoc>: Bar` is not satisfied
}

fn main() {}
