//@ compile-flags: -Znext-solver

// Check that `<T::Assoc as DiscriminantKind>::Discriminant` doesn't normalize
// to itself and cause overflow/ambiguity.

trait Foo {
    type Assoc;
}

trait Bar {}
fn needs_bar(_: impl Bar) {}

fn foo<T: Foo>(x: T::Assoc) {
    needs_bar(std::mem::discriminant(&x));
    //~^ ERROR trait `Bar` is not implemented for `Discriminant<<T as Foo>::Assoc>`
}

fn main() {}
