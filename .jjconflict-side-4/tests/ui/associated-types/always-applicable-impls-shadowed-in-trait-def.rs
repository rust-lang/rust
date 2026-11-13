//@ revisions: old next
//@[next] compile-flags: -Znext-solver

// Testing that even if there's an always applicable blanket impl, the trait
// definition cannot use that impl to normalize its own associated types.

trait Eq<T> {}
impl<T> Eq<T> for T {}
struct IsEqual<T: Eq<U>, U>(T, U);

trait Trait: Sized {
    type Assoc;
    fn foo() -> IsEqual<Self, Self::Assoc> {
        //~^ ERROR the trait bound `Self: Eq<<Self as Trait>::Assoc>` is not satisfied
        todo!()
    }
}

impl<T> Trait for T {
    type Assoc = T;
}

fn main() {}
