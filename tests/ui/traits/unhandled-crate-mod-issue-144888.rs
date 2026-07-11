// Regression test for <https://github.com/rust-lang/rust/issues/144888>.
// This used to ICE with `unhandled node Crate(Mod)`.

trait Super {
    type Assoc;
}

impl dyn Foo<()> {}

trait Foo<T>: Super<Assoc = T>
//~^ ERROR type mismatch resolving
//~| ERROR the size for values of type `Self` cannot be known at compilation time
//~| ERROR type mismatch resolving
//~| ERROR the size for values of type `Self` cannot be known at compilation time
where
    <Self as Mirror>::Assoc: Clone,
    //~^ ERROR type mismatch resolving
    //~| ERROR the size for values of type `Self` cannot be known at compilation time
{
    fn transmute(&self) {}
    //~^ ERROR type mismatch resolving
    //~| ERROR the size for values of type `Self` cannot be known at compilation time
    //~| ERROR type mismatch resolving
    //~| ERROR the size for values of type `Self` cannot be known at compilation time
}

trait Mirror {
    type Assoc;
}

impl<T: Super<Assoc = ()>> Mirror for T {}
//~^ ERROR not all trait items implemented

fn main() {}
