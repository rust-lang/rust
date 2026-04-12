// Regression test for #144888.
//
// `generics_of` was called with the crate root DefId during dyn-compatibility
// checking, which caused an ICE because Node::Crate was not handled in the
// match in `generics_of`.

trait Super {
    type Assoc;
}
impl dyn Foo<()> {}
trait Foo<T>: Super<Assoc = T>
//~^ ERROR type mismatch resolving `<Self as Super>::Assoc == ()`
//~| ERROR the size for values of type `Self` cannot be known
where
    <Self as Mirror>::Assoc: Clone,
    //~^ ERROR type mismatch resolving `<Self as Super>::Assoc == ()`
    //~| ERROR the size for values of type `Self` cannot be known
    //~| ERROR type mismatch resolving `<Self as Super>::Assoc == ()`
    //~| ERROR the size for values of type `Self` cannot be known
{
    fn transmute(&self) {}
    //~^ ERROR type mismatch resolving `<Self as Super>::Assoc == ()`
    //~| ERROR the size for values of type `Self` cannot be known
    //~| ERROR type mismatch resolving `<Self as Super>::Assoc == ()`
    //~| ERROR the size for values of type `Self` cannot be known
}

trait Mirror {
    type Assoc;
}
impl<T: Super<Assoc = ()>> Mirror for T {}
//~^ ERROR not all trait items implemented

fn main() {}
