//! Checks that associated type defaults are properly validated.
//!
//! This means:
//! * Default types are wfchecked
//! * Default types are checked against where clauses on the assoc. type
//!   (eg. `type Assoc: Clone = NotClone`), and also against where clauses on
//!   the trait itself when possible

// compile-fail

#![feature(associated_type_defaults)]

struct NotClone;

// Assoc. type bounds must hold for the default type
trait Tr {
    type Ty: Clone = NotClone;
    //~^ ERROR the trait bound `NotClone: std::clone::Clone` is not satisfied
}

// Where-clauses defined on the trait must also be considered
trait Tr2 where Self::Ty: Clone {
    //~^ ERROR the trait bound `NotClone: std::clone::Clone` is not satisfied
    type Ty = NotClone;
}

// Independent of where-clauses (there are none here), default types must always be wf
trait Tr3 {
    type Ty = Vec<[u8]>;
    //~^ ERROR the size for values of type `[u8]` cannot be known at compilation time
}

// Involved type parameters must fulfill all bounds required by defaults that mention them
trait Foo<T> {
    type Bar: Clone = Vec<T>;
    //~^ ERROR the trait bound `T: std::clone::Clone` is not satisfied
}

trait Bar: Sized {
    // `(): Foo<Self>` might hold for some possible impls but not all.
    type Assoc: Foo<Self> = ();
    //~^ ERROR the trait bound `(): Foo<Self>` is not satisfied
}

trait IsU8<T> {}
impl<T> IsU8<u8> for T {}

// Test that mentioning the assoc. type inside where clauses works
trait C where
    Vec<Self::Assoc>: Clone,
    Self::Assoc: IsU8<Self::Assoc>,
    bool: IsU8<Self::Assoc>,
{
    type Assoc = u8;
}

// Test that we get all expected errors if that default is unsuitable
trait D where
    Vec<Self::Assoc>: Clone,
    //~^ ERROR the trait bound `NotClone: std::clone::Clone` is not satisfied
    Self::Assoc: IsU8<Self::Assoc>,
    //~^ ERROR the trait bound `NotClone: IsU8<NotClone>` is not satisfied
    bool: IsU8<Self::Assoc>,
    //~^ ERROR the trait bound `bool: IsU8<NotClone>` is not satisfied
{
    type Assoc = NotClone;
}

trait Foo2<T> where
    <Self as Foo2<T>>::Bar: Clone,
    //~^ ERROR the trait bound `<Self as Foo2<T>>::Baz: std::clone::Clone` is not satisfied
{
    type Bar = Vec<Self::Baz>;
    type Baz = T;
}

trait Foo3<T: Clone> where
    <Self as Foo3<T>>::Bar: Clone,
    //~^ ERROR the trait bound `<Self as Foo3<T>>::Baz: std::clone::Clone` is not satisfied
{
    type Bar = Vec<Self::Baz>;
    type Baz = T;
}

trait Foo4<T> where
    <Self as Foo4<T>>::Bar: Clone,
{
    type Bar = Vec<Self::Baz>;
    type Baz: Clone = T;
    //~^ ERROR the trait bound `T: std::clone::Clone` is not satisfied
}

fn main() {}
