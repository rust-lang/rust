// check-pass

// Test that we do not get a cycle due to
// resolving `Self::Bar` in the where clauses
// on a trait definition (in particular, in
// a where clause that is defining a superpredicate).

trait Foo {
    type Bar;
}
trait Qux
where
    Self: Foo,
    Self: AsRef<Self::Bar>,
{
}
trait Foo2 {}

trait Qux2
where
    Self: Foo2,
    Self: AsRef<Self::Bar>,
{
    type Bar;
}

fn main() {}
