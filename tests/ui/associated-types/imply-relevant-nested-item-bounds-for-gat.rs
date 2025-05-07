//@ check-pass

// Test that `for<'a> Self::Gat<'a>: Debug` is implied in the definition of `Foo`,
// just as it would be if it weren't a GAT but just a regular associated type.

use std::fmt::Debug;

trait Foo
where
    for<'a> Self::Gat<'a>: Debug,
{
    type Gat<'a>;
}

fn test<T: Foo>(x: T::Gat<'static>) {
    println!("{:?}", x);
}

fn main() {}
