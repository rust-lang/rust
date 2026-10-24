//@ check-pass

// rust-lang/rust#32382: Borrow checker used to complain about
// `foobar_3` in the `impl` below, presumably due to some interaction
// between the use of a lifetime in the associated type and the use of
// the overloaded operator[]. This regression test ensures that we do
// not resume complaining about it in the future.


use std::marker::PhantomData;
use std::ops::Index;

pub trait Context: Clone {
    type Container: ?Sized;
    fn foobar_1( container: &Self::Container ) -> &str;
    fn foobar_2( container: &Self::Container ) -> &str;
    fn foobar_3( container: &Self::Container ) -> &str;
}

#[derive(Clone)]
struct Foobar<'a> {
    phantom: PhantomData<&'a ()>
}

impl<'a> Context for Foobar<'a> {
    type Container = [&'a str];

    fn foobar_1<'r>( container: &'r [&'a str] ) -> &'r str {
        container[0]
    }

    fn foobar_2<'r>( container: &'r Self::Container ) -> &'r str {
        container.index( 0 )
    }

    fn foobar_3<'r>( container: &'r Self::Container ) -> &'r str {
        container[0]
    }
}

fn main() { }
