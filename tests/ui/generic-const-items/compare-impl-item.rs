#![feature(generic_const_items)]
#![allow(incomplete_features)]

trait Trait<P> {
    const A: ();
    const B<const K: u64, const Q: u64>: u64;
    const C<T>: T;
    const D<const N: usize>: usize;

    const E: usize;
    const F<T: PartialEq>: ();
}

impl<P> Trait<P> for () {
    const A<T>: () = ();
    //~^ ERROR const `A` has 1 type parameter but its trait declaration has 0 type parameters
    const B<const K: u64>: u64 = 0;
    //~^ ERROR const `B` has 1 const parameter but its trait declaration has 2 const parameters
    const C<'a>: &'a str = "";
    //~^ ERROR const `C` has 0 type parameters but its trait declaration has 1 type parameter
    const D<const N: u16>: u16 = N;
    //~^ ERROR const `D` has an incompatible generic parameter for trait `Trait`

    const E: usize = 1024
    where
        P: Copy; //~ ERROR impl has stricter requirements than trait
    const F<T: Eq>: () = (); //~ ERROR impl has stricter requirements than trait
}

fn main() {}
