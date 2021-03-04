// Regression test for #81712.

#![feature(generic_associated_types)]
#![allow(incomplete_features)]

trait A {
    type BType: B<AType = Self>;
}

trait B {
    type AType: A<BType = Self>;
}
trait C {
    type DType<T>: D<T, CType = Self>;
    //~^ ERROR: missing generics for associated type `C::DType` [E0107]
}
trait D<T> {
    type CType: C<DType = Self>;
}

fn main() {}
