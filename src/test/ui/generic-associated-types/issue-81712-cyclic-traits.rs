// Regression test for #81712.

trait A {
    type BType: B<AType = Self>;
}

trait B {
    type AType: A<BType = Self>;
}
trait C {
    type DType<T>: D<T, CType = Self>;
}
trait D<T> {
    type CType: C<DType = Self>;
    //~^ ERROR missing generics for associated type
}

fn main() {}
