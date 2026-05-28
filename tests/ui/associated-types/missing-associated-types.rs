use std::ops::{Add, Sub, Mul, Div};
trait X<Rhs>: Mul<Rhs> + Div<Rhs> {}
trait Y<Rhs>: Div<Rhs, Output = Rhs> {
    type A;
}
trait Z<Rhs>: Div<Rhs> {
    type A;
    type B;
}
trait Fine<Rhs>: Div<Rhs, Output = Rhs> {}

type Foo<Rhs> = dyn Add<Rhs> + Sub<Rhs> + X<Rhs> + Y<Rhs>;
//~^ ERROR only auto traits can be used as additional traits in a trait object
type Bar<Rhs> = dyn Add<Rhs> + Sub<Rhs> + X<Rhs> + Z<Rhs>;
//~^ ERROR only auto traits can be used as additional traits in a trait object
type Baz<Rhs> = dyn Add<Rhs> + Sub<Rhs> + Y<Rhs>;
//~^ ERROR only auto traits can be used as additional traits in a trait object
type Bat<Rhs> = dyn Add<Rhs> + Sub<Rhs> + Fine<Rhs>;
//~^ ERROR only auto traits can be used as additional traits in a trait object
type Bal<Rhs> = dyn X<Rhs>;
//~^ ERROR the value of the associated types

fn main() {}
