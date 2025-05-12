// A regression test for #109815.

#![feature(min_specialization)]
#![feature(rustc_attrs)]

#[rustc_specialization_trait]
trait X {}
trait Y: X {}
trait Z {
    type Assoc: Y;
}
struct A<T>(T);

impl<T: X> Z for A<T> {}
//~^ ERROR not all trait items implemented

trait MyFrom<T> {
    fn from(other: T) -> Self;
}

impl<T> MyFrom<T> for T {
    default fn from(other: T) -> T {
        other
    }
}

impl<T: X> MyFrom<<A<T> as Z>::Assoc> for T {
    fn from(other: <A<T> as Z>::Assoc) -> T {
        other
    }
}

fn main() {}
