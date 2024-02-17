// Another regression test for #109815.

//@ check-pass

#![feature(min_specialization)]
#![feature(rustc_attrs)]

#[rustc_specialization_trait]
trait X {}
trait Z {
    type Assoc: X;
}
struct A<T>(T);

impl X for () {}

impl<T: X> Z for A<T> {
    type Assoc = ();
}

trait MyFrom<T> {
    fn from(other: T) -> Self;
}

impl<T> MyFrom<()> for T {
    default fn from(other: ()) -> T {
        panic!();
    }
}

impl<T: X> MyFrom<<A<T> as Z>::Assoc> for T {
    fn from(other: ()) -> T {
        panic!();
    }
}

fn main() {}
