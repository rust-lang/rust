#![feature(specialization)] //~ WARN the feature `specialization` is incomplete

// Regression test for a specialization-related ICE (#39448).

trait A: Sized {
    fn foo(self, _: Self) -> Self {
        self
    }
}

impl A for u8 {}
impl A for u16 {}

impl FromA<u8> for u16 {
    fn from(x: u8) -> u16 {
        x as u16
    }
}

trait FromA<T> {
    fn from(t: T) -> Self;
}

impl<T: A, U: A + FromA<T>> FromA<T> for U {
    //~^ ERROR cycle detected when computing whether impls specialize one another
    default fn from(x: T) -> Self {
        ToA::to(x)
    }
}

trait ToA<T> {
    fn to(self) -> T;
}

impl<T, U> ToA<U> for T
where
    U: FromA<T>,
{
    fn to(self) -> U {
        U::from(self)
    }
}

#[allow(dead_code)]
fn foo<T: A, U: A>(x: T, y: U) -> U {
    x.foo(y.to()).to()
}

fn main() {
    let z = foo(8u8, 1u16);
}
