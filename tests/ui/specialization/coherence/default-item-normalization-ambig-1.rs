// regression test for #73299.
#![feature(specialization)] //~ WARN the feature `specialization` is incomplete

trait X {
    type U;
    fn f(&self) -> Self::U {
        loop {}
    }
}

impl<T> X for T {
    default type U = ();
}

trait Y {
    fn g(&self) {}
}

impl Y for <() as X>::U {}
impl Y for <i32 as X>::U {}
//~^ ERROR conflicting implementations of trait `Y` for type `<() as X>::U`

fn main() {
    ().f().g();
}
