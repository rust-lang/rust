//@ known-bug: #74299
#![feature(specialization)]

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

fn main() {
    ().f().g();
}
