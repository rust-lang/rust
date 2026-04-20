#![feature(supertrait_item_shadowing)]

trait A {
    fn hello(&self) -> &'static str {
        "A"
    }
}
impl<T> A for T {}

trait B {
    fn hello(&self) -> &'static str {
        "B"
    }
}
impl<T> B for T {}

trait C: A + B {
    fn hello(&self) -> &'static str {
        "C"
    }
}
impl<T> C for T {}

// Since `D` is not a subtrait of `C`,
// we have no obvious lower bound.

trait D: B {
    fn hello(&self) -> &'static str {
        "D"
    }
}
impl<T> D for T {}

fn main() {
    ().hello();
    //~^ ERROR multiple applicable items in scope
}
