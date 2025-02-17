#![feature(supertrait_item_shadowing)]

trait A {
    fn hello(&self) {
        println!("A");
    }
}
impl<T> A for T {}

trait B {
    fn hello(&self) {
        println!("B");
    }
}
impl<T> B for T {}

trait C: A + B {
    fn hello(&self) {
        println!("C");
    }
}
impl<T> C for T {}

// Since `D` is not a subtrait of `C`,
// we have no obvious lower bound.

trait D: B {
    fn hello(&self) {
        println!("D");
    }
}
impl<T> D for T {}

fn main() {
    ().hello();
    //~^ ERROR multiple applicable items in scope
}
