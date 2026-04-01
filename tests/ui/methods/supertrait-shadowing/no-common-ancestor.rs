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

fn main() {
    ().hello();
    //~^ ERROR multiple applicable items in scope
}
