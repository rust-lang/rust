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

fn main() {
    ().hello();
    //~^ ERROR multiple applicable items in scope
}
