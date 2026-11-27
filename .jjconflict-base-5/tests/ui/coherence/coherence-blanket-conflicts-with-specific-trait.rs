// Test that a blank impl for all T:PartialEq conflicts with an impl for some
// specific T when T:PartialEq.

trait OtherTrait {
    fn noop(&self);
}

trait MyTrait {
    fn get(&self) -> usize;
}

impl<T:OtherTrait> MyTrait for T {
    fn get(&self) -> usize { 0 }
}

struct MyType {
    dummy: usize
}

impl MyTrait for MyType {
//~^ ERROR E0119
    fn get(&self) -> usize { self.dummy }
}

impl OtherTrait for MyType {
    fn noop(&self) { }
}

fn main() { }
