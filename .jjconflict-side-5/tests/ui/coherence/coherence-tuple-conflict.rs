use std::fmt::Debug;
use std::default::Default;

// Test that a blank impl for all T conflicts with an impl for some
// specific T.

trait MyTrait {
    fn get(&self) -> usize;
}

impl<T> MyTrait for (T,T) {
    fn get(&self) -> usize { 0 }
}

impl<A,B> MyTrait for (A,B) {
//~^ ERROR E0119
    fn get(&self) -> usize { self.dummy }
    //~^ ERROR: no field `dummy`
}

fn main() { }
