struct Foo<T: Clone>(T);

use std::ops::Add;

impl<T: Clone, Add> Add for Foo<T> {
//~^ ERROR expected trait, found type parameter
    type Output = usize;

    fn add(self, rhs: Self) -> Self::Output {
        //~^ ERROR ambiguous associated type
        unimplemented!();
    }
}

fn main() {}
