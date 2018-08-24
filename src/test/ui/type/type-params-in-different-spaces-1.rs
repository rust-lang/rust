use std::ops::Add;

trait BrokenAdd: Copy + Add<Output=Self> {
    fn broken_add<T>(&self, rhs: T) -> Self {
        *self + rhs //~  ERROR mismatched types
                    //~| expected type `Self`
                    //~| found type `T`
                    //~| expected Self, found type parameter
    }
}

impl<T: Copy + Add<Output=T>> BrokenAdd for T {}

pub fn main() {
    let foo: u8 = 0;
    let x: u8 = foo.broken_add("hello darkness my old friend".to_string());
    println!("{}", x);
}
