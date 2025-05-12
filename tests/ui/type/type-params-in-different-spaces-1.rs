//@ dont-require-annotations: NOTE

use std::ops::Add;

trait BrokenAdd: Copy + Add<Output=Self> {
    fn broken_add<T>(&self, rhs: T) -> Self {
        *self + rhs //~  ERROR mismatched types
                    //~| NOTE expected type parameter `Self`, found type parameter `T`
                    //~| NOTE expected type parameter `Self`
                    //~| NOTE found type parameter `T`
    }
}

impl<T: Copy + Add<Output=T>> BrokenAdd for T {}

pub fn main() {
    let foo: u8 = 0;
    let x: u8 = foo.broken_add("hello darkness my old friend".to_string());
    println!("{}", x);
}
