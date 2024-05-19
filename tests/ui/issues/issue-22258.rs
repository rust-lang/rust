//@ run-pass
use std::ops::Add;

fn f<T: Add>(a: T, b: T) -> <T as Add>::Output {
    a + b
}

fn main() {
    println!("a + b is {}", f::<f32>(100f32, 200f32));
}
