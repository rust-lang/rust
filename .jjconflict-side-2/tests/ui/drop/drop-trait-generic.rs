//@ run-pass
#![allow(dead_code)]
struct S<T> {
    x: T
}

impl<T> ::std::ops::Drop for S<T> {
    fn drop(&mut self) {
        println!("bye");
    }
}

pub fn main() {
    let _x = S { x: 1 };
}
