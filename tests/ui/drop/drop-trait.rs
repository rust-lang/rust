//@ run-pass
#![allow(dead_code)]
struct Foo {
    x: isize
}

impl Drop for Foo {
    fn drop(&mut self) {
        println!("bye");
    }
}

pub fn main() {
    let _x: Foo = Foo { x: 3 };
}
