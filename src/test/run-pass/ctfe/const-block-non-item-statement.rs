// run-pass
#![allow(dead_code)]

#[repr(u8)]
enum Foo {
    Bar = { let x = 1; 3 }
}

pub fn main() {
    assert_eq!(3, Foo::Bar as u8);
}
