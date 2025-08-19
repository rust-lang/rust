#![feature(pin_ergonomics)]
#![allow(incomplete_features)]

// Makes sure we don't accidentally accept `&pin Foo` without the `const` keyword.

fn main() {
    let _x: &pin i32 = todo!(); //~ ERROR found `i32`
}
