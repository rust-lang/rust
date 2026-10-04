#![feature(pin_ergonomics)]
#![allow(incomplete_features)]

// Makes sure we don't accidentally accept `&pin Foo` without the `const` keyword.

fn ty() {
    let _x: &pin i32 = todo!(); //~ ERROR found `i32`
}

fn expr() {
    let x = 0_i32;
    let _x = &pin x; //~ ERROR found `x`
}

fn pat() {
    let &pin _x: &pin i32 = todo!(); //~ ERROR found `_x`
}

fn binding() {
    let ref pin _x: i32 = todo!(); //~ ERROR found `_x`
}

fn main() {
}
