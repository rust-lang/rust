#![deny(dead_code)]

#[derive(PartialEq)]
struct MyStruct {
    x: i32,
    y: i32, //~ ERROR field `y` is never read
}

struct MyStruct2 {
    x: i32,
    y: i32, //~ ERROR field `y` is never read
}

pub fn main() {
    let ms = MyStruct { x: 1, y: 2 };
    let _ = ms.x;

    let ms = MyStruct2 { x: 1, y: 2 };
    let _ = ms.x;
}
