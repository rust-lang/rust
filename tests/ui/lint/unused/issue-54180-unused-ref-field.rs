//@ run-rustfix

#![deny(unused)]

pub struct S {
    pub f1: i32,
}

pub struct Point {
    pub x: i32,
    pub y: i32,
}

pub enum E {
    Variant { field: String }
}

pub fn foo(arg: &E) {
    match arg {
        E::Variant { ref field } => (), //~ ERROR unused variable
    }
}

fn main() {
    let s = S { f1: 123 };
    let S { ref f1 } = s; //~ ERROR unused variable

    let points = vec![Point { x: 1, y: 2 }];
    let _: i32 = points.iter().map(|Point { x, y }| y).sum(); //~ ERROR unused variable

    match (Point { x: 1, y: 2 }) {
        Point { y, ref mut x } => y, //~ ERROR unused variable
    };
}
