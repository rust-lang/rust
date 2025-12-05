#![feature(box_patterns)]

struct A;

fn main() {
    let a: Box<[A]> = Box::new([A]);
    match a { //~ ERROR cannot move out of type `[A]`, a non-copy slice
        box [a] => {},
        _ => {}
    }
}
