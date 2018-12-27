#![allow(dead_code)]
#[derive(Debug)]
struct Pair<T, U> { a: T, b: U }
struct Triple { x: isize, y: isize, z: isize }

fn f<T,U>(x: T, y: U) -> Pair<T, U> { return Pair {a: x, b: y}; }

pub fn main() {
    println!("{}", f(Triple {x: 3, y: 4, z: 5}, 4).a.x);
    println!("{}", f(5, 6).a);
}
