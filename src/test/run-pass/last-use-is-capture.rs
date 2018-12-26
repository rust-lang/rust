#![allow(dead_code)]
// Make sure #1399 stays fixed

#![feature(box_syntax)]

struct A { a: Box<isize> }

pub fn main() {
    fn invoke<F>(f: F) where F: FnOnce() { f(); }
    let k: Box<_> = box 22;
    let _u = A {a: k.clone()};
    invoke(|| println!("{}", k.clone()) )
}
