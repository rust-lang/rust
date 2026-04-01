//@ run-pass
#![allow(dead_code)]
// Issue #976



fn f<T>(x: Box<T>) {
    let _x2 = x;
}
pub fn main() { }
