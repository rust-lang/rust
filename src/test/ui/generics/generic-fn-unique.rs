// run-pass
#![feature(box_syntax)]

fn f<T>(x: Box<T>) -> Box<T> { return x; }

pub fn main() { let x = f(box 3); println!("{}", *x); }
