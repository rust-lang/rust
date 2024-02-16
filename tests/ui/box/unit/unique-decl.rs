//@ run-pass
#![allow(dead_code)]


pub fn main() {
    let _: Box<isize>;
}

fn f(_i: Box<isize>) -> Box<isize> {
    panic!();
}
