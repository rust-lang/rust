//@ run-pass
#![allow(dead_code)]

pub fn main() {
    let _: Box<_> = Box::new(100);
}

fn vec() {
    vec![0];
}
