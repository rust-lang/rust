//@ build-pass
#![allow(unused)]

fn main() {
    let ref x: str = *"";
}

fn foo(r: &(usize, str)) -> usize {
    let (x, _): (usize, str) = *r;
    x
}
