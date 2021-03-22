// check-pass

#![deny(unused_variables)]
#![feature(never_type)]

fn never() -> core::convert::Infallible {
    panic!()
}

fn main() {
    let n = never();
    match n {}
}
