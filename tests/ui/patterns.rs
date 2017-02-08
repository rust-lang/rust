#![feature(plugin)]
#![plugin(clippy)]
#![allow(unused)]
#![deny(clippy)]

fn main() {
    let v = Some(true);
    match v {
        Some(x) => (),
        y @ _   => (),
    }
    match v {
        Some(x)  => (),
        y @ None => (),  // no error
    }
}
