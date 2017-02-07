#![feature(plugin)]
#![plugin(clippy)]
#![allow(unused)]
#![deny(clippy)]

fn main() {
    let v = Some(true);
    match v {
        Some(x) => (),
        y @ _   => (),  //~ERROR the `y @ _` pattern can be written as just `y`
    }
    match v {
        Some(x)  => (),
        y @ None => (),  // no error
    }
}
