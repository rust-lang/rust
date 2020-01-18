// run-rustfix
#![allow(unused)]
#![warn(clippy::all)]

fn main() {
    let v = Some(true);
    let s = [0, 1, 2, 3, 4];
    match v {
        Some(x) => (),
        y @ _ => (),
    }
    match v {
        Some(x) => (),
        y @ None => (), // no error
    }
    match s {
        [x, inside @ .., y] => (), // no error
        [..] => (),
    }
}
