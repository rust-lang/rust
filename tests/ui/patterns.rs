#![allow(unused)]
#![warn(clippy::all)]

fn main() {
    let v = Some(true);
    match v {
        Some(x) => (),
        y @ _ => (),
    }
    match v {
        Some(x) => (),
        y @ None => (), // no error
    }
}
