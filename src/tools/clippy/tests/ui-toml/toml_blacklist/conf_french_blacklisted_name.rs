#![allow(dead_code)]
#![allow(clippy::single_match)]
#![allow(unused_variables)]
#![warn(clippy::blacklisted_name)]

fn test(toto: ()) {}

fn main() {
    let toto = 42;
    let tata = 42;
    let titi = 42;

    let tatab = 42;
    let tatatataic = 42;

    match (42, Some(1337), Some(0)) {
        (toto, Some(tata), titi @ Some(_)) => (),
        _ => (),
    }
}
