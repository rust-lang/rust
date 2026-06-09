//@ run-rustfix
#![allow(unused, dead_code)]

#[derive(Clone, Copy)]
struct Stuff {
    count: i32,
}
struct Error;

fn demo() -> Result<Stuff, Error> {
    let count = Ok(1);
    Ok(Stuff { count }) //~ ERROR mismatched types
}

fn demo_unwrap() -> Stuff {
    let count = Some(1);
    Stuff { count } //~ ERROR mismatched types
}

fn main() {}
