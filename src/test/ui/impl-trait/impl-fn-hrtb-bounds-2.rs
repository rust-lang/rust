use std::fmt::Debug;

fn a() -> impl Fn(&u8) -> impl Debug {
    |x| x //~ ERROR lifetime may not live long enough
}

fn main() {}
