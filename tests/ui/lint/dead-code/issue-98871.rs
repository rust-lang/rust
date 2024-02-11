#![deny(dead_code)]

#[derive(Default)]
struct T {} //~ ERROR struct `T` is never constructed

fn main() {}
