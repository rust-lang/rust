use std as derive;

#[derive(Default)] //~ ERROR cannot determine resolution for the attribute macro `derive`
struct S;

fn main() {}
