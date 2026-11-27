use std::fmt::Debug;

#[derive(Debug)]
pub struct Irrelevant<Irrelevant> { //~ ERROR type arguments are not allowed on type parameter
    irrelevant: Irrelevant,
}

fn main() {}
