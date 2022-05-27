use std::fmt::Debug;

#[derive(Debug)]
pub struct Irrelevant<Irrelevant> { //~ ERROR type arguments are not allowed for this type
    irrelevant: Irrelevant,
}

fn main() {}
