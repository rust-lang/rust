use std::fmt::Debug;

#[derive(Debug)] //~ ERROR expected struct, variant or union type, found type parameter `Irrelevant`
pub struct Irrelevant<Irrelevant> { //~ ERROR type arguments are not allowed for this type
    irrelevant: Irrelevant,
}

fn main() {}
