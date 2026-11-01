//@ check-pass

#![allow(non_camel_case_types)] // genus is always capitalized

pub(crate) struct Snail;

mod sea {
    pub(super) struct Turtle;
}

struct Tortoise;

pub struct Shell<T> {
    pub(crate) creature: T,
}

pub type Helix_pomatia = Shell<Snail>;
//~^ WARNING type `Snail` is more private than the item `Helix_pomatia`
pub type Dermochelys_coriacea = Shell<sea::Turtle>;
//~^ WARNING type `Turtle` is more private than the item `Dermochelys_coriacea`
pub type Testudo_graeca = Shell<Tortoise>;
//~^ WARNING type `Tortoise` is more private than the item `Testudo_graeca`

fn main() {}
