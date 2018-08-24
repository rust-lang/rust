#![allow(non_camel_case_types)]  // genus is always capitalized

pub(crate) struct Snail;
//~^ NOTE `Snail` declared as crate-visible

mod sea {
    pub(super) struct Turtle;
    //~^ NOTE `sea::Turtle` declared as restricted
}

struct Tortoise;
//~^ NOTE `Tortoise` declared as private

pub struct Shell<T> {
    pub(crate) creature: T,
}

pub type Helix_pomatia = Shell<Snail>;
//~^ ERROR crate-visible type `Snail` in public interface
//~| NOTE can't leak crate-visible type
pub type Dermochelys_coriacea = Shell<sea::Turtle>;
//~^ ERROR restricted type `sea::Turtle` in public interface
//~| NOTE can't leak restricted type
pub type Testudo_graeca = Shell<Tortoise>;
//~^ ERROR private type `Tortoise` in public interface
//~| NOTE can't leak private type

fn main() {}
