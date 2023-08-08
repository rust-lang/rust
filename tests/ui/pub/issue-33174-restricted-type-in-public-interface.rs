#![feature(type_privacy_lints)]
#![allow(non_camel_case_types)] // genus is always capitalized
#![warn(private_interfaces)]
//~^ NOTE the lint level is defined here

// In this test both old and new private-in-public diagnostic were emitted.
// Old diagnostic will be deleted soon.
// See https://rust-lang.github.io/rfcs/2145-type-privacy.html.

pub(crate) struct Snail;
//~^ NOTE `Snail` declared as private
//~| NOTE but type `Snail` is only usable at visibility `pub(crate)`

mod sea {
    pub(super) struct Turtle;
    //~^ NOTE `Turtle` declared as crate-private
    //~| NOTE but type `Turtle` is only usable at visibility `pub(crate)`
}

struct Tortoise;
//~^ NOTE `Tortoise` declared as private
//~| NOTE but type `Tortoise` is only usable at visibility `pub(crate)`

pub struct Shell<T> {
    pub(crate) creature: T,
}

pub type Helix_pomatia = Shell<Snail>;
//~^ ERROR private type `Snail` in public interface
//~| WARNING type `Snail` is more private than the item `Helix_pomatia`
//~| NOTE can't leak private type
//~| NOTE type alias `Helix_pomatia` is reachable at visibility `pub`
pub type Dermochelys_coriacea = Shell<sea::Turtle>;
//~^ ERROR crate-private type `Turtle` in public interface
//~| WARNING type `Turtle` is more private than the item `Dermochelys_coriacea`
//~| NOTE can't leak crate-private type
//~| NOTE type alias `Dermochelys_coriacea` is reachable at visibility `pub`
pub type Testudo_graeca = Shell<Tortoise>;
//~^ ERROR private type `Tortoise` in public interface
//~| WARNING type `Tortoise` is more private than the item `Testudo_graeca`
//~| NOTE can't leak private type
//~| NOTE type alias `Testudo_graeca` is reachable at visibility `pub`

fn main() {}
