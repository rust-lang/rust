// aux-build:stability-attribute-collision-safe.rs
#![deny(unstable_name_collisions)]

extern crate stability_attribute_collision_safe;
use stability_attribute_collision_safe::{Trait, OtherTrait};

pub trait LocalTrait {
    fn not_safe(&self) -> u32 {
        1
    }

    fn safe(&self) -> u32 {
        1
    }
}

impl LocalTrait for char {}


fn main() {
    // Despite having `collision_safe` on defn, the fn chosen doesn't have a stability attribute,
    // thus could be user code (and is in this test), so the lint is still appropriate..
    assert_eq!('x'.safe(), 1);
    //~^ ERROR an associated function with this name may be added to the standard library in the future
    //~^^ WARN once this associated item is added to the standard library, the ambiguity may cause an error or change in behavior!

    // ..but with `collision_safe` on defn, if the chosen item has a stability attribute, then
    // assumed to be from std or somewhere that's been checked to be okay, so no lint..
    assert_eq!('x'.safe_and_shadowing_a_stable_item(), 4); // okay!

    // ..and not safe functions should, of course, still lint..
    assert_eq!('x'.not_safe(), 1);
    //~^ ERROR an associated function with this name may be added to the standard library in the future
    //~^^ WARN once this associated item is added to the standard library, the ambiguity may cause an error or change in behavior!
}
