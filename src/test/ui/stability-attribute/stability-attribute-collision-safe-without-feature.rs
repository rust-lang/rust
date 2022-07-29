// aux-build:stability-attribute-collision-safe.rs
#![deny(unstable_name_collisions)]

extern crate stability_attribute_collision_safe;
use stability_attribute_collision_safe::Foo;

fn main() {
    let f = Foo;
    assert_eq!(f.example_safe(), 2); // okay! has `collision_safe` on defn

    assert_eq!(f.example(), 3);
    //~^ ERROR an associated function with this name may be added to the standard library in the future
    //~^^ WARN once this associated item is added to the standard library, the ambiguity may cause an error or change in behavior!
}
