// aux-build:stability-attribute-collision-safe.rs
// check-pass
#![feature(new_feature)]

extern crate stability_attribute_collision_safe;
use stability_attribute_collision_safe::Foo;

fn main() {
    let f = Foo;
    assert_eq!(f.example_safe(), 2); // okay! has `collision_safe` on defn

    assert_eq!(f.example(), 4); // okay! have feature!
}
