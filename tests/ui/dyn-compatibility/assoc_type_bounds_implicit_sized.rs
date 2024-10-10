//@ run-rustfix
#![allow(dead_code)]
trait TraitWithAType {
    type Item;
}
trait Trait {}
struct A {}
impl TraitWithAType for A {
    type Item = dyn Trait; //~ ERROR E0277
}
fn main() {}
