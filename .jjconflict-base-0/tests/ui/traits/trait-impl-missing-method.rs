//! Trait impls must define all required methods.

trait MyTrait {
    fn trait_method(&self);
}

struct ImplType;

impl MyTrait for ImplType {} //~ ERROR not all trait items implemented, missing: `trait_method`

fn main() {
    let _ = ImplType;
}
