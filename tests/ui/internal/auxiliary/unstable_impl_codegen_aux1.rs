#![allow(internal_features)]
#![feature(staged_api)]
#![feature(impl_stability)]
#![stable(feature = "a", since = "1.1.1" )]

/// Aux crate for unstable impl codegen test.

#[stable(feature = "a", since = "1.1.1" )]
trait Trait {
    fn method(&self);
}

#[unstable_feature_bound(foo)]
impl Trait for T { 
// FIXME: this line above failed with cannot find type `T` in this scope
    fn method(&self) {
        println!("hi");
    }
}

fn main() {
}
