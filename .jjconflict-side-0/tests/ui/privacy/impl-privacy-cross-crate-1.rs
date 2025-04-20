//@ run-pass
//@ aux-build:impl-privacy-cross-crate-1.rs


extern crate impl_privacy_cross_crate_1;

pub fn main() {
    let fish = impl_privacy_cross_crate_1::Fish { x: 1 };
    fish.swim();
}
