//@ run-pass
//@ aux-build:impl_privacy_xc_1.rs


extern crate impl_privacy_xc_1;

pub fn main() {
    let fish = impl_privacy_xc_1::Fish { x: 1 };
    fish.swim();
}
