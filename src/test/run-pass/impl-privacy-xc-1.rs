// aux-build:impl_privacy_xc_1.rs

// pretty-expanded FIXME #23616

extern crate impl_privacy_xc_1;

pub fn main() {
    let fish = impl_privacy_xc_1::Fish { x: 1 };
    fish.swim();
}
