// aux-build:impl_privacy_xc_1.rs
// xfail-fast

extern mod impl_privacy_xc_1;

pub fn main() {
    let fish = impl_privacy_xc_1::Fish { x: 1 };
    fish.swim();
}
