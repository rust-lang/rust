//@ run-pass
//@ aux-build:impl_privacy_xc_2.rs

extern crate impl_privacy_xc_2;

pub fn main() {
    let fish1 = impl_privacy_xc_2::Fish { x: 1 };
    let fish2 = impl_privacy_xc_2::Fish { x: 2 };
    if fish1.eq(&fish2) { println!("yes") } else { println!("no") };
}
