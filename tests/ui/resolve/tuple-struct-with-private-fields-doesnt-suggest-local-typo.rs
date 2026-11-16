//@ run-rustfix
// Issue #66067.
#![allow(unused)]
mod outer {
    mod a {
        // This struct can't be constructed from `crate::` because of the private fields.
        pub struct Triplet(usize, usize, f64);
    }
    pub use self::a::Triplet;
}

use outer::Triplet;

fn main() {
    let mut triplets = Vec::new();
    // Ensure that we do not suggest changing `Triplet` to `triplets`:
    triplets.push(Triplet(0, 0, 1.0)); //~ ERROR: E0423
}
