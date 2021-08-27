// Check that different const types are different.
#![feature(const_param_types)]
#![allow(incomplete_features)]

struct Const<const V: [usize; 1]> {}

fn main() {
    let mut x = Const::<{ [3] }> {};
    x = Const::<{ [4] }> {};
    //~^ ERROR mismatched types
}
