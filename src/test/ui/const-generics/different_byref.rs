// Check that different const types are different.
// revisions: full min

#![cfg_attr(full, feature(const_generics))]
#![cfg_attr(full, allow(incomplete_features))]

struct Const<const V: [usize; 1]> {}
//[min]~^ ERROR `[usize; 1]` is forbidden

fn main() {
    let mut x = Const::<{ [3] }> {};
    x = Const::<{ [4] }> {};
    //[full]~^ ERROR mismatched types
}
