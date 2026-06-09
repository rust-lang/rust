// Check that types with different const arguments are different.
//@ revisions: full min

#![cfg_attr(full, feature(generic_const_exprs))]
#![cfg_attr(full, allow(incomplete_features))]

struct ConstUsize<const V: usize> {}

fn main() {
    let mut u = ConstUsize::<3> {};
    u = ConstUsize::<4> {};
    //~^ ERROR mismatched types
}
