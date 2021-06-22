// Test that we do NOT warn for inherent methods invoked via `T::` form.
//
// check-pass

#![deny(future_prelude_collision)]

pub struct MySeq {}

impl MySeq {
    pub fn from_iter(_: impl IntoIterator<Item = u32>) {}
}

fn main() {
    MySeq::from_iter(Some(22));
}
