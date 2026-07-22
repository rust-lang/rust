//! Regression test for <https://github.com/rust-lang/rust/issues/41742>.
//! Test indexing with the wrong type doesn't cause ICE.

use std::ops::{Index, IndexMut};

struct S;
struct H;

impl S {
    fn f(&mut self) {}
}

impl Index<u32> for H {
    type Output = S;
    fn index(&self, index: u32) -> &S {
        unimplemented!()
    }
}

impl IndexMut<u32> for H {
    fn index_mut(&mut self, index: u32) -> &mut S {
        unimplemented!()
    }
}

fn main() {
    H["?"].f(); //~ ERROR mismatched types
}
