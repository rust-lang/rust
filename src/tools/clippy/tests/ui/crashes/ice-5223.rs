//@ check-pass
// Regression test for #5233
#![warn(clippy::indexing_slicing, clippy::iter_cloned_collect)]

pub struct KotomineArray<T, const N: usize> {
    arr: [T; N],
}

impl<T: std::clone::Clone, const N: usize> KotomineArray<T, N> {
    pub fn ice(self) {
        let _ = self.arr[..];
        let _ = self.arr.iter().cloned().collect::<Vec<_>>();
    }
}

fn main() {}
