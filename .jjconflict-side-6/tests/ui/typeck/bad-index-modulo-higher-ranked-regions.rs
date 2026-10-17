// Test against ICE in #118111

use std::ops::Index;

struct Map<T, F> {
    f: F,
    inner: T,
}

impl<T, F, Idx> Index<Idx> for Map<T, F>
where
    T: Index<Idx>,
    F: FnOnce(&T, Idx) -> Idx,
{
    type Output = T::Output;

    fn index(&self, index: Idx) -> &Self::Output {
        todo!()
    }
}

fn main() {
    Map { inner: [0_usize], f: |_, i: usize| 1_usize }[0];
    //~^ ERROR cannot index into a value of type
    // Problem here is that
    //   `f: |_, i: usize| ...`
    // should be
    //   `f: |_: &_, i: usize| ...`
}
