// Regression test for #82612.

use std::marker::PhantomData;

pub trait SparseSetIndex {
    fn sparse_set_index(&self) -> usize;
}
pub struct SparseArray<I, V = I> {
    values: Vec<Option<V>>,
    marker: PhantomData<I>,
}

impl<I: SparseSetIndex, V> SparseArray<I, V> {
    pub fn get_or_insert_with(&mut self, index: I, func: impl FnOnce() -> V) -> &mut V {
        let index = index.sparse_set_index();
        if index < self.values.len() {
            let value = unsafe { self.values.get_unchecked_mut(index) };
            value.get_or_insert_with(func) //~ ERROR mismatched types
        }
        unsafe { self.values.get_unchecked_mut(index).as_mut().unwrap() }
    }
}

fn main() {}
