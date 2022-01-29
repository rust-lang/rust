// check-pass

pub trait Indexable<T>: std::ops::Index<usize, Output = T> {
    fn index2(&self, i: usize) -> &T {
        &self[i]
    }
}
fn main() {}
