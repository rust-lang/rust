//@ known-bug: #153947
#![expect(drop_bounds)]
pub struct Thing<T>(T) where [T]: Sized, Self: Drop;
impl<T> Drop for Thing<T> where [T]: Sized, Self: Drop {
    fn drop(&mut self) {}
}
impl<T> Drop for Thing<T> where [T]: Sized, Self: Drop {
    fn drop(&mut self) {}
}
fn main() {}
