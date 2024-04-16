//@ known-bug: #123141
trait ConstChunksExactTrait<T> {
    fn const_chunks_exact<const N: usize>(&self) -> ConstChunksExact<'_, T, { N }>;
}

impl<T> ConstChunksExactTrait<T> for [T] {}

struct ConstChunksExact<'a, T: 'a, const N: usize> {}

impl<'a, T, const N: usize> Iterator for ConstChunksExact<'a, T, { rem }> {
    type Item = &'a [T; N];
}

fn main() {
    let slice = &[1i32, 2, 3, 4, 5, 6, 7, 7, 9, 1i32];

    let mut iter = [[1, 2, 3], [4, 5, 6], [7, 8, 9]].iter();

    for a in slice.const_chunks_exact::<3>() {
        assert_eq!(a, iter.next().unwrap());
    }
}
