//@ revisions:rpass1

struct FakeArray<T, const N: usize>(T);

impl<T, const N: usize> FakeArray<T, N> {
    fn len(&self) -> usize {
        N
    }
}

fn main() {
    let fa = FakeArray::<u32, { 32 }>(1);
    assert_eq!(fa.len(), 32);
}
