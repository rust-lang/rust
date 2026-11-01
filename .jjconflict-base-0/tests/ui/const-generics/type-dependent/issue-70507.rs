//@ run-pass

trait ConstChunksExactTrait<T> {
    fn const_chunks_exact<const N: usize>(&self) -> ConstChunksExact<'_, T, {N}>;
}

impl <T> ConstChunksExactTrait<T> for [T] {
    fn const_chunks_exact<const N: usize>(&self) -> ConstChunksExact<'_, T, {N}> {
        assert!(N != 0);
        let rem = self.len() % N;
        let len = self.len() - rem;
        let (fst, _) = self.split_at(len);
        ConstChunksExact { v: fst, }
    }
}

struct ConstChunksExact<'a, T: 'a, const N: usize> {
    v: &'a [T],
}

impl <'a, T: std::fmt::Debug, const N: usize> Iterator for ConstChunksExact<'a, T, {N}> {
    type Item = &'a [T; N];

    fn next(&mut self) -> Option<Self::Item> {
        if self.v.len() < N {
            None
        } else {
            let (fst, snd) = self.v.split_at(N);

            self.v = snd;
            let ptr = fst.as_ptr() as *const _;
            Some(unsafe { &*ptr})
        }
    }
}

fn main() {
    let slice = &[1i32, 2, 3, 4, 5, 6, 7, 8, 9, 10];

    let mut iter = [[1, 2, 3], [4, 5, 6], [7, 8, 9]].iter();

    for a in slice.const_chunks_exact::<3>() {
        assert_eq!(a, iter.next().unwrap());
    }
}
