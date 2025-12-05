//@ check-pass

struct Test<T, const N: usize>([T; N]);

impl<T: Copy + Default, const N: usize> Default for Test<T, N> {
    fn default() -> Self {
        Self([T::default(); N])
    }
}

impl<T, const N: usize> std::ops::Deref for Test<T, N> {
    type Target = [T; N];

    fn deref(&self) -> &[T; N] {
        &self.0
    }
}

fn test() -> Test<u64, 16> {
    let test = Test::default();
    println!("{}", test.len());
    test
}

fn main() {
    test();
}
