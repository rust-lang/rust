struct Foo<T, const N: usize> {
    array: [T; N],
}

trait Bar<const N: usize> {}

impl<T, const N: usize> Foo<T, N> {
    fn trigger(self) {
        self.unsatisfied()
        //~^ ERROR trait `Bar<N>` is not implemented for `T`
    }

    fn unsatisfied(self)
    where
        T: Bar<N>,
    {
    }
}

fn main() {}
