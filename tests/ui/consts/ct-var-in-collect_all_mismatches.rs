struct Foo<T, const N: usize> {
    array: [T; N],
}

trait Bar<const N: usize> {}

impl<T, const N: usize> Foo<T, N> {
    fn trigger(self) {
        self.unsatisfied()
        //~^ ERROR the trait bound `T: Bar<N>` is not satisfied
    }

    fn unsatisfied(self)
    where
        T: Bar<N>,
    {
    }
}

fn main() {}
