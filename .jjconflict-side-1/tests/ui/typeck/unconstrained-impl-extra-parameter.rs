//! Regression test for <https://github.com/rust-lang/rust/issues/16562>.

trait MatrixShape {}

struct Col<D, C> {
    data: D,
    col: C,
}

trait Collection { fn len(&self) -> usize; }

impl<T, M: MatrixShape> Collection for Col<M, usize> {
//~^ ERROR type parameter `T` is not constrained
    fn len(&self) -> usize {
        unimplemented!()
    }
}

fn main() {}
