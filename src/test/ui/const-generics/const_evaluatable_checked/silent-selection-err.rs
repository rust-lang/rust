// run-pass
#![feature(const_generics, const_evaluatable_checked)]
#![allow(incomplete_features)]

struct Array<const N: usize>;

trait Matrix {
    fn do_it(&self) -> usize;
}

impl Matrix for Array<0> {
    fn do_it(&self) -> usize {
        0
    }
}

impl Matrix for Array<1> {
    fn do_it(&self) -> usize {
        1
    }
}

impl Matrix for Array<2> {
    fn do_it(&self) -> usize {
        2
    }
}

impl<const N: usize> Matrix for Array<N>
where
    [u8; N - 3]: Sized,
{
    fn do_it(&self) -> usize {
        N + 1
    }
}

fn main() {
    assert_eq!(Array::<0>.do_it(), 0);
    assert_eq!(Array::<1>.do_it(), 1);
    assert_eq!(Array::<2>.do_it(), 2);
    assert_eq!(Array::<3>.do_it(), 4);
}
