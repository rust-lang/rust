//@check-pass

use std::marker::PhantomData;

trait Trait {
    type Assoc;
}

struct A<T: Trait>(<T as Trait>::Assoc);

struct X;

impl Trait for X {
    type Assoc = A<X>;
}

struct Iter<T>(PhantomData<T>);

impl<T> Iterator for Iter<T> {
    type Item = ();

    fn next(&mut self) -> Option<()> {
        None
    }
}

impl<T> DoubleEndedIterator for Iter<T> {
    fn next_back(&mut self) -> Option<()> {
        None
    }
}

fn main() {
    #[expect(clippy::double_ended_iterator_last)]
    let _ = Iter::<A<X>>(PhantomData).last();
}
