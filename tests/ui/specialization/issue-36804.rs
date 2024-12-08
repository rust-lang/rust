//@ check-pass
#![feature(specialization)] //~ WARN the feature `specialization` is incomplete

pub struct Cloned<I>(I);

impl<'a, I, T: 'a> Iterator for Cloned<I>
where
    I: Iterator<Item = &'a T>,
    T: Clone,
{
    type Item = T;

    fn next(&mut self) -> Option<T> {
        unimplemented!()
    }

    default fn count(self) -> usize where Self: Sized {
        self.fold(0, |cnt, _| cnt + 1)
    }
}

impl<'a, I, T: 'a> Iterator for Cloned<I>
where
    I: Iterator<Item = &'a T>,
    T: Copy,
{
    fn count(self) -> usize {
        unimplemented!()
    }
}

fn main() {
    let a = [1,2,3,4];
    Cloned(a.iter()).count();
}
