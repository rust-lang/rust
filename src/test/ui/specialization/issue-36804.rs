// check-pass
#![feature(specialization)]

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
