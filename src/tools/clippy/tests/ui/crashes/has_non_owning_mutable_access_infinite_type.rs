//@ check-pass
#![warn(clippy::double_ended_iterator_last, clippy::needless_collect)]
use std::mem::ManuallyDrop;

pub trait Set {
    type Exp: Set;
}
pub struct Term<T: Set>(T, ManuallyDrop<Box<Term<T::Exp>>>);

pub fn f<T: Set>(v: Vec<Term<T>>) -> Vec<Term<T>> {
    v.into_iter().collect()
}

fn main() {}
