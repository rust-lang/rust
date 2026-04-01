//@ known-bug: #126268
#![feature(min_specialization)]

trait Trait {}

impl<T> Trait for T {}

trait Data {
    type Elem;
}

struct DatasetIter<'a, R: Data> {
    data: &'a R::Elem,
}

pub struct ArrayBase {}

impl<'a> Trait for DatasetIter<'a, ArrayBase> {}
