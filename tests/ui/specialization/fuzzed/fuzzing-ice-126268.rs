// This test previous triggered an assertion that there are no inference variables
// returned by `wf::obligations`. We ended up with an infer var as we failed to
// normalize `R::Elem`.
//
// This assert has now been removed.
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
//~^ ERROR specialization impl does not specialize any associated items

fn main() {}
