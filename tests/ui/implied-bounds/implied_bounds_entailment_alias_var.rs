//@ check-pass

trait Data {
    type Elem;
}

impl<F, S: Data<Elem = F>> Data for ArrayBase<S> {
    type Elem = F;
}

struct DatasetIter<'a, R: Data> {
    data: &'a R::Elem,
}

pub struct ArrayBase<S> {
    data: S,
}

trait Trait {
    type Item;
    fn next() -> Option<Self::Item>;
}

impl<'a, D: Data> Trait for DatasetIter<'a, ArrayBase<D>> {
    type Item = ();

    fn next() -> Option<Self::Item> {
        None
    }
}

fn main() {}
