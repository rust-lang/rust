// check-pass

#![allow(non_camel_case_types)]

trait HasAssoc {
    type Assoc;
}

trait Iterate<S: HasAssoc> {
    type Iter<'a>
    where
        Self: 'a;
}

struct KeySegment_Broken<T> {
    key: T,
}
impl<S: HasAssoc> Iterate<S> for KeySegment_Broken<S::Assoc> {
    type Iter<'a> = ()
    where
        Self: 'a;
}

fn main() {}
