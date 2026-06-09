//@ check-pass
//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver

use std::ops::Deref;

pub struct TypeFieldIterator<'a, T: 'a> {
    _t: &'a T,
}

pub struct Type<Id, T> {
    _types: Vec<(Id, T)>,
}

impl<'a, Id: 'a, T> Iterator for TypeFieldIterator<'a, T>
where T: Deref<Target = Type<Id, T>> {
    type Item = &'a (Id, T);

    fn next(&mut self) -> Option<&'a (Id, T)> {
        || self.next();
        None
    }
}

fn main() { }
