//@ check-pass
#![allow(dead_code)]
struct Foo<T>(T);

struct IntoIter<T>(T);

impl<'a, T: 'a> Iterator for IntoIter<T> {
    type Item = ();

    fn next(&mut self) -> Option<()> {
        None
    }
}

impl<T> IntoIterator for Foo<T> {
    type Item = ();
    type IntoIter = IntoIter<T>;

    fn into_iter(self) -> IntoIter<T> {
        IntoIter(self.0)
    }
}

fn main() {}
