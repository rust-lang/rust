//@ compile-flags: -Zvalidate-mir -C opt-level=3
//@ build-pass
//@ edition: 2024

struct TupleIter<T, I: Iterator<Item = T>> {
    inner: I,
}

impl<T, I: Iterator<Item = T>> Iterator for TupleIter<T, I> {
    type Item = (T, T, T);

    fn next(&mut self) -> Option<Self::Item> {
        let inner = &mut self.inner;

        if let Some(first) = inner.next()
            && let Some(second) = inner.next()
            && let Some(third) = inner.next()
        {
            Some((first, second, third))
        } else {
            None
        }
    }
}

fn main() {
    let vec: Vec<u8> = Vec::new();
    let mut tup_iter = TupleIter {
        inner: vec.into_iter(),
    };
    tup_iter.next();
}
