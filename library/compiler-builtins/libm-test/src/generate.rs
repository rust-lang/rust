//! Different generators that can create random or systematic bit patterns.

pub mod case_list;
pub mod edge_cases;
pub mod random;
pub mod spaced;

/// A wrapper to turn any iterator into an `ExactSizeIterator`. Asserts the final result to ensure
/// the provided size was correct.
#[derive(Debug)]
pub struct KnownSize<I> {
    total: u64,
    current: u64,
    iter: I,
}

impl<I> KnownSize<I> {
    pub fn new(iter: I, total: u64) -> Self {
        Self {
            total,
            current: 0,
            iter,
        }
    }
}

impl<I: Iterator> Iterator for KnownSize<I> {
    type Item = I::Item;

    fn next(&mut self) -> Option<Self::Item> {
        let next = self.iter.next();
        if next.is_some() {
            self.current += 1;
            return next;
        }

        assert_eq!(
            self.current, self.total,
            "total items did not match expected"
        );
        None
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = usize::try_from(self.total - self.current).unwrap();
        (remaining, Some(remaining))
    }
}

impl<I: Iterator> ExactSizeIterator for KnownSize<I> {}

/// Yield `(a0, b0), ..., (a0, bn), ..., (an, bn)` for iterators `[a0, ..., an]` and
/// `[b0, ..., bn]`.
fn product2<I0, I1>(i0: I0, i1: I1) -> impl Iterator<Item = (I0::Item, I1::Item)>
where
    I0: Iterator<Item: Copy>,
    I1: Iterator<Item: Copy> + Clone,
{
    i0.flat_map(move |first| i1.clone().map(move |second| (first, second)))
}

/// Yield `(a0, b0, c0), ..., (a0, b0, cn), ..., (a0, bn, cn), ..., (an, bn, cn)` for iterators
/// `[a0, ..., an]`, `[b0, ..., bn]` and `[c0, ..., cn]`.
fn product3<I0, I1, I2>(
    i0: I0,
    i1: I1,
    i2: I2,
) -> impl Iterator<Item = (I0::Item, I1::Item, I2::Item)>
where
    I0: Iterator<Item: Copy>,
    I1: Iterator<Item: Copy> + Clone,
    I2: Iterator<Item: Copy> + Clone,
{
    product2(product2(i0, i1), i2).map(|((first, second), third)| (first, second, third))
}
