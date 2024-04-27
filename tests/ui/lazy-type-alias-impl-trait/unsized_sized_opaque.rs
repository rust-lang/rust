//@ check-pass

fn main() {}

pub struct PairSlices<'a, 'b, T> {
    pub(crate) a0: &'a mut [T],
    pub(crate) a1: &'a mut [T],
    pub(crate) b0: &'b [T],
    pub(crate) b1: &'b [T],
}

impl<'a, 'b, T> PairSlices<'a, 'b, T> {
    pub fn remainder(self) -> impl Iterator<Item = &'b [T]> {
        IntoIterator::into_iter([self.b0, self.b1])
    }
}
