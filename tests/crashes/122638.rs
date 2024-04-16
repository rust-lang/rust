//@ known-bug: #122638
#![feature(min_specialization)]

impl<'a, T: std::fmt::Debug, const N: usize> Iterator for ConstChunksExact<'a, T, { N }> {
    fn next(&mut self) -> Option<Self::Item> {}
}

struct ConstChunksExact<'a, T: '_, const assert: usize> {}

impl<'a, T: std::fmt::Debug, const N: usize> Iterator for ConstChunksExact<'a, T, {}> {
    type Item = &'a [T; N];
}
