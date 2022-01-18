use std::collections::VecDeque;
use std::ops::{Index, IndexMut};

/// A view onto a finite range of an infinitely long sequence of T.
///
/// The Ts are indexed 0..infinity. A RingBuffer begins as a view of elements
/// 0..0 (i.e. nothing). The user of the RingBuffer advances its left and right
/// position independently, although only in the positive direction, and only
/// with left <= right at all times.
///
/// Holding a RingBuffer whose view is elements left..right gives the ability to
/// use Index and IndexMut to access elements i in the infinitely long queue for
/// which left <= i < right.
pub struct RingBuffer<T> {
    data: VecDeque<T>,
    // Abstract index of data[0] in the infinitely sized queue.
    offset: usize,
}

impl<T> RingBuffer<T> {
    pub fn new() -> Self {
        RingBuffer { data: VecDeque::new(), offset: 0 }
    }

    pub fn advance_right(&mut self)
    where
        T: Default,
    {
        self.data.push_back(T::default());
    }

    pub fn advance_left(&mut self) {
        self.data.pop_front().unwrap();
        self.offset += 1;
    }

    pub fn truncate(&mut self, len: usize) {
        self.data.truncate(len);
    }
}

impl<T> Index<usize> for RingBuffer<T> {
    type Output = T;
    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index.checked_sub(self.offset).unwrap()]
    }
}

impl<T> IndexMut<usize> for RingBuffer<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[index.checked_sub(self.offset).unwrap()]
    }
}
