use crate::vec::Vec;
use core::{
    fmt, mem,
    ops::{Index, Range},
    slice,
};

#[cfg(test)]
mod tests;

/// Fixed Size Queue:
/// A linear queue implemented with a static ring buffer of owned nodes.
///
/// The `FixedQueue` allows pushing and popping elements in constant time.
///
/// The "default" usage of this type is to use [`push`] to add to
/// the queue, and [`pop`] to remove from the queue. Iterating over
/// `FixedQueue` goes front to back.
///
/// A `FixedQueue` with a known list of items can be initialized from an array:
/// ```
/// use alloc::collections::FixedQueue;
///
/// let fque = FixedQueue::from([1, 2, 3]);
/// ```
///
/// Since `FixedQueue` is an array ring buffer, its elements are contiguous
/// in memory.
///
/// [`push`]: FixedQueue::push
/// [`pop`]: FixedQueue::pop
#[derive(Debug)]
#[unstable(feature = "fixed_queue", issue = "126204")]
pub struct FixedQueue<T, const N: usize> {
    buffer: [Option<T>; N],
    head: usize,
    tail: usize,
    len: usize,
}

impl<T, const N: usize> FixedQueue<T, N> {
    /// Create a new FixedQueue with given fields.
    #[inline]
    const fn with(
        buffer: [Option<T>; N],
        head: usize,
        tail: usize,
        len: usize,
    ) -> FixedQueue<T, N> {
        FixedQueue { buffer, head, tail, len }
    }

    /// Create a new FixedQueue with a given capacity.
    #[unstable(feature = "fixed_queue", issue = "126204")]
    pub const fn new() -> FixedQueue<T, N>
    where
        Option<T>: Copy,
    {
        FixedQueue::with([None; N], 0, 0, 0)
    }

    /// Return the max capacity of the FixedQueue.
    #[inline]
    #[unstable(feature = "fixed_queue", issue = "126204")]
    pub const fn capacity(&self) -> usize {
        N
    }

    /// Returns the number of elements in the FixedQueue.
    #[inline]
    #[unstable(feature = "fixed_queue", issue = "126204")]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Check if the queue is empty.
    #[inline]
    #[unstable(feature = "fixed_queue", issue = "126204")]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Check if the queue is full.
    #[inline]
    #[unstable(feature = "fixed_queue", issue = "126204")]
    pub const fn is_full(&self) -> bool {
        self.len == N
    }

    /// Removes all elements from the queue.
    #[unstable(feature = "fixed_queue", issue = "126204")]
    pub fn clear(&mut self) {
        for i in 0..N {
            drop(self.buffer[i].take());
        }
        self.head = 0;
        self.tail = 0;
        self.len = 0;
    }

    /// Fills the queue with an element.
    #[unstable(feature = "fixed_queue", issue = "126204")]
    pub fn fill(&mut self, item: T)
    where
        Option<T>: Copy,
    {
        self.buffer = [Some(item); N];
        self.head = 0;
        self.tail = 0;
        self.len = N;
    }

    /// Add an element to the queue. If queue is full, the first element
    /// is popped and returned.
    #[unstable(feature = "fixed_queue", issue = "126204")]
    pub fn push(&mut self, item: T) -> Option<T> {
        // 'pop' first
        let overwritten = self.buffer[self.tail].take();
        // overwrite head/tail element
        self.buffer[self.tail] = Some(item);
        // shift tail with 'push'
        self.tail = (self.tail + 1) % N;
        if overwritten.is_some() {
            // shift head ptr on collision
            self.head = (self.head + 1) % N;
        } else {
            // increase len if no collision
            self.len += 1;
        }
        return overwritten;
    }

    /// Removes and returns the oldest element from the queue.
    #[inline]
    #[unstable(feature = "fixed_queue", issue = "126204")]
    pub fn pop(&mut self) -> Option<T>
    where
        Option<T>: Copy,
    {
        if self.len == 0 {
            return None;
        }
        let popped = self.buffer[self.head].take();
        self.head = (self.head + 1) % N;
        self.len -= 1;
        popped
    }

    /// Converts the queue into its array equivalent.
    #[unstable(feature = "fixed_queue", issue = "126204")]
    pub fn to_option_array(self) -> [Option<T>; N]
    where
        Option<T>: Copy,
    {
        let mut arr: [Option<T>; N] = [None; N];
        for i in 0..N {
            arr[i] = self.buffer[(self.head + i) % N];
        }
        arr
    }

    /// Converts the queue into its vec equivalent.
    #[unstable(feature = "fixed_queue", issue = "126204")]
    pub fn to_vec(self) -> Vec<T>
    where
        T: Copy,
    {
        let mut vec: Vec<T> = Vec::new();
        for i in 0..N {
            if let Some(e) = self.buffer[(self.head + i) % N] {
                vec.push(e);
            }
        }
        vec
    }
}

#[unstable(feature = "fixed_queue", issue = "126204")]
impl<T, const N: usize> From<[T; N]> for FixedQueue<T, N> {
    /// Creates a FixedQueue from a fixed size array.
    fn from(array: [T; N]) -> Self {
        FixedQueue::with(array.map(Some), 0, 0, N)
    }
}

#[unstable(feature = "fixed_queue", issue = "126204")]
impl<T: Copy, const N: usize> From<&[T; N]> for FixedQueue<T, N> {
    /// Creates a FixedQueue from a fixed size slice.
    fn from(array: &[T; N]) -> Self {
        FixedQueue::with(array.map(Some), 0, 0, N)
    }
}

#[unstable(feature = "fixed_queue", issue = "126204")]
impl<T: Copy, const N: usize> From<&[T]> for FixedQueue<T, N> {
    /// Creates a FixedQueue from an unsized slice. Copies a maximum of N
    /// elements of the slice, and a minimum of the slice length into the
    /// queue. {[0, 0], 0} - [0, 0]
    fn from(array: &[T]) -> Self {
        let mut buf: [Option<T>; N] = [None; N];
        let length = N.min(array.len());
        for i in 0..length {
            buf[i] = Some(array[i]);
        }
        FixedQueue::with(buf, 0, array.len() - 1, length)
    }
}

#[unstable(feature = "fixed_queue", issue = "126204")]
impl<T: PartialEq, const N: usize> PartialEq for FixedQueue<T, N> {
    /// This method tests if a FixedQueue is equal to another FixedQueue.
    fn eq(&self, other: &FixedQueue<T, N>) -> bool {
        if other.len != self.len {
            return false;
        }
        (0..N).all(|x| self.buffer[(self.head + x) % N] == other.buffer[(other.head + x) % N])
    }
}

#[unstable(feature = "fixed_queue", issue = "126204")]
impl<T: PartialEq, const N: usize, const M: usize> PartialEq<[T; M]> for FixedQueue<T, N> {
    /// This method tests if a FixedQueue is equal to a fixed size array.
    fn eq(&self, other: &[T; M]) -> bool {
        if M != self.len {
            return false;
        }
        (0..M).all(|x| self.buffer[(self.head + x) % N].as_ref() == Some(&other[x]))
    }
}

#[unstable(feature = "fixed_queue", issue = "126204")]
impl<T, const N: usize> Index<usize> for FixedQueue<T, N> {
    type Output = Option<T>;

    fn index(&self, index: usize) -> &Self::Output {
        if index >= N {
            panic!("Index out of bounds");
        }
        &self.buffer[(self.head + index) % N]
    }
}

#[unstable(feature = "fixed_queue", issue = "126204")]
impl<T, const N: usize> Index<Range<usize>> for FixedQueue<T, N>
where
    T: Copy + Default,
{
    type Output = [T];

    fn index(&self, range: Range<usize>) -> &Self::Output {
        let start = range.start;
        let end = range.end;

        // check bounds
        assert!(start <= end && end <= self.len, "Index out of bounds");

        // create temporary array to store the results
        let mut temp = Vec::with_capacity(end - start);

        for i in start..end {
            let idx = (self.head + i) % N;
            if let Some(value) = self.buffer[idx] {
                temp.push(value);
            }
        }

        // Return a slice from the temporary array
        // SAFETY: This is safe because temp will live long enough within this function call.
        let result = unsafe { slice::from_raw_parts(temp.as_ptr(), temp.len()) };
        mem::forget(temp);
        result
    }
}

#[unstable(feature = "fixed_queue", issue = "126204")]
impl<T: fmt::Display, const N: usize> fmt::Display for FixedQueue<T, N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.len == 0 {
            return write!(f, "{{}}");
        }
        write!(f, "{{")?;
        for x in 0..(self.len - 1) {
            write!(f, "{}, ", self.buffer[(self.head + x) % N].as_ref().unwrap())?;
        }
        write!(f, "{}}}", self.buffer[(self.head + self.len - 1) % N].as_ref().unwrap())
    }
}
