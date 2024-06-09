use core::{
    fmt::{Debug, Display},
    ops::{Index, Range},
};

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
/// use std::collections::FixedQueue;
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
    pub const fn new() -> FixedQueue<T, N>
    where
        Option<T>: Copy,
    {
        FixedQueue::with([None; N], 0, 0, 0)
    }

    /// Return the max capacity of the FixedQueue.
    #[inline]
    pub const fn capacity(&self) -> usize {
        N
    }

    /// Returns the number of elements in the FixedQueue.
    #[inline]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Check if the queue is empty.
    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Check if the queue is full.
    #[inline]
    pub const fn is_full(&self) -> bool {
        self.len == N
    }

    /// Removes all elements from the queue.
    pub fn clear(&mut self) {
        for i in 0..N {
            drop(self.buffer[i].take());
        }
        self.head = 0;
        self.tail = 0;
        self.len = 0;
    }

    /// Fills the queue with an element.
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

impl<T, const N: usize> From<[T; N]> for FixedQueue<T, N> {
    /// Creates a FixedQueue from a fixed size array.
    fn from(array: [T; N]) -> Self {
        FixedQueue::with(array.map(Some), 0, 0, N)
    }
}

impl<T: Copy, const N: usize> From<&[T; N]> for FixedQueue<T, N> {
    /// Creates a FixedQueue from a fixed size slice.
    fn from(array: &[T; N]) -> Self {
        FixedQueue::with(array.map(Some), 0, 0, N)
    }
}

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

impl<T: PartialEq, const N: usize> PartialEq for FixedQueue<T, N> {
    /// This method tests if a FixedQueue is equal to another FixedQueue.
    fn eq(&self, other: &FixedQueue<T, N>) -> bool {
        if other.len != self.len {
            return false;
        }
        (0..N).all(|x| self.buffer[(self.head + x) % N] == other.buffer[(other.head + x) % N])
    }
}

impl<T: PartialEq, const N: usize, const M: usize> PartialEq<[T; M]> for FixedQueue<T, N> {
    /// This method tests if a FixedQueue is equal to a fixed size array.
    fn eq(&self, other: &[T; M]) -> bool {
        if M != self.len {
            return false;
        }
        (0..M).all(|x| self.buffer[(self.head + x) % N].as_ref() == Some(&other[x]))
    }
}

impl<T, const N: usize> Index<usize> for FixedQueue<T, N> {
    type Output = Option<T>;

    fn index(&self, index: usize) -> &Self::Output {
        if index >= N {
            panic!("Index out of bounds");
        }
        &self.buffer[(self.head + index) % N]
    }
}

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
        let result = unsafe { std::slice::from_raw_parts(temp.as_ptr(), temp.len()) };
        std::mem::forget(temp);
        result
    }
}

impl<T: Display, const N: usize> Display for FixedQueue<T, N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
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

#[cfg(test)]
mod test {
    use super::FixedQueue;

    #[test]
    fn with() {
        let x = FixedQueue::<usize, 3>::with([None; 3], 789, 456, 123);
        let y = FixedQueue::<usize, 3> { buffer: [None; 3], head: 789, tail: 456, len: 123 };
        assert_eq!(x, y);
    }

    #[test]
    fn partial_eq_self() {
        let x = FixedQueue::<usize, 3>::new();
        let mut y = FixedQueue::with([None; 3], 0, 0, 0);
        assert_eq!(x, y);
        y.push(1);
        assert_ne!(x, y);
        y.push(2);
        assert_ne!(x, y);
        y.push(3);
        assert_ne!(x, y);
        y.clear();
        assert_eq!(x, y);
        let w = FixedQueue::<usize, 3>::with([None; 3], 0, 0, 0);
        let z = FixedQueue::<usize, 3>::with([None; 3], 1, 1, 0);
        assert_eq!(w, z);
        let u = FixedQueue::<usize, 3>::with([Some(20), None, None], 0, 1, 1);
        let v = FixedQueue::<usize, 3>::with([None, Some(20), None], 1, 2, 1);
        assert_eq!(u, v);
    }

    #[test]
    fn partial_eq_array() {
        let x = FixedQueue::<usize, 3>::from([1, 2, 3]);
        assert_eq!(x, [1, 2, 3]);
        assert_ne!(x, [20, 2, 3]);
        let y = FixedQueue::<usize, 1>::from([80]);
        assert_eq!(y, [80]);
        let z = FixedQueue::<usize, 3>::with([Some(1), Some(2), Some(3)], 1, 1, 3);
        assert_eq!(z, [2, 3, 1]);
        let w = FixedQueue::<usize, 3>::with([Some(20), None, None], 0, 1, 1);
        assert_eq!(w, [20]);
        let u = FixedQueue::<usize, 3>::with([None, Some(20), None], 1, 2, 1);
        assert_eq!(u, [20]);
    }

    #[test]
    fn new() {
        let x = FixedQueue::<usize, 3>::new();
        let y = FixedQueue::<usize, 3>::with([None; 3], 0, 0, 0);
        assert_eq!(x, y);
    }

    #[test]
    fn from_array() {
        let x = FixedQueue::from([1i32, 2i32, 3i32]);
        let y = FixedQueue::<i32, 3>::with([Some(1i32), Some(2i32), Some(3i32)], 0, 0, 3);
        assert_eq!(x, y);
        let z = FixedQueue::from([true, false, true]);
        let w = FixedQueue::<bool, 3>::with([Some(true), Some(false), Some(true)], 0, 0, 3);
        assert_eq!(z, w);
    }

    #[test]
    fn from_sized_slice() {
        let x = FixedQueue::from(&[3i32, 2i32, 1i32]);
        let y = FixedQueue::<i32, 3>::with([Some(3i32), Some(2i32), Some(1i32)], 0, 0, 3);
        assert_eq!(x, y);
    }

    #[test]
    fn from_slice() {
        let array = [3i32, 2i32, 1i32];
        let x = FixedQueue::<i32, 1>::from(&array[0..1]);
        let y = FixedQueue::<i32, 1>::with([Some(3i32)], 0, 0, 1);
        assert_eq!(x, y);
        let w = FixedQueue::<i32, 2>::from(&array[0..2]);
        let z = FixedQueue::<i32, 2>::with([Some(3i32), Some(2i32)], 0, 0, 2);
        assert_eq!(w, z);
        let u = FixedQueue::<i32, 3>::from(&array[0..3]);
        let v = FixedQueue::<i32, 3>::with([Some(3i32), Some(2i32), Some(1i32)], 0, 0, 3);
        assert_eq!(u, v);
        let s = FixedQueue::<i32, 3>::from(&array[..]);
        let t = FixedQueue::<i32, 3>::with([Some(3i32), Some(2i32), Some(1i32)], 0, 0, 3);
        assert_eq!(s, t);
    }

    #[test]
    fn index() {
        let x = FixedQueue::<&str, 3>::from(["a", "b", "c"]);
        assert_eq!(x[0], Some("a"));
        assert_eq!(x[1], Some("b"));
        assert_eq!(x[2], Some("c"));
    }

    #[test]
    fn index_range() {
        let x = FixedQueue::<&str, 3>::from(["a", "b", "c"]);
        assert!(x[0..0].is_empty());
        assert_eq!(x[0..1], ["a"]);
        assert_eq!(x[0..2], ["a", "b"]);
        assert_eq!(x[0..3], ["a", "b", "c"]);
    }

    #[test]
    fn display() {
        let mut x = FixedQueue::<usize, 3>::new();
        assert_eq!(format!("{}", x), String::from("{}"));
        x.push(10);
        assert_eq!(format!("{}", x), String::from("{10}"));
        x.pop();
        assert_eq!(format!("{}", x), String::from("{}"));
        x.push(20);
        assert_eq!(format!("{}", x), String::from("{20}"));
        x.push(30);
        assert_eq!(format!("{}", x), String::from("{20, 30}"));
        x.push(40);
        assert_eq!(format!("{}", x), String::from("{20, 30, 40}"));
        x.push(50);
        assert_eq!(format!("{}", x), String::from("{30, 40, 50}"));
        x.pop();
        assert_eq!(format!("{}", x), String::from("{40, 50}"));
        x.pop();
        assert_eq!(format!("{}", x), String::from("{50}"));
        x.pop();
        assert_eq!(format!("{}", x), String::from("{}"));
    }

    #[test]
    fn capacity() {
        let x = FixedQueue::<usize, 1>::new();
        assert_eq!(x.capacity(), 1);
        let y = FixedQueue::<usize, 2>::new();
        assert_eq!(y.capacity(), 2);
        let z = FixedQueue::<usize, 3>::new();
        assert_eq!(z.capacity(), 3);
    }

    #[test]
    fn len() {
        let mut x = FixedQueue::<bool, 3>::new();
        assert_eq!(x.len(), 0);
        x.push(true);
        assert_eq!(x.len(), 1);
        x.push(false);
        assert_eq!(x.len(), 2);
        x.push(true);
        assert_eq!(x.len(), 3);
        x.pop();
        assert_eq!(x.len(), 2);
        x.pop();
        assert_eq!(x.len(), 1);
        x.pop();
        assert_eq!(x.len(), 0);
    }

    #[test]
    fn is_empty() {
        let mut x = FixedQueue::<usize, 3>::new();
        assert!(x.is_empty());
        x.push(1);
        assert!(!x.is_empty());
    }

    #[test]
    fn is_full() {
        let mut x = FixedQueue::<usize, 3>::new();
        assert!(!x.is_full());
        x.push(1);
        assert!(!x.is_full());
        x.push(1);
        assert!(!x.is_full());
        x.push(1);
        assert!(x.is_full());
    }

    #[test]
    fn clear() {
        let mut x = FixedQueue::from([1, 2, 3]);
        assert!(!x.is_empty());
        x.clear();
        assert!(x.is_empty());
    }

    #[test]
    fn fill() {
        let mut x = FixedQueue::<usize, 3>::new();
        assert!(!x.is_full());
        x.fill(10);
        assert!(x.is_full());
    }

    #[test]
    fn push() {
        let mut x = FixedQueue::<String, 2>::from([(); 2].map(|_| String::new()));
        assert!(x.is_full());
        x.clear();
        assert!(x.is_empty());
        x.push(String::from("a"));
        assert_eq!(x.len(), 1);
        assert_eq!(x, [String::from("a")]);
        x.push(String::from("b"));
        assert_eq!(x.len(), 2);
        assert_eq!(x, [String::from("a"), String::from("b")]);
        x.push(String::from("c"));
        assert_eq!(x.len(), 2);
        assert_eq!(x, [String::from("b"), String::from("c")]);
        x.push(String::from("d"));
        assert_eq!(x.len(), 2);
        assert_eq!(x, [String::from("c"), String::from("d")]);
        x.push(String::from("e"));
        assert_eq!(x.len(), 2);
        assert_eq!(x, [String::from("d"), String::from("e")]);
    }

    #[test]
    fn pop() {
        let mut u = FixedQueue::<&str, 3>::from(["a", "b", "c"]);
        assert!(!u.is_empty());
        let w = u.pop();
        assert_eq!(w, Some("a"));
        assert_eq!(u, ["b", "c"]);
        let x = u.pop();
        assert_eq!(x, Some("b"));
        assert_eq!(u, ["c"]);
        let y = u.pop();
        assert_eq!(y, Some("c"));
        assert_eq!(u, []);
        let z = u.pop();
        assert_eq!(z, None);
        assert_eq!(u, []);
    }

    #[test]
    fn to_option_array() {
        let x = FixedQueue::<&str, 3>::from(["a", "b", "c"]);
        assert_eq!(x.to_option_array(), [Some("a"), Some("b"), Some("c")]);
        let mut y = FixedQueue::<&str, 3>::from(["a", "b", "c"]);
        y.pop();
        assert_eq!(y.to_option_array(), [Some("b"), Some("c"), None])
    }

    #[test]
    fn to_vec() {
        let x = FixedQueue::<&str, 3>::from(["a", "b", "c"]);
        let y = Vec::from(["a", "b", "c"]);
        assert_eq!(x.to_vec(), y);
    }
}
