use core::ptr::{self};

/// Returns the two slices that cover the `VecDeque`'s valid range
pub trait RingSlices: Sized {
    fn slice(self, from: usize, to: usize) -> Self;
    fn split_at(self, i: usize) -> (Self, Self);

    fn ring_slices(buf: Self, head: usize, tail: usize) -> (Self, Self) {
        let contiguous = tail <= head;
        if contiguous {
            let (empty, buf) = buf.split_at(0);
            (buf.slice(tail, head), empty)
        } else {
            let (mid, right) = buf.split_at(tail);
            let (left, _) = mid.split_at(head);
            (right, left)
        }
    }
}

impl<T> RingSlices for &[T] {
    fn slice(self, from: usize, to: usize) -> Self {
        &self[from..to]
    }
    fn split_at(self, i: usize) -> (Self, Self) {
        (*self).split_at(i)
    }
}

impl<T> RingSlices for &mut [T] {
    fn slice(self, from: usize, to: usize) -> Self {
        &mut self[from..to]
    }
    fn split_at(self, i: usize) -> (Self, Self) {
        (*self).split_at_mut(i)
    }
}

impl<T> RingSlices for *mut [T] {
    fn slice(self, from: usize, to: usize) -> Self {
        assert!(from <= to && to < self.len());
        // Not using `get_unchecked_mut` to keep this a safe operation.
        let len = to - from;
        ptr::slice_from_raw_parts_mut(self.as_mut_ptr().wrapping_add(from), len)
    }

    fn split_at(self, mid: usize) -> (Self, Self) {
        let len = self.len();
        let ptr = self.as_mut_ptr();
        assert!(mid <= len);
        (
            ptr::slice_from_raw_parts_mut(ptr, mid),
            ptr::slice_from_raw_parts_mut(ptr.wrapping_add(mid), len - mid),
        )
    }
}
