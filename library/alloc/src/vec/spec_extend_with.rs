use core::clone::TrivialClone;
use core::{mem, ptr};

use crate::alloc::Allocator;
use crate::vec::{SetLenOnDrop, Vec};

pub(super) trait SpecExtendWith<T> {
    /// Extend the vector by `n` clones of value.
    fn spec_extend_with(&mut self, n: usize, value: T);
}

impl<T: Clone, A: Allocator> SpecExtendWith<T> for Vec<T, A> {
    default fn spec_extend_with(&mut self, n: usize, value: T) {
        self.reserve(n);

        // SAFETY: Since `reserve` guarantees `self.len() + n` <= `capacity`, this
        // pointer is always within bounds, and valid for writes for the next `n` items.
        let mut ptr = unsafe { self.as_mut_ptr().add(self.len()) };
        // Use SetLenOnDrop to work around bug where compiler
        // might not realize the store through `ptr` through self.set_len()
        // don't alias.
        let mut local_len = SetLenOnDrop::new(&mut self.len);

        // Write all elements except the last one
        for _ in 1..n {
            // three things have been taken into consideration:
            // - `T::clone` can be provided by the user, and can panic.
            // - `T` may implement Drop.
            // - Since we already have a `value: T`, we can clone `n - 1` times.

            // This loop runs at most `n - 1` times, for all slots except the last one.
            // As per the guarantee of `reserve`, increments to `ptr` within this range won't wrap around.

            // if clone() panics and starts unwinding;
            // - the write has not been done yet
            // - the length will be set by SetLenOnDrop, to the amount of initialized elements
            // - later, when the Vec is dropped, only initialized elements are dropped.
            // SAFETY: According to the diagram above, this write is within bounds.
            unsafe { ptr::write(ptr, value.clone()) };
            // SAFETY: According to the diagram above, this increment is within bounds of the
            // allocation and won't wrap around
            ptr = unsafe { ptr.add(1) };
            // Increment the length of SetLenOnDrop in every step in case clone() panics to drop the
            // already-initialized elements.
            local_len.increment_len(1);
        }

        if n > 0 {
            // We can write the last element directly without cloning needlessly

            // SAFETY: According to the diagram above, the last element is not yet written,
            // so this write is within bounds.
            unsafe { ptr::write(ptr, value) };
            local_len.increment_len(1);
        }

        // len set by scope guard
    }
}

impl<T: TrivialClone, A: Allocator> SpecExtendWith<T> for Vec<T, A> {
    fn spec_extend_with(&mut self, n: usize, value: T) {
        // The `n == 0` check is already being inserted by the loop canonicalization pass,
        // so this costs nothing.
        // https://llvm.org/docs/LoopTerminology.html#more-canonical-loops
        if n == 0 {
            // three things:
            // `self.reserve(0)` does not need to be run, since reserving space for 0 elements
            // is a no-op. since the `value` argument did not have anywhere to be moved to, it is
            // dropped here. with this branch present here, we can elide the write through self.len,
            // since `self.len += 0` would be a no-op.
            return;
        }

        self.reserve(n);

        // SAFETY: Since `reserve` guarantees `self.len() + n` <= `capacity`, this
        // pointer is always within bounds, and valid for writes for the next `n` items.
        let ptr = unsafe { self.as_mut_ptr().add(self.len()) };

        let mut i = 0;
        // Write all the elements by copying `value`.
        // We use an incrementing index here instead of incrementing pointer because the former
        // has been observed to have less register pressure in the loop initialization.
        // https://llvm.org/docs/Vectorizers.html#inductions
        while i < n {
            // SAFETY: `TrivialClone` indicates that copying `value` is equivalent to
            // calling `Clone::clone` for `T`. As per the safety comment for `ptr`, this pointer
            // is within bounds, won't wrap around while `i < n`, and valid for writes. The values
            // do not overlap since `value` is stored as a function argument, and the target is
            // in the Vec.
            unsafe { ptr.add(i).copy_from_nonoverlapping(&value, 1) };

            // Because above is infallible, we don't need to worry about setting the len
            // in this loop, and instead we just increment the loop induction variable.
            i += 1;
        }

        // Above, `n` items have been infallibly copied into the vector
        self.len += n;

        // The reason we did not treat the last position in the vector specially by doing
        // `n - 1` copies and semantically moving `value` into it is that we are forced to do `n`
        // copies of the value no matter what, and it's better to do it in the above loop than
        // by manually writing since it can be vectorized.
        // Then, `value` was never semantically moved into the vec, only copied from,
        // so we forget it here to prevent the Drop.
        mem::forget(value);
    }
}
