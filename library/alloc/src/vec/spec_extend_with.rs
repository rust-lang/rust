use core::clone::TrivialClone;
use core::{mem, ptr};

use crate::alloc::Allocator;
use crate::vec::{SetLenOnDrop, Vec};

pub(super) trait SpecExtendWith<T> {
    fn spec_extend_with(&mut self, n: usize, value: T);
}

impl<T: Clone, A: Allocator> SpecExtendWith<T> for Vec<T, A> {
    /// Extend the vector by `n` clones of value.
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
            // This loop runs at most `n - 1` times, for all elements except the last:

            // [a, b, c, d, ..., n - 1, n];
            //  ^ptr             ^----ptr final

            // As per the guarantee of `reserve`, increments to `ptr` within this range won't wrap around.

            // SAFETY: According to the diagram above, this write is within bounds.
            // if clone() panics and starts unwinding, we have not yet written anything nor incremented len,
            // so this is unwind safe.
            unsafe { ptr::write(ptr, value.clone()) };
            // SAFETY: According to the diagram above, this increment is within bounds of the
            // allocation and won't wrap around
            ptr = unsafe { ptr.add(1) };
            // Increment the length in every step in case clone() panics to avoid dropping
            // uninitialized items.
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
        self.reserve(n);

        // SAFETY: Since `reserve` guarantees `self.len() + n` <= `capacity`, this
        // pointer is always within bounds, and valid for writes for the next `n` items.
        let ptr = unsafe { self.as_mut_ptr().add(self.len()) };
        // We use an incrementing index here instead of incrementing pointer because this is
        // a more preferred form of loop for LLVM. https://rocm.docs.amd.com/projects/llvm-project/en/latest/LLVM/llvm/html/LoopTerminology.html#more-canonical-loops
        let mut i = 0;

        // Write all the elements by copying `value`
        while i < n {
            // SAFETY: `TrivialClone` indicates that copying `value` is equivalent to
            // calling `Clone::clone` for `T`. As per the safety comment for `ptr`, this pointer
            // is within bounds, won't wrap around while `i < n`, and valid for
            // writes.
            unsafe { ptr.add(i).write(ptr::read(&value)) };
            i += 1;
        }

        self.len += n;

        if n > 0 {
            // Forget the `value` we never moved in case T: Drop. Treating the last
            // location specially does not benefit us here, since clones for `TrivialClone`
            // are cheap copies, and the branch on `n > 0` is not avoidable.
            mem::forget(value);
        }
    }
}
