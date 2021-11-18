use crate::cmp::Ordering;
use crate::iter::{FusedIterator, TrustedLen, TrustedRandomAccess, TrustedRandomAccessNoCoerce};
use crate::mem;
use crate::ops::Range;

macro_rules! impl_iterator_for_ptr_range {
    ($mutability:ident /* const or mut */) => {
        /// Iteration of a pointer range, as is common in code that interfaces
        /// with C++ iterators.
        ///
        /// # Safety
        ///
        /// Traversing a pointer range is always safe, but **using the resulting
        /// pointers** is not!
        ///
        /// The pointers between the start and end of a range "remember" the
        /// [allocated object] that they refer into. Pointers resulting from
        /// pointer arithmetic must not be used to read or write to any other
        /// allocated object.
        ///
        /// As a consequence, pointers from a range traversal are only
        /// dereferenceable if start and end of the original range both point
        /// into the same allocated object. Dereferencing a pointer obtained via
        /// iteration when this is not the case is Undefined Behavior.
        ///
        /// [allocated object]: crate::ptr#allocated-object
        ///
        /// # Example
        ///
        #[doc = example!($mutability)]
        #[stable(feature = "iterate_ptr_range", since = "1.58.0")]
        impl<T> Iterator for Range<*$mutability T> {
            type Item = *$mutability T;

            fn next(&mut self) -> Option<Self::Item> {
                if self.is_empty() {
                    None
                } else {
                    let curr = self.start;
                    let next = curr.wrapping_add(1);
                    self.start = if (curr..self.end).contains(&next) {
                        next
                    } else {
                        // Saturate to self.end if the wrapping_add wrapped or
                        // landed beyond end.
                        self.end
                    };
                    Some(curr)
                }
            }

            fn size_hint(&self) -> (usize, Option<usize>) {
                if self.is_empty() {
                    (0, Some(0))
                } else if mem::size_of::<T>() == 0 {
                    // T is zero sized so there are infinity of them in the
                    // nonempty range.
                    (usize::MAX, None)
                } else {
                    // In between self.start and self.end there are some number
                    // of whole elements of type T, followed by possibly a
                    // remainder element if T's size doesn't evenly divide the
                    // byte distance between the endpoints. The remainder
                    // element still counts as being part of this range, since
                    // the pointer to it does lie between self.start and
                    // self.end.
                    let byte_offset = self.end as usize - self.start as usize;
                    let number_of_whole_t = byte_offset / mem::size_of::<T>();
                    let remainder_bytes = byte_offset % mem::size_of::<T>();
                    let maybe_remainder_t = (remainder_bytes > 0) as usize;
                    let hint = number_of_whole_t + maybe_remainder_t;
                    (hint, Some(hint))
                }
            }

            fn nth(&mut self, n: usize) -> Option<Self::Item> {
                let _ = self.advance_by(n);
                self.next()
            }

            fn last(mut self) -> Option<Self::Item> {
                self.next_back()
            }

            fn min(mut self) -> Option<Self::Item> {
                self.next()
            }

            fn max(mut self) -> Option<Self::Item> {
                self.next_back()
            }

            fn is_sorted(self) -> bool {
                true
            }

            fn advance_by(&mut self, n: usize) -> Result<(), usize> {
                match self.size_hint().1 {
                    None => {
                        // T is zero sized. Advancing does nothing.
                        Ok(())
                    }
                    Some(len) => match n.cmp(&len) {
                        Ordering::Less => {
                            // Advance past n number of whole elements.
                            self.start = self.start.wrapping_add(n);
                            Ok(())
                        }
                        Ordering::Equal => {
                            // Advance past every single element in the
                            // iterator, including perhaps the remainder
                            // element, leaving an empty iterator.
                            self.start = self.end;
                            Ok(())
                        }
                        Ordering::Greater => {
                            // Advance too far.
                            self.start = self.end;
                            Err(len)
                        }
                    }
                }
            }

            #[doc(hidden)]
            unsafe fn __iterator_get_unchecked(&mut self, idx: usize) -> Self::Item {
                self.start.wrapping_add(idx)
            }
        }

        #[stable(feature = "iterate_ptr_range", since = "1.58.0")]
        impl<T> DoubleEndedIterator for Range<*$mutability T> {
            fn next_back(&mut self) -> Option<Self::Item> {
                match self.size_hint().1 {
                    None => {
                        // T is zero sized so the iterator never progresses past
                        // start, even if going backwards.
                        Some(self.start)
                    }
                    Some(0) => {
                        None
                    }
                    Some(len) => {
                        self.end = self.start.wrapping_add(len - 1);
                        Some(self.end)
                    }
                }
            }

            fn nth_back(&mut self, n: usize) -> Option<Self::Item> {
                match self.size_hint().1 {
                    None => {
                        // T is zero sized.
                        Some(self.start)
                    }
                    Some(len) => {
                        if n < len {
                            self.end = self.start.wrapping_add(len - n - 1);
                            Some(self.end)
                        } else {
                            self.end = self.start;
                            None
                        }
                    }
                }
            }

            fn advance_back_by(&mut self, n: usize) -> Result<(), usize> {
                match self.size_hint().1 {
                    None => {
                        // T is zero sized. Advancing does nothing.
                        Ok(())
                    }
                    Some(len) => match n.cmp(&len) {
                        Ordering::Less => {
                            // Advance leaving `len - n` elements in the
                            // iterator. Careful to preserve the remainder
                            // element if told to advance by 0.
                            if n > 0 {
                                self.end = self.start.wrapping_add(len - n);
                            }
                            Ok(())
                        }
                        Ordering::Equal => {
                            // Advance past every single element in the
                            // iterator, leaving an empty iterator.
                            self.end = self.start;
                            Ok(())
                        }
                        Ordering::Greater => {
                            // Advance too far.
                            self.end = self.start;
                            Err(len)
                        }
                    }
                }
            }
        }

        #[stable(feature = "iterate_ptr_range", since = "1.58.0")]
        impl<T> FusedIterator for Range<*$mutability T> {}

        #[unstable(feature = "trusted_len", issue = "37572")]
        unsafe impl<T> TrustedLen for Range<*$mutability T> {}

        #[doc(hidden)]
        #[unstable(feature = "trusted_random_access", issue = "none")]
        unsafe impl<T> TrustedRandomAccess for Range<*$mutability T> {}

        #[doc(hidden)]
        #[unstable(feature = "trusted_random_access", issue = "none")]
        unsafe impl<T> TrustedRandomAccessNoCoerce for Range<*$mutability T> {
            const MAY_HAVE_SIDE_EFFECT: bool = false;
        }
    };
}

macro_rules! example {
    (const) => {
        doc_comment_to_literal! {
            /// ```
            /// // Designed to be called from C++ or C.
            /// #[no_mangle]
            /// unsafe extern "C" fn demo(start: *const u16, end: *const u16) {
            ///     for ptr in start..end {
            ///         println!("{}", *ptr);
            ///     }
            /// }
            ///
            /// fn main() {
            ///     let slice = &[1u16, 2, 3];
            ///     let range = slice.as_ptr_range();
            ///     unsafe { demo(range.start, range.end); }
            /// }
            /// ```
        }
    };

    (mut) => {
        doc_comment_to_literal! {
            /// ```
            /// #![feature(vec_spare_capacity)]
            ///
            /// use core::ptr;
            ///
            /// // Designed to be called from C++ or C.
            /// #[no_mangle]
            /// unsafe extern "C" fn demo(start: *mut u16, end: *mut u16) {
            ///     for (i, ptr) in (start..end).enumerate() {
            ///         ptr::write(ptr, i as u16);
            ///     }
            /// }
            ///
            /// fn main() {
            ///     let mut vec: Vec<u16> = Vec::with_capacity(100);
            ///     let range = vec.spare_capacity_mut().as_mut_ptr_range();
            ///     unsafe {
            ///         demo(range.start.cast::<u16>(), range.end.cast::<u16>());
            ///         vec.set_len(100);
            ///     }
            /// }
            /// ```
        }
    };
}

macro_rules! doc_comment_to_literal {
    ($(#[doc = $example:literal])*) => {
        concat!($($example, '\n'),*)
    };
}

impl_iterator_for_ptr_range!(const);
impl_iterator_for_ptr_range!(mut);
