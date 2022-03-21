use crate::alloc::Allocator;
use core::iter::TrustedLen;
use core::ptr::{self};
use core::slice::{self};

use super::{IntoIter, SetLenOnDrop, Vec, VecError};

// Specialization trait used for Vec::extend
pub(super) trait SpecExtend<T, I> {
    fn spec_extend<TError: VecError>(&mut self, iter: I) -> Result<(), TError>;
}

impl<T, I, A: Allocator> SpecExtend<T, I> for Vec<T, A>
where
    I: Iterator<Item = T>,
{
    default fn spec_extend<TError: VecError>(&mut self, iter: I) -> Result<(), TError> {
        self.extend_desugared(iter)
    }
}

impl<T, I, A: Allocator> SpecExtend<T, I> for Vec<T, A>
where
    I: TrustedLen<Item = T>,
{
    default fn spec_extend<TError: VecError>(&mut self, iterator: I) -> Result<(), TError> {
        // This is the case for a TrustedLen iterator.
        let (low, high) = iterator.size_hint();
        if let Some(additional) = high {
            debug_assert_eq!(
                low,
                additional,
                "TrustedLen iterator's size hint is not exact: {:?}",
                (low, high)
            );
            self.reserve_impl(additional)?;
            unsafe {
                let mut ptr = self.as_mut_ptr().add(self.len());
                let mut local_len = SetLenOnDrop::new(&mut self.len);
                iterator.for_each(move |element| {
                    ptr::write(ptr, element);
                    ptr = ptr.add(1);
                    // Since the loop executes user code which can panic we have to bump the pointer
                    // after each step.
                    // NB can't overflow since we would have had to alloc the address space
                    local_len.increment_len(1);
                });
            }

            Ok(())
        } else {
            // Per TrustedLen contract a `None` upper bound means that the iterator length
            // truly exceeds usize::MAX, which would eventually lead to a capacity overflow anyway.
            // Since the other branch already panics eagerly (via `reserve()`) we do the same here.
            // This avoids additional codegen for a fallback code path which would eventually
            // panic anyway.
            Err(TError::capacity_overflow())
        }
    }
}

impl<T, A: Allocator> SpecExtend<T, IntoIter<T>> for Vec<T, A> {
    fn spec_extend<TError: VecError>(&mut self, mut iterator: IntoIter<T>) -> Result<(), TError> {
        unsafe {
            self.append_elements(iterator.as_slice() as _)?;
        }
        iterator.forget_remaining_elements();
        Ok(())
    }
}

impl<'a, T: 'a, I, A: Allocator + 'a> SpecExtend<&'a T, I> for Vec<T, A>
where
    I: Iterator<Item = &'a T>,
    T: Clone,
{
    default fn spec_extend<TError: VecError>(&mut self, iterator: I) -> Result<(), TError> {
        self.spec_extend(iterator.cloned())
    }
}

impl<'a, T: 'a, A: Allocator + 'a> SpecExtend<&'a T, slice::Iter<'a, T>> for Vec<T, A>
where
    T: Copy,
{
    fn spec_extend<TError: VecError>(
        &mut self,
        iterator: slice::Iter<'a, T>,
    ) -> Result<(), TError> {
        let slice = iterator.as_slice();
        unsafe { self.append_elements(slice) }
    }
}
