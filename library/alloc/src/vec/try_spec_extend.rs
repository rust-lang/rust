use crate::alloc::Allocator;
use crate::alloc::failure_handling::Fallible;
use crate::collections::TryReserveError;
use core::iter::TrustedLen;
use core::slice::{self};

use super::{IntoIter, Vec};

// Specialization trait used for Vec::extend
pub(super) trait TrySpecExtend<T, I> {
    fn try_spec_extend(&mut self, iter: I) -> Result<(), TryReserveError>;
}

impl<T, I, A: Allocator> TrySpecExtend<T, I> for Vec<T, A, Fallible>
where
    I: Iterator<Item = T>,
{
    default fn try_spec_extend(&mut self, iter: I) -> Result<(), TryReserveError> {
        self.extend_desugared(iter)
    }
}

impl<T, I, A: Allocator> TrySpecExtend<T, I> for Vec<T, A, Fallible>
where
    I: TrustedLen<Item = T>,
{
    default fn try_spec_extend(&mut self, iterator: I) -> Result<(), TryReserveError> {
        self.extend_trusted(iterator)
    }
}

impl<T, A: Allocator> TrySpecExtend<T, IntoIter<T>> for Vec<T, A, Fallible> {
    fn try_spec_extend(&mut self, mut iterator: IntoIter<T>) -> Result<(), TryReserveError> {
        unsafe {
            self.append_elements(iterator.as_slice() as _)?;
        }
        iterator.forget_remaining_elements();
        Ok(())
    }
}

impl<'a, T: 'a, I, A: Allocator> TrySpecExtend<&'a T, I> for Vec<T, A, Fallible>
where
    I: Iterator<Item = &'a T>,
    T: Clone,
{
    default fn try_spec_extend(&mut self, iterator: I) -> Result<(), TryReserveError> {
        self.try_spec_extend(iterator.cloned())
    }
}

impl<'a, T: 'a, A: Allocator> TrySpecExtend<&'a T, slice::Iter<'a, T>> for Vec<T, A, Fallible>
where
    T: Copy,
{
    fn try_spec_extend(&mut self, iterator: slice::Iter<'a, T>) -> Result<(), TryReserveError> {
        let slice = iterator.as_slice();
        unsafe { self.append_elements(slice) }
    }
}
