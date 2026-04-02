use core::fmt;
use core::iter::FusedIterator;
use core::mem::ManuallyDrop;

use super::BinaryHeap;
use crate::alloc::{Allocator, Global};

/// An iterator which uses a closure to determine if an element should be removed.
///
/// This struct is created by [`BinaryHeap::extract_if`].
/// See its documentation for more.
///
/// # Example
///
/// ```
/// #![feature(binary_heap_extract_if)]
/// use crate::alloc::collections::BinaryHeap;
///
/// let mut heap: BinaryHeap<u32> = (0..128).collect();
/// let iter: Vec<u32> = heap.extract_if(|x| *x % 2 == 0).collect();
#[unstable(feature = "binary_heap_extract_if", issue = "42849")]
#[must_use = "iterators are lazy and do nothing unless consumed; \
    use `retain_mut` or `extract_if().for_each(drop)` to remove and discard elements"]
pub struct ExtractIf<
    'a,
    T: Ord,
    F,
    #[unstable(feature = "allocator_api", issue = "32838")] A: Allocator = Global,
> {
    heap_ptr: *mut BinaryHeap<T, A>,
    extract_if: ManuallyDrop<super::vec::ExtractIf<'a, T, F, A>>,
}

impl<T: Ord, F, A: Allocator> ExtractIf<'_, T, F, A>
where
    F: FnMut(&mut T) -> bool,
{
    pub(super) fn new<'a>(heap: &'a mut BinaryHeap<T, A>, predicate: F) -> ExtractIf<'a, T, F, A> {
        // We need to keep a reference around to the heap so that we can
        let heap_ptr: *mut BinaryHeap<T, A> = heap;
        let extract_if = ManuallyDrop::new(heap.data.extract_if(.., predicate));

        ExtractIf { heap_ptr, extract_if }
    }
}

#[unstable(feature = "binary_heap_extract_if", issue = "42849")]
impl<T: Ord, F, A: Allocator> Iterator for ExtractIf<'_, T, F, A>
where
    F: FnMut(&mut T) -> bool,
{
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        self.extract_if.next()
    }
}

#[unstable(feature = "binary_heap_extract_if", issue = "42849")]
impl<'a, T: Ord, F, A: Allocator> Drop for ExtractIf<'a, T, F, A> {
    fn drop(&mut self) {
        // SAFETY: We need to drop this before we rebuild the heap so that its descructor resets the vec info
        //      We also are only calling this hear during the drop of ExtractIf and then never using it again
        unsafe {
            ManuallyDrop::drop(&mut self.extract_if);
        }

        // SAFETY: We only generate this ptr from a reference so we know that it is never null
        let heap = unsafe { self.heap_ptr.as_mut_unchecked() };

        // Removing some items from the heap almost certainly has invalidated its invarients, we need to fix this up here
        heap.rebuild();
    }
}

#[unstable(feature = "binary_heap_extract_if", issue = "42849")]
impl<T: Ord, F, A: Allocator> FusedIterator for ExtractIf<'_, T, F, A> where F: FnMut(&mut T) -> bool
{}

#[unstable(feature = "binary_heap_extract_if", issue = "42849")]
impl<T: Ord, F, A> fmt::Debug for ExtractIf<'_, T, F, A>
where
    T: fmt::Debug,
    A: Allocator,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ExtractIf").finish_non_exhaustive()
    }
}
