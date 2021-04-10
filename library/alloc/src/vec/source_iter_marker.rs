use core::iter::{InPlaceIterable, SourceIter, TrustedRandomAccess};
use core::mem::{self, ManuallyDrop};
use core::ptr::{self};

use super::{AsIntoIter, SpecFromIter, SpecFromIterNested, Vec};
use core::slice;

/// Specialization marker for collecting an iterator pipeline into a Vec while reusing the
/// source allocation, i.e. executing the pipeline in place.
///
/// The SourceIter parent trait is necessary for the specializing function to access the allocation
/// which is to be reused. But it is not sufficient for the specialization to be valid. See
/// additional bounds on the impl.
#[rustc_unsafe_specialization_marker]
pub(super) trait SourceIterMarker: SourceIter<Source: AsIntoIter> {}

// The std-internal SourceIter/InPlaceIterable traits are only implemented by chains of
// Adapter<Adapter<Adapter<IntoIter>>> (all owned by core/std). Additional bounds
// on the adapter implementations (beyond `impl<I: Trait> Trait for Adapter<I>`) only depend on other
// traits already marked as specialization traits (Copy, TrustedRandomAccess, FusedIterator).
// I.e. the marker does not depend on lifetimes of user-supplied types. Modulo the Copy hole, which
// several other specializations already depend on.
impl<T> SourceIterMarker for T where T: SourceIter<Source: AsIntoIter> + InPlaceIterable {}

impl<T, I> SpecFromIter<T, I> for Vec<T>
where
    I: Iterator<Item = T> + SourceIterMarker,
{
    default fn from_iter(mut iterator: I) -> Self {
        // Additional requirements which cannot expressed via trait bounds. We rely on const eval
        // instead:
        // a) no ZSTs as there would be no allocation to reuse and pointer arithmetic would panic
        // b) size match as required by Alloc contract
        // c) alignments match as required by Alloc contract
        if mem::size_of::<T>() == 0
            || mem::size_of::<T>()
                != mem::size_of::<<<I as SourceIter>::Source as AsIntoIter>::Item>()
            || mem::align_of::<T>()
                != mem::align_of::<<<I as SourceIter>::Source as AsIntoIter>::Item>()
        {
            // fallback to more generic implementations
            return SpecFromIterNested::from_iter(iterator);
        }

        let (src_buf, initial_start_idx, dst_buf, max_len, cap) = unsafe {
            let inner = iterator.as_inner().as_into_iter();
            (
                inner.buf.as_ptr(),
                inner.alive.start,
                inner.buf.as_ptr() as *mut T,
                inner.len(),
                inner.cap,
            )
        };

        let len = SpecInPlaceCollect::collect_in_place(&mut iterator, dst_buf, max_len);

        let src = unsafe { iterator.as_inner().as_into_iter() };
        // check if SourceIter contract was upheld
        // caveat: if they weren't we may not even make it to this point
        debug_assert_eq!(src_buf, src.buf.as_ptr());
        // check InPlaceIterable contract. This is only possible if the iterator modified the
        // `IntoIter::alive` range. If it uses unchecked access via TrustedRandomAccess
        // then the range will keep its initial value and we can't use it for verification
        if src.alive.start != initial_start_idx {
            debug_assert!(
                len <= src.alive.start,
                "InPlaceIterable contract violation, write pointer advanced beyond read pointer"
            );
        }

        // drop any remaining values at the tail of the source
        // but prevent drop of the allocation itself once IntoIter goes out of scope
        // if the drop panics then we also leak any elements collected into dst_buf
        src.forget_allocation_drop_remaining();

        let vec = unsafe { Vec::from_raw_parts(dst_buf, len, cap) };

        vec
    }
}

fn write_in_place_with_drop<T>(
    max_len: usize,
) -> impl FnMut(InPlaceDropByLen<T>, T) -> Result<InPlaceDropByLen<T>, !> {
    move |mut drop_guard, item| {
        unsafe {
            // the InPlaceIterable contract cannot be verified precisely here since
            // try_fold has an exclusive reference to the source pointer
            // all we can do is check if it's still in range
            debug_assert!(drop_guard.len <= max_len, "InPlaceIterable contract violation");
            let dst = drop_guard.ptr.offset(drop_guard.len as isize);
            ptr::write(dst, item);
            drop_guard.len = drop_guard.len.unchecked_add(1);
        }
        Ok(drop_guard)
    }
}

/// Helper trait to hold specialized implementations of the in-place iterate-collect loop
trait SpecInPlaceCollect<T, I>: Iterator<Item = T> {
    /// Collects an iterator (`self`) into the destination buffer (`dst`) and returns the number of items
    /// collected. `end` is the last writable element of the allocation and used for bounds checks.
    fn collect_in_place(&mut self, dst: *mut T, max_len: usize) -> usize;
}

impl<T, I> SpecInPlaceCollect<T, I> for I
where
    I: Iterator<Item = T>,
{
    #[inline]
    default fn collect_in_place(&mut self, dst_buf: *mut T, max_len: usize) -> usize {
        // use try-fold since
        // - it vectorizes better for some iterator adapters
        // - unlike most internal iteration methods, it only takes a &mut self
        // - it lets us thread the write pointer through its innards and get it back in the end
        let drop_guard = InPlaceDropByLen { ptr: dst_buf, len: 0 };
        let drop_guard = self
            .try_fold::<_, _, Result<_, !>>(drop_guard, write_in_place_with_drop(max_len))
            .unwrap();
        // iteration succeeded, don't drop head
        ManuallyDrop::new(drop_guard).len
    }
}

impl<T, I> SpecInPlaceCollect<T, I> for I
where
    I: Iterator<Item = T> + TrustedRandomAccess,
{
    #[inline]
    fn collect_in_place(&mut self, dst_buf: *mut T, max_len: usize) -> usize {
        let len = self.size();
        debug_assert!(len <= max_len, "InPlaceIterable contract violation");
        let mut drop_guard = InPlaceDropByLen { ptr: dst_buf, len: 0 };
        for i in 0..len {
            // Safety: InplaceIterable contract guarantees that for every element we read
            // one slot in the underlying storage will have been freed up and we can immediately
            // write back the result.
            unsafe {
                let dst = dst_buf.offset(i as isize);
                ptr::write(dst, self.__iterator_get_unchecked(i));
                drop_guard.len = i + 1;
            }
        }
        mem::forget(drop_guard);
        len
    }
}

pub struct InPlaceDropByLen<T> {
    ptr: *mut T,
    len: usize,
}

impl<T> Drop for InPlaceDropByLen<T> {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            ptr::drop_in_place(slice::from_raw_parts_mut(self.ptr, self.len));
        }
    }
}
