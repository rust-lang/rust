use core::iter::{InPlaceIterable, SourceIter, TrustedRandomAccessNoCoerce};
use core::mem::{self, ManuallyDrop};
use core::ptr::{self};

use super::{AsIntoIter, InPlaceDrop, SpecFromIter, SpecFromIterNested, Vec};

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

        let (src_buf, src_ptr, dst_buf, dst_end, cap) = unsafe {
            let inner = iterator.as_inner().as_into_iter();
            (
                inner.buf.as_ptr(),
                inner.ptr,
                inner.buf.as_ptr() as *mut T,
                inner.end as *const T,
                inner.cap,
            )
        };

        let len = SpecInPlaceCollect::collect_in_place(&mut iterator, dst_buf, dst_end);

        let src = unsafe { iterator.as_inner().as_into_iter() };
        // check if SourceIter contract was upheld
        // caveat: if they weren't we might not even make it to this point
        debug_assert_eq!(src_buf, src.buf.as_ptr());
        // check InPlaceIterable contract. This is only possible if the iterator advanced the
        // source pointer at all. If it uses unchecked access via TrustedRandomAccess
        // then the source pointer will stay in its initial position and we can't use it as reference
        if src.ptr != src_ptr {
            debug_assert!(
                unsafe { dst_buf.add(len) as *const _ } <= src.ptr,
                "InPlaceIterable contract violation, write pointer advanced beyond read pointer"
            );
        }

        // drop any remaining values at the tail of the source
        // but prevent drop of the allocation itself once IntoIter goes out of scope
        // if the drop panics then we also leak any elements collected into dst_buf
        //
        // FIXME: Since `SpecInPlaceCollect::collect_in_place` above might use
        // `__iterator_get_unchecked` internally, this call might be operating on
        // a `vec::IntoIter` with incorrect internal state regarding which elements
        // have already been “consumed”. However, the `TrustedRandomIteratorNoCoerce`
        // implementation of `vec::IntoIter` is only present if the `Vec` elements
        // don’t have a destructor, so it doesn’t matter if elements are “dropped multiple times”
        // in this case.
        // This argument technically currently lacks justification from the `# Safety` docs for
        // `SourceIter`/`InPlaceIterable` and/or `TrustedRandomAccess`, so it might be possible that
        // someone could inadvertently create new library unsoundness
        // involving this `.forget_allocation_drop_remaining()` call.
        src.forget_allocation_drop_remaining();

        let vec = unsafe { Vec::from_raw_parts(dst_buf, len, cap) };

        vec
    }
}

fn write_in_place_with_drop<T>(
    src_end: *const T,
) -> impl FnMut(InPlaceDrop<T>, T) -> Result<InPlaceDrop<T>, !> {
    move |mut sink, item| {
        unsafe {
            // the InPlaceIterable contract cannot be verified precisely here since
            // try_fold has an exclusive reference to the source pointer
            // all we can do is check if it's still in range
            debug_assert!(sink.dst as *const _ <= src_end, "InPlaceIterable contract violation");
            ptr::write(sink.dst, item);
            // Since this executes user code which can panic we have to bump the pointer
            // after each step.
            sink.dst = sink.dst.add(1);
        }
        Ok(sink)
    }
}

/// Helper trait to hold specialized implementations of the in-place iterate-collect loop
trait SpecInPlaceCollect<T, I>: Iterator<Item = T> {
    /// Collects an iterator (`self`) into the destination buffer (`dst`) and returns the number of items
    /// collected. `end` is the last writable element of the allocation and used for bounds checks.
    ///
    /// This method is specialized and one of its implementations makes use of
    /// `Iterator::__iterator_get_unchecked` calls with a `TrustedRandomAccessNoCoerce` bound
    /// on `I` which means the caller of this method must take the safety conditions
    /// of that trait into consideration.
    fn collect_in_place(&mut self, dst: *mut T, end: *const T) -> usize;
}

impl<T, I> SpecInPlaceCollect<T, I> for I
where
    I: Iterator<Item = T>,
{
    #[inline]
    default fn collect_in_place(&mut self, dst_buf: *mut T, end: *const T) -> usize {
        // use try-fold since
        // - it vectorizes better for some iterator adapters
        // - unlike most internal iteration methods, it only takes a &mut self
        // - it lets us thread the write pointer through its innards and get it back in the end
        let sink = InPlaceDrop { inner: dst_buf, dst: dst_buf };
        let sink =
            self.try_fold::<_, _, Result<_, !>>(sink, write_in_place_with_drop(end)).unwrap();
        // iteration succeeded, don't drop head
        unsafe { ManuallyDrop::new(sink).dst.offset_from(dst_buf) as usize }
    }
}

impl<T, I> SpecInPlaceCollect<T, I> for I
where
    I: Iterator<Item = T> + TrustedRandomAccessNoCoerce,
{
    #[inline]
    fn collect_in_place(&mut self, dst_buf: *mut T, end: *const T) -> usize {
        let len = self.size();
        let mut drop_guard = InPlaceDrop { inner: dst_buf, dst: dst_buf };
        for i in 0..len {
            // Safety: InplaceIterable contract guarantees that for every element we read
            // one slot in the underlying storage will have been freed up and we can immediately
            // write back the result.
            unsafe {
                let dst = dst_buf.offset(i as isize);
                debug_assert!(dst as *const _ <= end, "InPlaceIterable contract violation");
                ptr::write(dst, self.__iterator_get_unchecked(i));
                // Since this executes user code which can panic we have to bump the pointer
                // after each step.
                drop_guard.dst = dst.add(1);
            }
        }
        mem::forget(drop_guard);
        len
    }
}
