//! Inplace iterate-and-collect specialization for `Vec`
//!
//! The specialization in this module applies to iterators in the shape of
//! `source.adapter().adapter().adapter().collect::<Vec<U>>()`
//! where `source` is an owning iterator obtained from [`Vec<T>`], [`Box<[T]>`] (by conversion to `Vec`)
//! or [`BinaryHeap<T>`], the adapters each consume one or more items per step
//! (represented by [`InPlaceIterable`]), provide transitive access to `source` (via [`SourceIter`])
//! and thus the underlying allocation. And finally the layouts of `T` and `U` must
//! have the same size and alignment, this is currently ensured via const eval instead of trait
//! bounds.
//!
//! [`BinaryHeap<T>`]: crate::collections::BinaryHeap
//! [`Box<[T]>`]: crate::boxed::Box
//!
//! By extension some other collections which use `collect::Vec<_>()` internally in their
//! `FromIterator` implementation benefit from this too.
//!
//! Access to the underlying source goes through a further layer of indirection via the private
//! trait [`AsIntoIter`] to hide the implementation detail that other collections may use
//! `vec::IntoIter` internally.
//!
//! In-place iteration depends on the interaction of several unsafe traits, implementation
//! details of multiple parts in the iterator pipeline and often requires holistic reasoning
//! across multiple structs since iterators are executed cooperatively rather than having
//! a central evaluator/visitor struct executing all iterator components.
//!
//! # Reading from and writing to the same allocation
//!
//! By its nature collecting in place means that the reader and writer side of the iterator
//! use the same allocation. Since `fold()` and co. take a reference to the iterator for the
//! duration of the iteration that means we can't interleave the step of reading a value
//! and getting a reference to write to. Instead raw pointers must be used on the reader
//! and writer side.
//!
//! That writes never clobber a yet-to-be-read item is ensured by the [`InPlaceIterable`] requirements.
//!
//! # Layout constraints
//!
//! [`Allocator`] requires that `allocate()` and `deallocate()` have matching alignment and size.
//! Additionally this specialization doesn't make sense for ZSTs as there is no reallocation to
//! avoid and it would make pointer arithmetic more difficult.
//!
//! [`Allocator`]: core::alloc::Allocator
//!
//! # Drop- and panic-safety
//!
//! Iteration can panic, requiring dropping the already written parts but also the remainder of
//! the source. Iteration can also leave some source items unconsumed which must be dropped.
//! All those drops in turn can panic which then must either leak the allocation or abort to avoid
//! double-drops.
//!
//! These tasks are handled by [`InPlaceDrop`] and [`vec::IntoIter::forget_allocation_drop_remaining()`]
//!
//! [`vec::IntoIter::forget_allocation_drop_remaining()`]: super::IntoIter::forget_allocation_drop_remaining()
//!
//! # O(1) collect
//!
//! The main iteration itself is further specialized when the iterator implements
//! [`TrustedRandomAccessNoCoerce`] to let the optimizer see that it is a counted loop with a single
//! induction variable. This can turn some iterators into a noop, i.e. it reduces them from O(n) to
//! O(1). This particular optimization is quite fickle and doesn't always work, see [#79308]
//!
//! [#79308]: https://github.com/rust-lang/rust/issues/79308
//!
//! Since unchecked accesses through that trait do not advance the read pointer of `IntoIter`
//! this would interact unsoundly with the requirements about dropping the tail described above.
//! But since the normal `Drop` implementation of `IntoIter` would suffer from the same problem it
//! is only correct for `TrustedRandomAccessNoCoerce` to be implemented when the items don't
//! have a destructor. Thus that implicit requirement also makes the specialization safe to use for
//! in-place collection.
//!
//! # Adapter implementations
//!
//! The invariants for adapters are documented in [`SourceIter`] and [`InPlaceIterable`], but
//! getting them right can be rather subtle for multiple, sometimes non-local reasons.
//! For example `InPlaceIterable` would be valid to implement for [`Peekable`], except
//! that it is stateful, cloneable and `IntoIter`'s clone implementation shortens the underlying
//! allocation which means if the iterator has been peeked and then gets cloned there no longer is
//! enough room, thus breaking an invariant (#85322).
//!
//! [#85322]: https://github.com/rust-lang/rust/issues/85322
//! [`Peekable`]: core::iter::Peekable
//!
//!
//! # Examples
//!
//! Some cases that are optimized by this specialization, more can be found in the `Vec`
//! benchmarks:
//!
//! ```rust
//! # #[allow(dead_code)]
//! /// Converts a usize vec into an isize one.
//! pub fn cast(vec: Vec<usize>) -> Vec<isize> {
//!   // Does not allocate, free or panic. On optlevel>=2 it does not loop.
//!   // Of course this particular case could and should be written with `into_raw_parts` and
//!   // `from_raw_parts` instead.
//!   vec.into_iter().map(|u| u as isize).collect()
//! }
//! ```
//!
//! ```rust
//! # #[allow(dead_code)]
//! /// Drops remaining items in `src` and if the layouts of `T` and `U` match it
//! /// returns an empty Vec backed by the original allocation. Otherwise it returns a new
//! /// empty vec.
//! pub fn recycle_allocation<T, U>(src: Vec<T>) -> Vec<U> {
//!   src.into_iter().filter_map(|_| None).collect()
//! }
//! ```
//!
//! ```rust
//! let vec = vec![13usize; 1024];
//! let _ = vec.into_iter()
//!   .enumerate()
//!   .filter_map(|(idx, val)| if idx % 2 == 0 { Some(val+idx) } else {None})
//!   .collect::<Vec<_>>();
//!
//! // is equivalent to the following, but doesn't require bounds checks
//!
//! let mut vec = vec![13usize; 1024];
//! let mut write_idx = 0;
//! for idx in 0..vec.len() {
//!    if idx % 2 == 0 {
//!       vec[write_idx] = vec[idx] + idx;
//!       write_idx += 1;
//!    }
//! }
//! vec.truncate(write_idx);
//! ```
use core::iter::{InPlaceIterable, SourceIter, TrustedRandomAccessNoCoerce};
use core::mem::{self, ManuallyDrop};
use core::ptr::{self};

use super::{InPlaceDrop, SpecFromIter, SpecFromIterNested, Vec};

/// Specialization marker for collecting an iterator pipeline into a Vec while reusing the
/// source allocation, i.e. executing the pipeline in place.
#[rustc_unsafe_specialization_marker]
pub(super) trait InPlaceIterableMarker {}

impl<T> InPlaceIterableMarker for T where T: InPlaceIterable {}

impl<T, I> SpecFromIter<T, I> for Vec<T>
where
    I: Iterator<Item = T> + SourceIter<Source: AsIntoIter> + InPlaceIterableMarker,
{
    default fn from_iter(mut iterator: I) -> Self {
        // See "Layout constraints" section in the module documentation. We rely on const
        // optimization here since these conditions currently cannot be expressed as trait bounds
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

        // Drop any remaining values at the tail of the source but prevent drop of the allocation
        // itself once IntoIter goes out of scope.
        // If the drop panics then we also leak any elements collected into dst_buf.
        //
        // Note: This access to the source wouldn't be allowed by the TrustedRandomIteratorNoCoerce
        // contract (used by SpecInPlaceCollect below). But see the "O(1) collect" section in the
        // module documenttation why this is ok anyway.
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

/// Internal helper trait for in-place iteration specialization.
///
/// Currently this is only implemented by [`vec::IntoIter`] - returning a reference to itself - and
/// [`binary_heap::IntoIter`] which returns a reference to its inner representation.
///
/// Since this is an internal trait it hides the implementation detail `binary_heap::IntoIter`
/// uses `vec::IntoIter` internally.
///
/// [`vec::IntoIter`]: super::IntoIter
/// [`binary_heap::IntoIter`]: crate::collections::binary_heap::IntoIter
///
/// # Safety
///
/// In-place iteration relies on implementation details of `vec::IntoIter`, most importantly that
/// it does not create references to the whole allocation during iteration, only raw pointers
#[rustc_specialization_trait]
pub(crate) unsafe trait AsIntoIter {
    type Item;
    fn as_into_iter(&mut self) -> &mut super::IntoIter<Self::Item>;
}
