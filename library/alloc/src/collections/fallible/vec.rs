use core::mem::MaybeUninit;
use core::ops::{self, Index, IndexMut};
use core::slice::{self, SliceIndex};

use super::AllocError;
use crate::alloc::{Allocator, Global};
use crate::vec::{IntoIter, Vec as InfallibleVec};

/// A fallible alternative to [`Vec`][InfallibleVec].
///
/// All allocating methods will return [`AllocError`][super::AllocError] if allocation fails.
/// Otherwise are the same as their infallible `Vec` counterparts.
///
/// Both `fallible::Vec` and `Vec` are fully ABI compatible, which means you can freely cast between with no cost.
#[repr(transparent)]
#[rustc_insignificant_dtor]
#[unstable(feature = "fallible_vec", issue = "157392")]
#[derive(Debug, Clone, Hash, Eq, PartialEq, Ord, PartialOrd)]
pub struct FallibleVec<
    T,
    #[unstable(feature = "allocator_api", issue = "32838")] A: Allocator = Global,
> {
    buf: InfallibleVec<T, A>,
}

////////////////////////////////////////////////////////////////////////////////
// Inherent methods
////////////////////////////////////////////////////////////////////////////////

#[unstable(feature = "fallible_vec", issue = "157392")]
impl<T> FallibleVec<T> {
    /// Constructs a new, empty `FallibleVec<T>`.
    ///
    /// The vector will not allocate until elements are pushed onto it.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(fallible_vec)]
    /// # #![allow(unused_mut)]
    /// use std::collections::fallible::FallibleVec;
    /// let mut vec: FallibleVec<i32> = FallibleVec::new();
    /// ```
    #[inline]
    #[must_use]
    #[rustc_const_unstable(feature = "fallible_vec", issue = "157392")]
    pub const fn new() -> Self {
        InfallibleVec::new().to_fallible()
    }

    /// Constructs a new, empty `FallibleVec<T>` with at least the specified capacity.
    ///
    /// The vector will be able to hold at least `capacity` elements without
    /// reallocating. This method is allowed to allocate for more elements than
    /// `capacity`. If `capacity` is zero, the vector will not allocate.
    ///
    /// # Errors
    ///
    /// Returns an error if the capacity exceeds `isize::MAX` _bytes_,
    /// or if the allocator reports allocation failure.
    #[inline]
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Result<Self, AllocError> {
        InfallibleVec::try_with_capacity(capacity).map(|v| v.to_fallible())
    }

    /// Creates a `FallibleVec<T>` directly from a pointer, a length, and a capacity.
    ///
    /// # Safety
    ///
    /// This is highly unsafe, due to the number of invariants that aren't
    /// checked:
    ///
    /// * If `T` is not a zero-sized type and the capacity is nonzero, `ptr` must have
    ///   been allocated using the global allocator, such as via the [`alloc::alloc`]
    ///   function. If `T` is a zero-sized type or the capacity is zero, `ptr` need
    ///   only be non-null and aligned.
    /// * `T` needs to have the same alignment as what `ptr` was allocated with,
    ///   if the pointer is required to be allocated.
    ///   (`T` having a less strict alignment is not sufficient, the alignment really
    ///   needs to be equal to satisfy the [`dealloc`] requirement that memory must be
    ///   allocated and deallocated with the same layout.)
    /// * The size of `T` times the `capacity` (i.e. the allocated size in bytes), if
    ///   nonzero, needs to be the same size as the pointer was allocated with.
    ///   (Because similar to alignment, [`dealloc`] must be called with the same
    ///   layout `size`.)
    /// * `length` needs to be less than or equal to `capacity`.
    /// * The first `length` values must be properly initialized values of type `T`.
    /// * `capacity` needs to be the capacity that the pointer was allocated with,
    ///   if the pointer is required to be allocated.
    /// * The allocated size in bytes must be no larger than `isize::MAX`.
    ///   See the safety documentation of [`pointer::offset`].
    ///
    /// These requirements are always upheld by any `ptr` that has been allocated
    /// via `FallibleVec<T>`. Other allocation sources are allowed if the invariants are
    /// upheld.
    ///
    /// Violating these may cause problems like corrupting the allocator's
    /// internal data structures. For example it is normally **not** safe
    /// to build a `FallibleVec<u8>` from a pointer to a C `char` array with length
    /// `size_t`, doing so is only safe if the array was initially allocated by
    /// a `FallibleVec` or `String`.
    /// It's also not safe to build one from a `FallibleVec<u16>` and its length, because
    /// the allocator cares about the alignment, and these two types have different
    /// alignments. The buffer was allocated with alignment 2 (for `u16`), but after
    /// turning it into a `FallibleVec<u8>` it'll be deallocated with alignment 1. To avoid
    /// these issues, it is often preferable to do casting/transmuting using
    /// [`slice::from_raw_parts`] instead.
    ///
    /// The ownership of `ptr` is effectively transferred to the
    /// `FallibleVec<T>` which may then deallocate, reallocate or change the
    /// contents of memory pointed to by the pointer at will. Ensure
    /// that nothing else uses the pointer after calling this
    /// function.
    ///
    /// [`String`]: crate::string::String
    /// [`alloc::alloc`]: crate::alloc::alloc
    /// [`dealloc`]: crate::alloc::GlobalAlloc::dealloc
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(fallible_vec)]
    ///
    /// use std::collections::fallible::FallibleVec;
    /// use std::ptr;
    ///
    /// let v = vec![1, 2, 3].to_fallible();
    ///
    /// // Deconstruct the vector into parts.
    /// let (p, len, cap) = v.into_raw_parts();
    ///
    /// unsafe {
    ///     // Overwrite memory with 4, 5, 6
    ///     for i in 0..len {
    ///         ptr::write(p.add(i), 4 + i);
    ///     }
    ///
    ///     // Put everything back together into a FallibleVec
    ///     let rebuilt = FallibleVec::from_raw_parts(p, len, cap);
    ///     assert_eq!(rebuilt, [4, 5, 6]);
    /// }
    /// ```
    ///
    /// Using memory that was allocated elsewhere:
    ///
    /// ```rust
    /// #![feature(fallible_vec)]
    /// use std::alloc::{alloc, Layout};
    /// use std::collections::fallible::FallibleVec;
    ///
    /// fn main() {
    ///     let layout = Layout::array::<u32>(16).expect("overflow cannot happen");
    ///
    ///     let vec = unsafe {
    ///         let mem = alloc(layout).cast::<u32>();
    ///         if mem.is_null() {
    ///             return;
    ///         }
    ///
    ///         mem.write(1_000_000);
    ///
    ///         FallibleVec::from_raw_parts(mem, 1, 16)
    ///     };
    ///
    ///     assert_eq!(vec, &[1_000_000]);
    ///     assert_eq!(vec.capacity(), 16);
    /// }
    /// ```
    #[inline]
    #[rustc_const_unstable(feature = "const_heap", issue = "79597")]
    pub const unsafe fn from_raw_parts(ptr: *mut T, length: usize, capacity: usize) -> Self {
        unsafe { Self::from_raw_parts_in(ptr, length, capacity, Global) }
    }

    /// Decomposes a `FallibleVec<T>` into its raw components: `(pointer, length, capacity)`.
    ///
    /// Returns the raw pointer to the underlying data, the length of
    /// the vector (in elements), and the allocated capacity of the
    /// data (in elements). These are the same arguments in the same
    /// order as the arguments to [`from_raw_parts`].
    ///
    /// After calling this function, the caller is responsible for the
    /// memory previously managed by the `FallibleVec`. Most often, one does
    /// this by converting the raw pointer, length, and capacity back
    /// into a `FallibleVec` with the [`from_raw_parts`] function; more generally,
    /// if `T` is non-zero-sized and the capacity is nonzero, one may use
    /// any method that calls [`dealloc`] with a layout of
    /// `Layout::array::<T>(capacity)`; if `T` is zero-sized or the
    /// capacity is zero, nothing needs to be done.
    ///
    /// [`from_raw_parts`]: FallibleVec::from_raw_parts
    /// [`dealloc`]: crate::alloc::GlobalAlloc::dealloc
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(fallible_vec)]
    /// use std::collections::fallible::FallibleVec;
    /// let v: FallibleVec<i32> = vec![-1, 0, 1].to_fallible();
    ///
    /// let (ptr, len, cap) = v.into_raw_parts();
    ///
    /// let rebuilt = unsafe {
    ///     // We can now make changes to the components, such as
    ///     // transmuting the raw pointer to a compatible type.
    ///     let ptr = ptr as *mut u32;
    ///
    ///     FallibleVec::from_raw_parts(ptr, len, cap)
    /// };
    /// assert_eq!(rebuilt, [4294967295, 0, 1]);
    /// ```
    #[must_use = "losing the pointer will leak memory"]
    #[rustc_const_unstable(feature = "const_heap", issue = "79597")]
    pub const fn into_raw_parts(self) -> (*mut T, usize, usize) {
        self.buf.into_raw_parts()
    }
}

impl<T, A: Allocator> InfallibleVec<T, A> {
    /// Cast a `Vec` to a `FallibleVec`.
    #[unstable(feature = "fallible_vec", issue = "157392")]
    #[rustc_const_unstable(feature = "fallible_vec", issue = "157392")]
    pub const fn to_fallible(self) -> FallibleVec<T, A> {
        FallibleVec { buf: self }
    }
}

#[unstable(feature = "fallible_vec", issue = "157392")]
impl<T, A: Allocator> FallibleVec<T, A> {
    /// Cast a `FallibleVec` to a `Vec`.
    #[rustc_const_unstable(feature = "fallible_vec", issue = "157392")]
    #[inline]
    pub const fn to_infallible(self) -> InfallibleVec<T, A> {
        let FallibleVec { buf } = self;
        buf
    }

    /// Constructs a new, empty `FallibleVec<T, A>`.
    ///
    /// The vector will not allocate until elements are pushed onto it.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(fallible_vec)]
    /// #![feature(allocator_api)]
    ///
    /// use std::collections::fallible::FallibleVec;
    /// use std::alloc::System;
    ///
    /// let vec: FallibleVec<i32, System> = FallibleVec::new_in(System);
    /// ```
    #[inline]
    #[unstable(feature = "allocator_api", issue = "32838")]
    #[rustc_const_unstable(feature = "allocator_api", issue = "32838")]
    pub const fn new_in(alloc: A) -> Self {
        InfallibleVec::new_in(alloc).to_fallible()
    }

    /// Constructs a new, empty `FallibleVec<T, A>` with at least the specified capacity
    /// with the provided allocator.
    ///
    /// The vector will be able to hold at least `capacity` elements without
    /// reallocating. This method is allowed to allocate for more elements than
    /// `capacity`. If `capacity` is zero, the vector will not allocate.
    ///
    /// It is important to note that although the returned vector has the
    /// minimum *capacity* specified, the vector will have a zero *length*. For
    /// an explanation of the difference between length and capacity, see
    /// *[Capacity and reallocation]*.
    ///
    /// If it is important to know the exact allocated capacity of a `FallibleVec`,
    /// always use the [`capacity`] method after construction.
    ///
    /// For `FallibleVec<T, A>` where `T` is a zero-sized type, there will be no allocation
    /// and the capacity will always be `usize::MAX`.
    ///
    /// [Capacity and reallocation]: #capacity-and-reallocation
    /// [`capacity`]: FallibleVec::capacity
    ///
    /// # Panics
    ///
    /// Panics if the new capacity exceeds `isize::MAX` _bytes_.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(fallible_vec)]
    /// #![feature(allocator_api)]
    ///
    /// use std::collections::fallible::FallibleVec;
    /// use std::alloc::System;
    ///
    /// let mut vec = FallibleVec::with_capacity_in(10, System)?;
    ///
    /// // The vector contains no items, even though it has capacity for more
    /// assert_eq!(vec.len(), 0);
    /// assert!(vec.capacity() >= 10);
    ///
    /// // These are all done without reallocating...
    /// for i in 0..10 {
    ///     vec.push(i)?;
    /// }
    /// assert_eq!(vec.len(), 10);
    /// assert!(vec.capacity() >= 10);
    ///
    /// // ...but this may make the vector reallocate
    /// vec.push(11)?;
    /// assert_eq!(vec.len(), 11);
    /// assert!(vec.capacity() >= 11);
    ///
    /// // A vector of a zero-sized type will always over-allocate, since no
    /// // allocation is necessary
    /// let vec_units = FallibleVec::<(), System>::with_capacity_in(10, System)?;
    /// assert_eq!(vec_units.capacity(), usize::MAX);
    /// # Ok::<(), std::collections::fallible::AllocError>(())
    /// ```
    #[inline]
    #[unstable(feature = "allocator_api", issue = "32838")]
    pub fn with_capacity_in(capacity: usize, alloc: A) -> Result<Self, AllocError> {
        InfallibleVec::try_with_capacity_in(capacity, alloc).map(|v| v.to_fallible())
    }

    /// Creates a `FallibleVec<T, A>` directly from a pointer, a length, a capacity,
    /// and an allocator.
    ///
    /// # Safety
    ///
    /// This is highly unsafe, due to the number of invariants that aren't
    /// checked:
    ///
    /// * `ptr` must be [*currently allocated*] via the given allocator `alloc`.
    /// * `T` needs to have the same alignment as what `ptr` was allocated with.
    ///   (`T` having a less strict alignment is not sufficient, the alignment really
    ///   needs to be equal to satisfy the [`dealloc`] requirement that memory must be
    ///   allocated and deallocated with the same layout.)
    /// * The size of `T` times the `capacity` (i.e. the allocated size in bytes) needs
    ///   to be the same size as the pointer was allocated with. (Because similar to
    ///   alignment, [`dealloc`] must be called with the same layout `size`.)
    /// * `length` needs to be less than or equal to `capacity`.
    /// * The first `length` values must be properly initialized values of type `T`.
    /// * `capacity` needs to [*fit*] the layout size that the pointer was allocated with.
    /// * The allocated size in bytes must be no larger than `isize::MAX`.
    ///   See the safety documentation of [`pointer::offset`].
    ///
    /// These requirements are always upheld by any `ptr` that has been allocated
    /// via `FallibleVec<T, A>`. Other allocation sources are allowed if the invariants are
    /// upheld.
    ///
    /// Violating these may cause problems like corrupting the allocator's
    /// internal data structures. For example it is **not** safe
    /// to build a `FallibleVec<u8>` from a pointer to a C `char` array with length `size_t`.
    /// It's also not safe to build one from a `FallibleVec<u16>` and its length, because
    /// the allocator cares about the alignment, and these two types have different
    /// alignments. The buffer was allocated with alignment 2 (for `u16`), but after
    /// turning it into a `FallibleVec<u8>` it'll be deallocated with alignment 1.
    ///
    /// The ownership of `ptr` is effectively transferred to the
    /// `FallibleVec<T>` which may then deallocate, reallocate or change the
    /// contents of memory pointed to by the pointer at will. Ensure
    /// that nothing else uses the pointer after calling this
    /// function.
    ///
    /// [`String`]: crate::string::String
    /// [`dealloc`]: crate::alloc::GlobalAlloc::dealloc
    /// [*currently allocated*]: crate::alloc::Allocator#currently-allocated-memory
    /// [*fit*]: crate::alloc::Allocator#memory-fitting
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(fallible_vec)]
    /// #![feature(allocator_api)]
    ///
    /// use std::collections::fallible::FallibleVec;
    /// use std::alloc::System;
    ///
    /// use std::ptr;
    ///
    /// let mut v = FallibleVec::with_capacity_in(3, System)?;
    /// v.push(1)?;
    /// v.push(2)?;
    /// v.push(3)?;
    ///
    /// // Deconstruct the vector into parts.
    /// let (p, len, cap, alloc) = v.into_raw_parts_with_alloc();
    ///
    /// unsafe {
    ///     // Overwrite memory with 4, 5, 6
    ///     for i in 0..len {
    ///         ptr::write(p.add(i), 4 + i);
    ///     }
    ///
    ///     // Put everything back together into a FallibleVec
    ///     let rebuilt = FallibleVec::from_raw_parts_in(p, len, cap, alloc.clone());
    ///     assert_eq!(rebuilt, [4, 5, 6]);
    /// }
    /// # Ok::<(), std::collections::fallible::AllocError>(())
    /// ```
    ///
    /// Using memory that was allocated elsewhere:
    ///
    /// ```rust
    /// #![feature(fallible_vec)]
    /// #![feature(allocator_api)]
    ///
    /// use std::collections::fallible::FallibleVec;
    /// use std::alloc::{AllocError, Allocator, Global, Layout};
    ///
    /// fn main() {
    ///     let layout = Layout::array::<u32>(16).expect("overflow cannot happen");
    ///
    ///     let vec = unsafe {
    ///         let mem = match Global.allocate(layout) {
    ///             Ok(mem) => mem.cast::<u32>().as_ptr(),
    ///             Err(AllocError) => return,
    ///         };
    ///
    ///         mem.write(1_000_000);
    ///
    ///         FallibleVec::from_raw_parts_in(mem, 1, 16, Global)
    ///     };
    ///
    ///     assert_eq!(vec, &[1_000_000]);
    ///     assert_eq!(vec.capacity(), 16);
    /// }
    /// ```
    #[inline]
    #[unstable(feature = "allocator_api", issue = "32838")]
    #[rustc_const_unstable(feature = "allocator_api", issue = "32838")]
    pub const unsafe fn from_raw_parts_in(
        ptr: *mut T,
        length: usize,
        capacity: usize,
        alloc: A,
    ) -> Self {
        unsafe { Self { buf: InfallibleVec::from_raw_parts_in(ptr, length, capacity, alloc) } }
    }

    /// Decomposes a `FallibleVec<T>` into its raw components: `(pointer, length, capacity, allocator)`.
    ///
    /// Returns the raw pointer to the underlying data, the length of the vector (in elements),
    /// the allocated capacity of the data (in elements), and the allocator. These are the same
    /// arguments in the same order as the arguments to [`from_raw_parts_in`].
    ///
    /// After calling this function, the caller is responsible for the
    /// memory previously managed by the `FallibleVec`. The only way to do
    /// this is to convert the raw pointer, length, and capacity back
    /// into a `FallibleVec` with the [`from_raw_parts_in`] function, allowing
    /// the destructor to perform the cleanup.
    ///
    /// [`from_raw_parts_in`]: FallibleVec::from_raw_parts_in
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(fallible_vec)]
    /// #![feature(allocator_api)]
    ///
    /// use std::collections::fallible::FallibleVec;
    /// use std::alloc::System;
    ///
    /// let mut v: FallibleVec<i32, System> = FallibleVec::new_in(System);
    /// v.push(-1)?;
    /// v.push(0)?;
    /// v.push(1)?;
    ///
    /// let (ptr, len, cap, alloc) = v.into_raw_parts_with_alloc();
    ///
    /// let rebuilt = unsafe {
    ///     // We can now make changes to the components, such as
    ///     // transmuting the raw pointer to a compatible type.
    ///     let ptr = ptr as *mut u32;
    ///
    ///     FallibleVec::from_raw_parts_in(ptr, len, cap, alloc)
    /// };
    /// assert_eq!(rebuilt, [4294967295, 0, 1]);
    /// # Ok::<(), std::collections::fallible::AllocError>(())
    /// ```
    #[must_use = "losing the pointer will leak memory"]
    #[unstable(feature = "allocator_api", issue = "32838")]
    #[rustc_const_unstable(feature = "allocator_api", issue = "32838")]
    pub const fn into_raw_parts_with_alloc(self) -> (*mut T, usize, usize, A) {
        self.buf.into_raw_parts_with_alloc()
    }

    /// Returns the total number of elements the vector can hold without
    /// reallocating.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(fallible_vec)]
    /// use std::collections::fallible::FallibleVec;
    ///
    /// let mut vec: FallibleVec<i32> = FallibleVec::with_capacity(10)?;
    /// vec.push(42)?;
    /// assert!(vec.capacity() >= 10);
    /// # Ok::<(), std::collections::fallible::AllocError>(())
    /// ```
    ///
    /// A vector with zero-sized elements will always have a capacity of usize::MAX:
    ///
    /// ```
    /// #![feature(fallible_vec)]
    /// #[derive(Clone)]
    /// struct ZeroSized;
    ///
    /// fn main() {
    ///     assert_eq!(std::mem::size_of::<ZeroSized>(), 0);
    ///     let v = vec![ZeroSized; 0].to_fallible();
    ///     assert_eq!(v.capacity(), usize::MAX);
    /// }
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    #[rustc_const_stable(feature = "const_vec_string_slice", since = "1.87.0")]
    pub const fn capacity(&self) -> usize {
        self.buf.capacity()
    }

    /// Tries to reserve capacity for at least `additional` more elements to be inserted
    /// in the given `FallibleVec<T>`. The collection may reserve more space to speculatively avoid
    /// frequent reallocations. After calling `reserve`, capacity will be
    /// greater than or equal to `self.len() + additional` if it returns
    /// `Ok(())`. Does nothing if capacity is already sufficient. This method
    /// preserves the contents even if an error occurs.
    ///
    /// # Errors
    ///
    /// If the capacity overflows, or the allocator reports a failure, then an error
    /// is returned.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(fallible_vec)]
    /// use std::collections::fallible::{FallibleVec, AllocError};
    ///
    /// fn process_data(data: &[u32]) -> Result<FallibleVec<u32>, AllocError> {
    ///     let mut output = FallibleVec::new();
    ///
    ///     // Pre-reserve the memory, exiting if we can't
    ///     output.reserve(data.len())?;
    ///
    ///     // Now we know this can't OOM in the middle of our complex work
    ///     for val in data.iter() {
    ///         output.push(val * 2 + 5).unwrap()
    ///     }
    ///
    ///     Ok(output)
    /// }
    /// # process_data(&[1, 2, 3]).expect("why is the test harness OOMing on 12 bytes?");
    /// ```
    pub fn reserve(&mut self, additional: usize) -> Result<(), AllocError> {
        self.buf.try_reserve(additional)
    }

    /// Tries to reserve the minimum capacity for at least `additional`
    /// elements to be inserted in the given `FallibleVec<T>`. Unlike [`reserve`],
    /// this will not deliberately over-allocate to speculatively avoid frequent
    /// allocations. After calling `reserve_exact`, capacity will be greater
    /// than or equal to `self.len() + additional` if it returns `Ok(())`.
    /// Does nothing if the capacity is already sufficient.
    ///
    /// Note that the allocator may give the collection more space than it
    /// requests. Therefore, capacity can not be relied upon to be precisely
    /// minimal. Prefer [`reserve`] if future insertions are expected.
    ///
    /// [`reserve`]: FallibleVec::reserve
    ///
    /// # Errors
    ///
    /// If the capacity overflows, or the allocator reports a failure, then an error
    /// is returned.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(fallible_vec)]
    /// use std::collections::fallible::{FallibleVec, AllocError};
    ///
    /// fn process_data(data: &[u32]) -> Result<FallibleVec<u32>, AllocError> {
    ///     let mut output = FallibleVec::new();
    ///
    ///     // Pre-reserve the memory, exiting if we can't
    ///     output.reserve_exact(data.len())?;
    ///
    ///     // Now we know this can't OOM in the middle of our complex work
    ///     for val in data.iter() {
    ///         output.push(val * 2 + 5).unwrap();
    ///     }
    ///
    ///     Ok(output)
    /// }
    /// # process_data(&[1, 2, 3]).expect("why is the test harness OOMing on 12 bytes?");
    /// ```
    pub fn reserve_exact(&mut self, additional: usize) -> Result<(), AllocError> {
        self.buf.try_reserve_exact(additional)
    }

    /// Tries to shrink the capacity of the vector as much as possible
    ///
    /// The behavior of this method depends on the allocator, which may either shrink the vector
    /// in-place or reallocate. The resulting vector might still have some excess capacity, just as
    /// is the case for [`with_capacity`]. See [`Allocator::shrink`] for more details.
    ///
    /// [`with_capacity`]: FallibleVec::with_capacity
    ///
    /// # Errors
    ///
    /// This function returns an error if the allocator fails to shrink the allocation,
    /// the vector thereafter is still safe to use, the capacity remains unchanged
    /// however. See [`Allocator::shrink`].
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(fallible_vec)]
    /// use std::collections::fallible::FallibleVec;
    ///
    /// let mut vec = FallibleVec::with_capacity(10)?;
    /// vec.push(1)?;
    /// vec.push(2)?;
    /// vec.push(3)?;
    /// assert!(vec.capacity() >= 10);
    /// vec.shrink_to_fit().expect("why is the test harness failing to shrink to 12 bytes");
    /// assert!(vec.capacity() >= 3);
    /// # Ok::<(), std::collections::fallible::AllocError>(())
    /// ```
    #[inline]
    pub fn shrink_to_fit(&mut self) -> Result<(), AllocError> {
        self.buf.try_shrink_to_fit()
    }

    /// Shrinks the capacity of the vector with a lower bound.
    ///
    /// The capacity will remain at least as large as both the length
    /// and the supplied value.
    ///
    /// If the current capacity is less than the lower limit, this is a no-op.
    ///
    /// # Errors
    ///
    /// This function returns an error if the allocator fails to shrink the allocation,
    /// the vector thereafter is still safe to use, the capacity remains unchanged
    /// however. See [`Allocator::shrink`].
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(fallible_vec)]
    ///
    /// use std::collections::fallible::FallibleVec;
    /// let mut vec = FallibleVec::with_capacity(10)?;
    /// vec.push(1)?;
    /// vec.push(2)?;
    /// vec.push(3)?;
    /// assert!(vec.capacity() >= 10);
    /// vec.shrink_to(4).expect("why is the test harness failing to shrink to 12 bytes");
    /// assert!(vec.capacity() >= 4);
    /// vec.shrink_to(0).expect("this is a no-op and thus the allocator isn't involved.");
    /// assert!(vec.capacity() >= 3);
    /// # Ok::<(), std::collections::fallible::AllocError>(())
    /// ```
    #[inline]
    pub fn shrink_to(&mut self, min_capacity: usize) -> Result<(), AllocError> {
        self.buf.try_shrink_to(min_capacity)
    }

    /// Shortens the vector, keeping the first `len` elements and dropping
    /// the rest.
    ///
    /// If `len` is greater or equal to the vector's current length, this has
    /// no effect.
    ///
    /// Note that this method has no effect on the allocated capacity
    /// of the vector.
    ///
    /// # Examples
    ///
    /// Truncating a five element vector to two elements:
    ///
    /// ```
    /// #![feature(fallible_vec)]
    /// let mut vec = vec![1, 2, 3, 4, 5].to_fallible();
    /// vec.truncate(2);
    /// assert_eq!(vec, [1, 2]);
    /// ```
    ///
    /// No truncation occurs when `len` is greater than the vector's current
    /// length:
    ///
    /// ```
    /// #![feature(fallible_vec)]
    /// let mut vec = vec![1, 2, 3].to_fallible();
    /// vec.truncate(8);
    /// assert_eq!(vec, [1, 2, 3]);
    /// ```
    ///
    /// Truncating when `len == 0` is equivalent to calling the [`clear`]
    /// method.
    ///
    /// ```
    /// #![feature(fallible_vec)]
    /// let mut vec = vec![1, 2, 3].to_fallible();
    /// vec.truncate(0);
    /// assert_eq!(vec, []);
    /// ```
    ///
    /// [`clear`]: FallibleVec::clear
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn truncate(&mut self, len: usize) {
        self.buf.truncate(len);
    }

    /// Extracts a slice containing the entire vector.
    ///
    /// Equivalent to `&s[..]`.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(fallible_vec)]
    /// use std::io::{self, Write};
    /// let buffer = vec![1, 2, 3, 5, 8].to_fallible();
    /// io::sink().write(buffer.as_slice()).unwrap();
    /// ```
    #[inline]
    #[rustc_const_unstable(feature = "fallible_vec", issue = "157392")]
    pub const fn as_slice(&self) -> &[T] {
        self.buf.as_slice()
    }

    /// Extracts a mutable slice of the entire vector.
    ///
    /// Equivalent to `&mut s[..]`.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(fallible_vec)]
    /// use std::io::{self, Read};
    /// let mut buffer = vec![0; 3].to_fallible();
    /// io::repeat(0b101).read_exact(buffer.as_mut_slice()).unwrap();
    /// ```
    #[inline]
    #[rustc_const_unstable(feature = "fallible_vec", issue = "157392")]
    pub const fn as_mut_slice(&mut self) -> &mut [T] {
        self.buf.as_mut_slice()
    }

    /// Returns a raw pointer to the vector's buffer, or a dangling raw pointer
    /// valid for zero sized reads if the vector didn't allocate.
    ///
    /// The caller must ensure that the vector outlives the pointer this
    /// function returns, or else it will end up dangling.
    /// Modifying the vector may cause its buffer to be reallocated,
    /// which would also make any pointers to it invalid.
    ///
    /// The caller must also ensure that the memory the pointer (non-transitively) points to
    /// is never written to (except inside an `UnsafeCell`) using this pointer or any pointer
    /// derived from it. If you need to mutate the contents of the slice, use [`as_mut_ptr`].
    ///
    /// This method guarantees that for the purpose of the aliasing model, this method
    /// does not materialize a reference to the underlying slice, and thus the returned pointer
    /// will remain valid when mixed with other calls to [`as_ptr`] and [`as_mut_ptr`].
    /// Note that calling other methods that materialize mutable references to the slice,
    /// or mutable references to specific elements you are planning on accessing through this pointer,
    /// as well as writing to those elements, may still invalidate this pointer.
    /// See the second example below for how this guarantee can be used.
    ///
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(fallible_vec)]
    /// let x = vec![1, 2, 4].to_fallible();
    /// let x_ptr = x.as_ptr();
    ///
    /// unsafe {
    ///     for i in 0..x.len() {
    ///         assert_eq!(*x_ptr.add(i), 1 << i);
    ///     }
    /// }
    /// ```
    ///
    /// Due to the aliasing guarantee, the following code is legal:
    ///
    /// ```rust
    /// #![feature(fallible_vec)]
    /// unsafe {
    ///     let mut v = vec![0, 1, 2].to_fallible();
    ///     let ptr1 = v.as_ptr();
    ///     let _ = ptr1.read();
    ///     let ptr2 = v.as_mut_ptr().offset(2);
    ///     ptr2.write(2);
    ///     // Notably, the write to `ptr2` did *not* invalidate `ptr1`
    ///     // because it mutated a different element:
    ///     let _ = ptr1.read();
    /// }
    /// ```
    ///
    /// [`as_mut_ptr`]: FallibleVec::as_mut_ptr
    /// [`as_ptr`]: FallibleVec::as_ptr
    #[rustc_const_unstable(feature = "fallible_vec", issue = "157392")]
    #[rustc_never_returns_null_ptr]
    #[rustc_as_ptr]
    #[inline]
    pub const fn as_ptr(&self) -> *const T {
        self.buf.as_ptr()
    }

    /// Returns a raw mutable pointer to the vector's buffer, or a dangling
    /// raw pointer valid for zero sized reads if the vector didn't allocate.
    ///
    /// The caller must ensure that the vector outlives the pointer this
    /// function returns, or else it will end up dangling.
    /// Modifying the vector may cause its buffer to be reallocated,
    /// which would also make any pointers to it invalid.
    ///
    /// This method guarantees that for the purpose of the aliasing model, this method
    /// does not materialize a reference to the underlying slice, and thus the returned pointer
    /// will remain valid when mixed with other calls to [`as_ptr`] and [`as_mut_ptr`].
    /// Note that calling other methods that materialize references to the slice,
    /// or references to specific elements you are planning on accessing through this pointer,
    /// may still invalidate this pointer.
    /// See the second example below for how this guarantee can be used.
    ///
    /// The method also guarantees that, as long as `T` is not zero-sized and the capacity is
    /// nonzero, the pointer may be passed into [`dealloc`] with a layout of
    /// `Layout::array::<T>(capacity)` in order to deallocate the backing memory. If this is done,
    /// be careful not to run the destructor of the `FallibleVec`, as dropping it will result in
    /// double-frees. Wrapping the `FallibleVec` in a [`ManuallyDrop`] is the typical way to achieve this.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(fallible_vec)]
    /// use std::collections::fallible::FallibleVec;
    /// // Allocate vector big enough for 4 elements.
    /// let size = 4;
    /// let mut x: FallibleVec<i32> = FallibleVec::with_capacity(size)?;
    /// let x_ptr = x.as_mut_ptr();
    ///
    /// // Initialize elements via raw pointer writes, then set length.
    /// unsafe {
    ///     for i in 0..size {
    ///         *x_ptr.add(i) = i as i32;
    ///     }
    ///     x.set_len(size);
    /// }
    /// assert_eq!(&*x, &[0, 1, 2, 3]);
    /// # Ok::<(), std::collections::fallible::AllocError>(())
    /// ```
    ///
    /// Due to the aliasing guarantee, the following code is legal:
    ///
    /// ```rust
    /// #![feature(fallible_vec)]
    /// unsafe {
    ///     let mut v = vec![0].to_fallible();
    ///     let ptr1 = v.as_mut_ptr();
    ///     ptr1.write(1);
    ///     let ptr2 = v.as_mut_ptr();
    ///     ptr2.write(2);
    ///     // Notably, the write to `ptr2` did *not* invalidate `ptr1`:
    ///     ptr1.write(3);
    /// }
    /// ```
    ///
    /// Deallocating a vector using [`Box`] (which uses [`dealloc`] internally):
    ///
    /// ```
    /// #![feature(fallible_vec)]
    /// use std::mem::{ManuallyDrop, MaybeUninit};
    ///
    /// let mut v = ManuallyDrop::new(vec![0, 1, 2].to_fallible());
    /// let ptr = v.as_mut_ptr();
    /// let capacity = v.capacity();
    /// let slice_ptr: *mut [MaybeUninit<i32>] =
    ///     std::ptr::slice_from_raw_parts_mut(ptr.cast(), capacity);
    /// drop(unsafe { Box::from_raw(slice_ptr) });
    /// ```
    ///
    /// [`Box`]: crate::boxed::Box
    /// [`as_mut_ptr`]: FallibleVec::as_mut_ptr
    /// [`as_ptr`]: FallibleVec::as_ptr
    /// [`dealloc`]: crate::alloc::GlobalAlloc::dealloc
    /// [`ManuallyDrop`]: core::mem::ManuallyDrop
    #[rustc_const_unstable(feature = "fallible_vec", issue = "157392")]
    #[rustc_never_returns_null_ptr]
    #[rustc_as_ptr]
    #[inline]
    pub const fn as_mut_ptr(&mut self) -> *mut T {
        self.buf.as_mut_ptr()
    }

    /// Forces the length of the vector to `new_len`.
    ///
    /// This is a low-level operation that maintains none of the normal
    /// invariants of the type. Normally changing the length of a vector
    /// is done using one of the safe operations instead, such as
    /// [`truncate`] or [`clear`].
    ///
    /// [`truncate`]: FallibleVec::truncate
    /// [`clear`]: FallibleVec::clear
    ///
    /// # Safety
    ///
    /// - `new_len` must be less than or equal to [`capacity()`].
    /// - The elements at `old_len..new_len` must be initialized.
    ///
    /// [`capacity()`]: FallibleVec::capacity
    ///
    /// # Examples
    ///
    /// See [`spare_capacity_mut()`] for an example with safe
    /// initialization of capacity elements and use of this method.
    ///
    /// `set_len()` can be useful for situations in which the vector
    /// is serving as a buffer for other code, particularly over FFI:
    ///
    /// ```no_run
    /// #![feature(fallible_vec)]
    /// # #![allow(dead_code)]
    /// use std::collections::fallible::FallibleVec;
    /// # // This is just a minimal skeleton for the doc example;
    /// # // don't use this as a starting point for a real library.
    /// # pub struct StreamWrapper { strm: *mut std::ffi::c_void }
    /// # const Z_OK: i32 = 0;
    /// # unsafe extern "C" {
    /// #     fn deflateGetDictionary(
    /// #         strm: *mut std::ffi::c_void,
    /// #         dictionary: *mut u8,
    /// #         dictLength: *mut usize,
    /// #     ) -> i32;
    /// # }
    /// # impl StreamWrapper {
    /// pub fn get_dictionary(&self) -> Option<FallibleVec<u8>> {
    ///     // Per the FFI method's docs, "32768 bytes is always enough".
    ///     let mut dict = FallibleVec::with_capacity(32_768).unwrap();
    ///     let mut dict_length = 0;
    ///     // SAFETY: When `deflateGetDictionary` returns `Z_OK`, it holds that:
    ///     // 1. `dict_length` elements were initialized.
    ///     // 2. `dict_length` <= the capacity (32_768)
    ///     // which makes `set_len` safe to call.
    ///     unsafe {
    ///         // Make the FFI call...
    ///         let r = deflateGetDictionary(self.strm, dict.as_mut_ptr(), &mut dict_length);
    ///         if r == Z_OK {
    ///             // ...and update the length to what was initialized.
    ///             dict.set_len(dict_length);
    ///             Some(dict)
    ///         } else {
    ///             None
    ///         }
    ///     }
    /// }
    /// # }
    /// ```
    ///
    /// While the following example is sound, there is a memory leak since
    /// the inner vectors were not freed prior to the `set_len` call:
    ///
    /// ```
    /// #![feature(fallible_vec)]
    /// let mut vec = vec![vec![1, 0, 0].to_fallible(),
    ///                    vec![0, 1, 0].to_fallible(),
    ///                    vec![0, 0, 1].to_fallible()];
    /// // SAFETY:
    /// // 1. `old_len..0` is empty so no elements need to be initialized.
    /// // 2. `0 <= capacity` always holds whatever `capacity` is.
    /// unsafe {
    ///     vec.set_len(0);
    /// #   // FIXME(https://github.com/rust-lang/miri/issues/3670):
    /// #   // use -Zmiri-disable-leak-check instead of unleaking in tests meant to leak.
    /// #   vec.set_len(3);
    /// }
    /// ```
    ///
    /// Normally, here, one would use [`clear`] instead to correctly drop
    /// the contents and thus not leak memory.
    ///
    /// [`spare_capacity_mut()`]: FallibleVec::spare_capacity_mut
    #[inline]
    pub unsafe fn set_len(&mut self, new_len: usize) {
        unsafe { self.buf.set_len(new_len) }
    }

    /// Appends an element to the back of a collection.
    ///
    /// # Panics
    ///
    /// Panics if the new capacity exceeds `isize::MAX` _bytes_.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(fallible_vec)]
    /// let mut vec = vec![1, 2].to_fallible();
    /// vec.push(3)?;
    /// assert_eq!(vec, [1, 2, 3]);
    /// # Ok::<(), std::collections::fallible::AllocError>(())
    /// ```
    ///
    /// # Time complexity
    ///
    /// Takes amortized *O*(1) time. If the vector's length would exceed its
    /// capacity after the push, *O*(*capacity*) time is taken to copy the
    /// vector's elements to a larger allocation. This expensive operation is
    /// offset by the *capacity* *O*(1) insertions it allows.
    #[inline]
    pub fn push(&mut self, value: T) -> Result<(), AllocError> {
        self.push_mut(value).map(|_| ())
    }

    /// Appends an element to the back of a collection, returning a reference to it.
    ///
    /// # Errors
    ///
    /// Error if an allocation fails or if new capacity exceeds `isize::MAX` _bytes_.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(fallible_vec)]
    ///
    /// let mut vec = vec![1, 2].to_fallible();
    /// let last = vec.push_mut(3)?;
    /// assert_eq!(*last, 3);
    /// assert_eq!(vec, [1, 2, 3]);
    ///
    /// let last = vec.push_mut(3)?;
    /// *last += 1;
    /// assert_eq!(vec, [1, 2, 3, 4]);
    /// # Ok::<(), alloc::collections::fallible::AllocError>(())
    /// ```
    ///
    /// # Time complexity
    ///
    /// Takes amortized *O*(1) time. If the vector's length would exceed its
    /// capacity after the push, *O*(*capacity*) time is taken to copy the
    /// vector's elements to a larger allocation. This expensive operation is
    /// offset by the *capacity* *O*(1) insertions it allows.
    #[inline]
    #[must_use = "if you don't need a reference to the value, use `Vec::push` instead"]
    pub fn push_mut(&mut self, value: T) -> Result<&mut T, AllocError> {
        self.buf.try_push_mut(value)
    }

    /// Removes the last element from a vector and returns it, or [`None`] if it
    /// is empty.
    ///
    /// If you'd like to pop the first element, consider using
    /// [`VecDeque::pop_front`] instead.
    ///
    /// [`VecDeque::pop_front`]: crate::collections::VecDeque::pop_front
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(fallible_vec)]
    /// let mut vec = vec![1, 2, 3].to_fallible();
    /// assert_eq!(vec.pop(), Some(3));
    /// assert_eq!(vec, [1, 2]);
    /// ```
    ///
    /// # Time complexity
    ///
    /// Takes *O*(1) time.
    #[inline]
    pub fn pop(&mut self) -> Option<T> {
        self.buf.pop()
    }

    /// Removes and returns the last element from a vector if the predicate
    /// returns `true`, or [`None`] if the predicate returns false or the vector
    /// is empty (the predicate will not be called in that case).
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(fallible_vec)]
    ///
    /// let mut vec = vec![1, 2, 3, 4].to_fallible();
    /// let pred = |x: &mut i32| *x % 2 == 0;
    ///
    /// assert_eq!(vec.pop_if(pred), Some(4));
    /// assert_eq!(vec, [1, 2, 3]);
    /// assert_eq!(vec.pop_if(pred), None);
    /// ```
    pub fn pop_if(&mut self, predicate: impl FnOnce(&mut T) -> bool) -> Option<T> {
        self.buf.pop_if(predicate)
    }

    /// Clears the vector, removing all values.
    ///
    /// Note that this method has no effect on the allocated capacity
    /// of the vector.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(fallible_vec)]
    /// let mut v = vec![1, 2, 3].to_fallible();
    ///
    /// v.clear();
    ///
    /// assert!(v.is_empty());
    /// ```
    #[inline]
    pub fn clear(&mut self) {
        self.buf.clear()
    }

    /// Returns the number of elements in the vector, also referred to
    /// as its 'length'.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(fallible_vec)]
    /// let a = vec![1, 2, 3].to_fallible();
    /// assert_eq!(a.len(), 3);
    /// ```
    #[inline]
    #[rustc_const_unstable(feature = "fallible_vec", issue = "157392")]
    pub const fn len(&self) -> usize {
        self.buf.len()
    }

    /// Returns `true` if the vector contains no elements.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(fallible_vec)]
    /// let mut v = Vec::new().to_fallible();
    /// assert!(v.is_empty());
    ///
    /// v.push(1)?;
    /// assert!(!v.is_empty());
    /// # Ok::<(), std::collections::fallible::AllocError>(())
    /// ```
    #[rustc_const_unstable(feature = "fallible_vec", issue = "157392")]
    pub const fn is_empty(&self) -> bool {
        self.buf.is_empty()
    }

    /// Consumes and leaks the `FallibleVec`, returning a mutable reference to the contents,
    /// `&'a mut [T]`.
    ///
    /// Note that the type `T` must outlive the chosen lifetime `'a`. If the type
    /// has only static references, or none at all, then this may be chosen to be
    /// `'static`.
    ///
    /// As of Rust 1.57, this method does not reallocate or shrink the `FallibleVec`,
    /// so the leaked allocation may include unused capacity that is not part
    /// of the returned slice.
    ///
    /// This function is mainly useful for data that lives for the remainder of
    /// the program's life. Dropping the returned reference will cause a memory
    /// leak.
    ///
    /// # Examples
    ///
    /// Simple usage:
    ///
    /// ```
    /// #![feature(fallible_vec)]
    /// let x = vec![1, 2, 3].to_fallible();
    /// let static_ref: &'static mut [usize] = x.leak();
    /// static_ref[0] += 1;
    /// assert_eq!(static_ref, &[2, 2, 3]);
    /// # // FIXME(https://github.com/rust-lang/miri/issues/3670):
    /// # // use -Zmiri-disable-leak-check instead of unleaking in tests meant to leak.
    /// # drop(unsafe { Box::from_raw(static_ref) });
    /// ```
    #[inline]
    pub fn leak<'a>(self) -> &'a mut [T]
    where
        A: 'a,
    {
        self.buf.leak()
    }

    /// Returns the remaining spare capacity of the vector as a slice of
    /// `MaybeUninit<T>`.
    ///
    /// The returned slice can be used to fill the vector with data (e.g. by
    /// reading from a file) before marking the data as initialized using the
    /// [`set_len`] method.
    ///
    /// [`set_len`]: FallibleVec::set_len
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(fallible_vec)]
    /// use std::collections::fallible::FallibleVec;
    /// // Allocate vector big enough for 10 elements.
    /// let mut v = FallibleVec::with_capacity(10)?;
    ///
    /// // Fill in the first 3 elements.
    /// let uninit = v.spare_capacity_mut();
    /// uninit[0].write(0);
    /// uninit[1].write(1);
    /// uninit[2].write(2);
    ///
    /// // Mark the first 3 elements of the vector as being initialized.
    /// unsafe {
    ///     v.set_len(3);
    /// }
    ///
    /// assert_eq!(&v, &[0, 1, 2]);
    /// # Ok::<(), std::collections::fallible::AllocError>(())
    /// ```
    #[inline]
    pub fn spare_capacity_mut(&mut self) -> &mut [MaybeUninit<T>] {
        self.buf.spare_capacity_mut()
    }
}

#[unstable(feature = "fallible_vec", issue = "157392")]
impl<T, A: Allocator> ops::Deref for FallibleVec<T, A> {
    type Target = [T];

    #[inline]
    fn deref(&self) -> &[T] {
        self.as_slice()
    }
}

#[unstable(feature = "fallible_vec", issue = "157392")]
impl<T, A: Allocator> ops::DerefMut for FallibleVec<T, A> {
    #[inline]
    fn deref_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

#[unstable(feature = "fallible_vec", issue = "157392")]
impl<T, I: SliceIndex<[T]>, A: Allocator> Index<I> for FallibleVec<T, A> {
    type Output = I::Output;

    #[inline]
    fn index(&self, index: I) -> &Self::Output {
        Index::index(&**self, index)
    }
}

#[unstable(feature = "fallible_vec", issue = "157392")]
impl<T, I: SliceIndex<[T]>, A: Allocator> IndexMut<I> for FallibleVec<T, A> {
    #[inline]
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        IndexMut::index_mut(&mut **self, index)
    }
}

#[unstable(feature = "fallible_vec", issue = "157392")]
impl<T, A: Allocator> IntoIterator for FallibleVec<T, A> {
    type Item = T;
    type IntoIter = IntoIter<T, A>;

    /// Creates a consuming iterator, that is, one that moves each value out of
    /// the vector (from start to end). The vector cannot be used after calling
    /// this.
    ///
    /// # Examples
    ///
    /// ```
    /// let v = vec!["a".to_string(), "b".to_string()];
    /// let mut v_iter = v.into_iter();
    ///
    /// let first_element: Option<String> = v_iter.next();
    ///
    /// assert_eq!(first_element, Some("a".to_string()));
    /// assert_eq!(v_iter.next(), Some("b".to_string()));
    /// assert_eq!(v_iter.next(), None);
    /// ```
    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.buf.into_iter()
    }
}

#[unstable(feature = "fallible_vec", issue = "157392")]
impl<'a, T, A: Allocator> IntoIterator for &'a FallibleVec<T, A> {
    type Item = &'a T;
    type IntoIter = slice::Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

#[unstable(feature = "fallible_vec", issue = "157392")]
impl<'a, T, A: Allocator> IntoIterator for &'a mut FallibleVec<T, A> {
    type Item = &'a mut T;
    type IntoIter = slice::IterMut<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

#[unstable(feature = "fallible_vec", issue = "157392")]
impl<T> Default for FallibleVec<T> {
    /// Creates an empty `FallibleVec<T>`.
    ///
    /// The vector will not allocate until elements are pushed onto it.
    fn default() -> FallibleVec<T> {
        FallibleVec::new()
    }
}

impl<T, A: Allocator> AsRef<FallibleVec<T, A>> for FallibleVec<T, A> {
    fn as_ref(&self) -> &FallibleVec<T, A> {
        self
    }
}

#[unstable(feature = "fallible_vec", issue = "157392")]
impl<T, A: Allocator> AsMut<FallibleVec<T, A>> for FallibleVec<T, A> {
    fn as_mut(&mut self) -> &mut FallibleVec<T, A> {
        self
    }
}

#[unstable(feature = "fallible_vec", issue = "157392")]
impl<T, A: Allocator> AsRef<[T]> for FallibleVec<T, A> {
    fn as_ref(&self) -> &[T] {
        self
    }
}

#[unstable(feature = "fallible_vec", issue = "157392")]
impl<T, A: Allocator> AsMut<[T]> for FallibleVec<T, A> {
    fn as_mut(&mut self) -> &mut [T] {
        self
    }
}
