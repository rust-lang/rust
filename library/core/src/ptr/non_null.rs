use crate::cmp::Ordering;
use crate::marker::{PointeeSized, Unsize};
use crate::mem::{MaybeUninit, SizedTypeProperties};
use crate::num::NonZero;
use crate::ops::{CoerceUnsized, DispatchFromDyn};
use crate::pin::PinCoerceUnsized;
use crate::ptr::Unique;
use crate::slice::{self, SliceIndex};
use crate::ub_checks::assert_unsafe_precondition;
use crate::{fmt, hash, intrinsics, mem, ptr};

/// `*mut T` but non-zero and [covariant].
///
/// This is often the correct thing to use when building data structures using
/// raw pointers, but is ultimately more dangerous to use because of its additional
/// properties. If you're not sure if you should use `NonNull<T>`, just use `*mut T`!
///
/// Unlike `*mut T`, the pointer must always be non-null, even if the pointer
/// is never dereferenced. This is so that enums may use this forbidden value
/// as a discriminant -- `Option<NonNull<T>>` has the same size as `*mut T`.
/// However the pointer may still dangle if it isn't dereferenced.
///
/// Unlike `*mut T`, `NonNull<T>` is covariant over `T`. This is usually the correct
/// choice for most data structures and safe abstractions, such as `Box`, `Rc`, `Arc`, `Vec`,
/// and `LinkedList`.
///
/// In rare cases, if your type exposes a way to mutate the value of `T` through a `NonNull<T>`,
/// and you need to prevent unsoundness from variance (for example, if `T` could be a reference
/// with a shorter lifetime), you should add a field to make your type invariant, such as
/// `PhantomData<Cell<T>>` or `PhantomData<&'a mut T>`.
///
/// Example of a type that must be invariant:
/// ```rust
/// use std::cell::Cell;
/// use std::marker::PhantomData;
/// struct Invariant<T> {
///     ptr: std::ptr::NonNull<T>,
///     _invariant: PhantomData<Cell<T>>,
/// }
/// ```
///
/// Notice that `NonNull<T>` has a `From` instance for `&T`. However, this does
/// not change the fact that mutating through a (pointer derived from a) shared
/// reference is undefined behavior unless the mutation happens inside an
/// [`UnsafeCell<T>`]. The same goes for creating a mutable reference from a shared
/// reference. When using this `From` instance without an `UnsafeCell<T>`,
/// it is your responsibility to ensure that `as_mut` is never called, and `as_ptr`
/// is never used for mutation.
///
/// # Representation
///
/// Thanks to the [null pointer optimization],
/// `NonNull<T>` and `Option<NonNull<T>>`
/// are guaranteed to have the same size and alignment:
///
/// ```
/// use std::ptr::NonNull;
///
/// assert_eq!(size_of::<NonNull<i16>>(), size_of::<Option<NonNull<i16>>>());
/// assert_eq!(align_of::<NonNull<i16>>(), align_of::<Option<NonNull<i16>>>());
///
/// assert_eq!(size_of::<NonNull<str>>(), size_of::<Option<NonNull<str>>>());
/// assert_eq!(align_of::<NonNull<str>>(), align_of::<Option<NonNull<str>>>());
/// ```
///
/// [covariant]: https://doc.rust-lang.org/reference/subtyping.html
/// [`PhantomData`]: crate::marker::PhantomData
/// [`UnsafeCell<T>`]: crate::cell::UnsafeCell
/// [null pointer optimization]: crate::option#representation
#[stable(feature = "nonnull", since = "1.25.0")]
#[repr(transparent)]
#[rustc_layout_scalar_valid_range_start(1)]
#[rustc_nonnull_optimization_guaranteed]
#[rustc_diagnostic_item = "NonNull"]
pub struct NonNull<T: PointeeSized> {
    // Remember to use `.as_ptr()` instead of `.pointer`, as field projecting to
    // this is banned by <https://github.com/rust-lang/compiler-team/issues/807>.
    pointer: *const T,
}

/// `NonNull` pointers are not `Send` because the data they reference may be aliased.
// N.B., this impl is unnecessary, but should provide better error messages.
#[stable(feature = "nonnull", since = "1.25.0")]
impl<T: PointeeSized> !Send for NonNull<T> {}

/// `NonNull` pointers are not `Sync` because the data they reference may be aliased.
// N.B., this impl is unnecessary, but should provide better error messages.
#[stable(feature = "nonnull", since = "1.25.0")]
impl<T: PointeeSized> !Sync for NonNull<T> {}

impl<T: Sized> NonNull<T> {
    /// Creates a pointer with the given address and no [provenance][crate::ptr#provenance].
    ///
    /// For more details, see the equivalent method on a raw pointer, [`ptr::without_provenance_mut`].
    ///
    /// This is a [Strict Provenance][crate::ptr#strict-provenance] API.
    #[stable(feature = "nonnull_provenance", since = "CURRENT_RUSTC_VERSION")]
    #[rustc_const_stable(feature = "nonnull_provenance", since = "CURRENT_RUSTC_VERSION")]
    #[must_use]
    #[inline]
    pub const fn without_provenance(addr: NonZero<usize>) -> Self {
        let pointer = crate::ptr::without_provenance(addr.get());
        // SAFETY: we know `addr` is non-zero.
        unsafe { NonNull { pointer } }
    }

    /// Creates a new `NonNull` that is dangling, but well-aligned.
    ///
    /// This is useful for initializing types which lazily allocate, like
    /// `Vec::new` does.
    ///
    /// Note that the pointer value may potentially represent a valid pointer to
    /// a `T`, which means this must not be used as a "not yet initialized"
    /// sentinel value. Types that lazily allocate must track initialization by
    /// some other means.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::ptr::NonNull;
    ///
    /// let ptr = NonNull::<u32>::dangling();
    /// // Important: don't try to access the value of `ptr` without
    /// // initializing it first! The pointer is not null but isn't valid either!
    /// ```
    #[stable(feature = "nonnull", since = "1.25.0")]
    #[rustc_const_stable(feature = "const_nonnull_dangling", since = "1.36.0")]
    #[must_use]
    #[inline]
    pub const fn dangling() -> Self {
        let align = crate::ptr::Alignment::of::<T>();
        NonNull::without_provenance(align.as_nonzero())
    }

    /// Converts an address back to a mutable pointer, picking up some previously 'exposed'
    /// [provenance][crate::ptr#provenance].
    ///
    /// For more details, see the equivalent method on a raw pointer, [`ptr::with_exposed_provenance_mut`].
    ///
    /// This is an [Exposed Provenance][crate::ptr#exposed-provenance] API.
    #[stable(feature = "nonnull_provenance", since = "CURRENT_RUSTC_VERSION")]
    #[inline]
    pub fn with_exposed_provenance(addr: NonZero<usize>) -> Self {
        // SAFETY: we know `addr` is non-zero.
        unsafe {
            let ptr = crate::ptr::with_exposed_provenance_mut(addr.get());
            NonNull::new_unchecked(ptr)
        }
    }

    /// Returns a shared references to the value. In contrast to [`as_ref`], this does not require
    /// that the value has to be initialized.
    ///
    /// For the mutable counterpart see [`as_uninit_mut`].
    ///
    /// [`as_ref`]: NonNull::as_ref
    /// [`as_uninit_mut`]: NonNull::as_uninit_mut
    ///
    /// # Safety
    ///
    /// When calling this method, you have to ensure that
    /// the pointer is [convertible to a reference](crate::ptr#pointer-to-reference-conversion).
    /// Note that because the created reference is to `MaybeUninit<T>`, the
    /// source pointer can point to uninitialized memory.
    #[inline]
    #[must_use]
    #[unstable(feature = "ptr_as_uninit", issue = "75402")]
    pub const unsafe fn as_uninit_ref<'a>(self) -> &'a MaybeUninit<T> {
        // SAFETY: the caller must guarantee that `self` meets all the
        // requirements for a reference.
        unsafe { &*self.cast().as_ptr() }
    }

    /// Returns a unique references to the value. In contrast to [`as_mut`], this does not require
    /// that the value has to be initialized.
    ///
    /// For the shared counterpart see [`as_uninit_ref`].
    ///
    /// [`as_mut`]: NonNull::as_mut
    /// [`as_uninit_ref`]: NonNull::as_uninit_ref
    ///
    /// # Safety
    ///
    /// When calling this method, you have to ensure that
    /// the pointer is [convertible to a reference](crate::ptr#pointer-to-reference-conversion).
    /// Note that because the created reference is to `MaybeUninit<T>`, the
    /// source pointer can point to uninitialized memory.
    #[inline]
    #[must_use]
    #[unstable(feature = "ptr_as_uninit", issue = "75402")]
    pub const unsafe fn as_uninit_mut<'a>(self) -> &'a mut MaybeUninit<T> {
        // SAFETY: the caller must guarantee that `self` meets all the
        // requirements for a reference.
        unsafe { &mut *self.cast().as_ptr() }
    }
}

impl<T: PointeeSized> NonNull<T> {
    /// Creates a new `NonNull`.
    ///
    /// # Safety
    ///
    /// `ptr` must be non-null.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::ptr::NonNull;
    ///
    /// let mut x = 0u32;
    /// let ptr = unsafe { NonNull::new_unchecked(&mut x as *mut _) };
    /// ```
    ///
    /// *Incorrect* usage of this function:
    ///
    /// ```rust,no_run
    /// use std::ptr::NonNull;
    ///
    /// // NEVER DO THAT!!! This is undefined behavior. ⚠️
    /// let ptr = unsafe { NonNull::<u32>::new_unchecked(std::ptr::null_mut()) };
    /// ```
    #[stable(feature = "nonnull", since = "1.25.0")]
    #[rustc_const_stable(feature = "const_nonnull_new_unchecked", since = "1.25.0")]
    #[inline]
    #[track_caller]
    pub const unsafe fn new_unchecked(ptr: *mut T) -> Self {
        // SAFETY: the caller must guarantee that `ptr` is non-null.
        unsafe {
            assert_unsafe_precondition!(
                check_language_ub,
                "NonNull::new_unchecked requires that the pointer is non-null",
                (ptr: *mut () = ptr as *mut ()) => !ptr.is_null()
            );
            NonNull { pointer: ptr as _ }
        }
    }

    /// Creates a new `NonNull` if `ptr` is non-null.
    ///
    /// # Panics during const evaluation
    ///
    /// This method will panic during const evaluation if the pointer cannot be
    /// determined to be null or not. See [`is_null`] for more information.
    ///
    /// [`is_null`]: ../primitive.pointer.html#method.is_null-1
    ///
    /// # Examples
    ///
    /// ```
    /// use std::ptr::NonNull;
    ///
    /// let mut x = 0u32;
    /// let ptr = NonNull::<u32>::new(&mut x as *mut _).expect("ptr is null!");
    ///
    /// if let Some(ptr) = NonNull::<u32>::new(std::ptr::null_mut()) {
    ///     unreachable!();
    /// }
    /// ```
    #[stable(feature = "nonnull", since = "1.25.0")]
    #[rustc_const_stable(feature = "const_nonnull_new", since = "1.85.0")]
    #[inline]
    pub const fn new(ptr: *mut T) -> Option<Self> {
        if !ptr.is_null() {
            // SAFETY: The pointer is already checked and is not null
            Some(unsafe { Self::new_unchecked(ptr) })
        } else {
            None
        }
    }

    /// Converts a reference to a `NonNull` pointer.
    #[stable(feature = "non_null_from_ref", since = "CURRENT_RUSTC_VERSION")]
    #[rustc_const_stable(feature = "non_null_from_ref", since = "CURRENT_RUSTC_VERSION")]
    #[inline]
    pub const fn from_ref(r: &T) -> Self {
        // SAFETY: A reference cannot be null.
        unsafe { NonNull { pointer: r as *const T } }
    }

    /// Converts a mutable reference to a `NonNull` pointer.
    #[stable(feature = "non_null_from_ref", since = "CURRENT_RUSTC_VERSION")]
    #[rustc_const_stable(feature = "non_null_from_ref", since = "CURRENT_RUSTC_VERSION")]
    #[inline]
    pub const fn from_mut(r: &mut T) -> Self {
        // SAFETY: A mutable reference cannot be null.
        unsafe { NonNull { pointer: r as *mut T } }
    }

    /// Performs the same functionality as [`std::ptr::from_raw_parts`], except that a
    /// `NonNull` pointer is returned, as opposed to a raw `*const` pointer.
    ///
    /// See the documentation of [`std::ptr::from_raw_parts`] for more details.
    ///
    /// [`std::ptr::from_raw_parts`]: crate::ptr::from_raw_parts
    #[unstable(feature = "ptr_metadata", issue = "81513")]
    #[inline]
    pub const fn from_raw_parts(
        data_pointer: NonNull<impl super::Thin>,
        metadata: <T as super::Pointee>::Metadata,
    ) -> NonNull<T> {
        // SAFETY: The result of `ptr::from::raw_parts_mut` is non-null because `data_pointer` is.
        unsafe {
            NonNull::new_unchecked(super::from_raw_parts_mut(data_pointer.as_ptr(), metadata))
        }
    }

    /// Decompose a (possibly wide) pointer into its data pointer and metadata components.
    ///
    /// The pointer can be later reconstructed with [`NonNull::from_raw_parts`].
    #[unstable(feature = "ptr_metadata", issue = "81513")]
    #[must_use = "this returns the result of the operation, \
                  without modifying the original"]
    #[inline]
    pub const fn to_raw_parts(self) -> (NonNull<()>, <T as super::Pointee>::Metadata) {
        (self.cast(), super::metadata(self.as_ptr()))
    }

    /// Gets the "address" portion of the pointer.
    ///
    /// For more details, see the equivalent method on a raw pointer, [`pointer::addr`].
    ///
    /// This is a [Strict Provenance][crate::ptr#strict-provenance] API.
    #[must_use]
    #[inline]
    #[stable(feature = "strict_provenance", since = "1.84.0")]
    pub fn addr(self) -> NonZero<usize> {
        // SAFETY: The pointer is guaranteed by the type to be non-null,
        // meaning that the address will be non-zero.
        unsafe { NonZero::new_unchecked(self.as_ptr().addr()) }
    }

    /// Exposes the ["provenance"][crate::ptr#provenance] part of the pointer for future use in
    /// [`with_exposed_provenance`][NonNull::with_exposed_provenance] and returns the "address" portion.
    ///
    /// For more details, see the equivalent method on a raw pointer, [`pointer::expose_provenance`].
    ///
    /// This is an [Exposed Provenance][crate::ptr#exposed-provenance] API.
    #[stable(feature = "nonnull_provenance", since = "CURRENT_RUSTC_VERSION")]
    pub fn expose_provenance(self) -> NonZero<usize> {
        // SAFETY: The pointer is guaranteed by the type to be non-null,
        // meaning that the address will be non-zero.
        unsafe { NonZero::new_unchecked(self.as_ptr().expose_provenance()) }
    }

    /// Creates a new pointer with the given address and the [provenance][crate::ptr#provenance] of
    /// `self`.
    ///
    /// For more details, see the equivalent method on a raw pointer, [`pointer::with_addr`].
    ///
    /// This is a [Strict Provenance][crate::ptr#strict-provenance] API.
    #[must_use]
    #[inline]
    #[stable(feature = "strict_provenance", since = "1.84.0")]
    pub fn with_addr(self, addr: NonZero<usize>) -> Self {
        // SAFETY: The result of `ptr::from::with_addr` is non-null because `addr` is guaranteed to be non-zero.
        unsafe { NonNull::new_unchecked(self.as_ptr().with_addr(addr.get()) as *mut _) }
    }

    /// Creates a new pointer by mapping `self`'s address to a new one, preserving the
    /// [provenance][crate::ptr#provenance] of `self`.
    ///
    /// For more details, see the equivalent method on a raw pointer, [`pointer::map_addr`].
    ///
    /// This is a [Strict Provenance][crate::ptr#strict-provenance] API.
    #[must_use]
    #[inline]
    #[stable(feature = "strict_provenance", since = "1.84.0")]
    pub fn map_addr(self, f: impl FnOnce(NonZero<usize>) -> NonZero<usize>) -> Self {
        self.with_addr(f(self.addr()))
    }

    /// Acquires the underlying `*mut` pointer.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::ptr::NonNull;
    ///
    /// let mut x = 0u32;
    /// let ptr = NonNull::new(&mut x).expect("ptr is null!");
    ///
    /// let x_value = unsafe { *ptr.as_ptr() };
    /// assert_eq!(x_value, 0);
    ///
    /// unsafe { *ptr.as_ptr() += 2; }
    /// let x_value = unsafe { *ptr.as_ptr() };
    /// assert_eq!(x_value, 2);
    /// ```
    #[stable(feature = "nonnull", since = "1.25.0")]
    #[rustc_const_stable(feature = "const_nonnull_as_ptr", since = "1.32.0")]
    #[rustc_never_returns_null_ptr]
    #[must_use]
    #[inline(always)]
    pub const fn as_ptr(self) -> *mut T {
        // This is a transmute for the same reasons as `NonZero::get`.

        // SAFETY: `NonNull` is `transparent` over a `*const T`, and `*const T`
        // and `*mut T` have the same layout, so transitively we can transmute
        // our `NonNull` to a `*mut T` directly.
        unsafe { mem::transmute::<Self, *mut T>(self) }
    }

    /// Returns a shared reference to the value. If the value may be uninitialized, [`as_uninit_ref`]
    /// must be used instead.
    ///
    /// For the mutable counterpart see [`as_mut`].
    ///
    /// [`as_uninit_ref`]: NonNull::as_uninit_ref
    /// [`as_mut`]: NonNull::as_mut
    ///
    /// # Safety
    ///
    /// When calling this method, you have to ensure that
    /// the pointer is [convertible to a reference](crate::ptr#pointer-to-reference-conversion).
    ///
    /// # Examples
    ///
    /// ```
    /// use std::ptr::NonNull;
    ///
    /// let mut x = 0u32;
    /// let ptr = NonNull::new(&mut x as *mut _).expect("ptr is null!");
    ///
    /// let ref_x = unsafe { ptr.as_ref() };
    /// println!("{ref_x}");
    /// ```
    ///
    /// [the module documentation]: crate::ptr#safety
    #[stable(feature = "nonnull", since = "1.25.0")]
    #[rustc_const_stable(feature = "const_nonnull_as_ref", since = "1.73.0")]
    #[must_use]
    #[inline(always)]
    pub const unsafe fn as_ref<'a>(&self) -> &'a T {
        // SAFETY: the caller must guarantee that `self` meets all the
        // requirements for a reference.
        // `cast_const` avoids a mutable raw pointer deref.
        unsafe { &*self.as_ptr().cast_const() }
    }

    /// Returns a unique reference to the value. If the value may be uninitialized, [`as_uninit_mut`]
    /// must be used instead.
    ///
    /// For the shared counterpart see [`as_ref`].
    ///
    /// [`as_uninit_mut`]: NonNull::as_uninit_mut
    /// [`as_ref`]: NonNull::as_ref
    ///
    /// # Safety
    ///
    /// When calling this method, you have to ensure that
    /// the pointer is [convertible to a reference](crate::ptr#pointer-to-reference-conversion).
    /// # Examples
    ///
    /// ```
    /// use std::ptr::NonNull;
    ///
    /// let mut x = 0u32;
    /// let mut ptr = NonNull::new(&mut x).expect("null pointer");
    ///
    /// let x_ref = unsafe { ptr.as_mut() };
    /// assert_eq!(*x_ref, 0);
    /// *x_ref += 2;
    /// assert_eq!(*x_ref, 2);
    /// ```
    ///
    /// [the module documentation]: crate::ptr#safety
    #[stable(feature = "nonnull", since = "1.25.0")]
    #[rustc_const_stable(feature = "const_ptr_as_ref", since = "1.83.0")]
    #[must_use]
    #[inline(always)]
    pub const unsafe fn as_mut<'a>(&mut self) -> &'a mut T {
        // SAFETY: the caller must guarantee that `self` meets all the
        // requirements for a mutable reference.
        unsafe { &mut *self.as_ptr() }
    }

    /// Casts to a pointer of another type.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::ptr::NonNull;
    ///
    /// let mut x = 0u32;
    /// let ptr = NonNull::new(&mut x as *mut _).expect("null pointer");
    ///
    /// let casted_ptr = ptr.cast::<i8>();
    /// let raw_ptr: *mut i8 = casted_ptr.as_ptr();
    /// ```
    #[stable(feature = "nonnull_cast", since = "1.27.0")]
    #[rustc_const_stable(feature = "const_nonnull_cast", since = "1.36.0")]
    #[must_use = "this returns the result of the operation, \
                  without modifying the original"]
    #[inline]
    pub const fn cast<U>(self) -> NonNull<U> {
        // SAFETY: `self` is a `NonNull` pointer which is necessarily non-null
        unsafe { NonNull { pointer: self.as_ptr() as *mut U } }
    }

    /// Try to cast to a pointer of another type by checking aligment.
    ///
    /// If the pointer is properly aligned to the target type, it will be
    /// cast to the target type. Otherwise, `None` is returned.
    ///
    /// # Examples
    ///
    /// ```rust
    /// #![feature(pointer_try_cast_aligned)]
    /// use std::ptr::NonNull;
    ///
    /// let mut x = 0u64;
    ///
    /// let aligned = NonNull::from_mut(&mut x);
    /// let unaligned = unsafe { aligned.byte_add(1) };
    ///
    /// assert!(aligned.try_cast_aligned::<u32>().is_some());
    /// assert!(unaligned.try_cast_aligned::<u32>().is_none());
    /// ```
    #[unstable(feature = "pointer_try_cast_aligned", issue = "141221")]
    #[must_use = "this returns the result of the operation, \
                  without modifying the original"]
    #[inline]
    pub fn try_cast_aligned<U>(self) -> Option<NonNull<U>> {
        if self.is_aligned_to(align_of::<U>()) { Some(self.cast()) } else { None }
    }

    /// Adds an offset to a pointer.
    ///
    /// `count` is in units of T; e.g., a `count` of 3 represents a pointer
    /// offset of `3 * size_of::<T>()` bytes.
    ///
    /// # Safety
    ///
    /// If any of the following conditions are violated, the result is Undefined Behavior:
    ///
    /// * The computed offset, `count * size_of::<T>()` bytes, must not overflow `isize`.
    ///
    /// * If the computed offset is non-zero, then `self` must be derived from a pointer to some
    ///   [allocation], and the entire memory range between `self` and the result must be in
    ///   bounds of that allocation. In particular, this range must not "wrap around" the edge
    ///   of the address space.
    ///
    /// Allocations can never be larger than `isize::MAX` bytes, so if the computed offset
    /// stays in bounds of the allocation, it is guaranteed to satisfy the first requirement.
    /// This implies, for instance, that `vec.as_ptr().add(vec.len())` (for `vec: Vec<T>`) is always
    /// safe.
    ///
    /// [allocation]: crate::ptr#allocation
    ///
    /// # Examples
    ///
    /// ```
    /// use std::ptr::NonNull;
    ///
    /// let mut s = [1, 2, 3];
    /// let ptr: NonNull<u32> = NonNull::new(s.as_mut_ptr()).unwrap();
    ///
    /// unsafe {
    ///     println!("{}", ptr.offset(1).read());
    ///     println!("{}", ptr.offset(2).read());
    /// }
    /// ```
    #[inline(always)]
    #[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
    #[must_use = "returns a new pointer rather than modifying its argument"]
    #[stable(feature = "non_null_convenience", since = "1.80.0")]
    #[rustc_const_stable(feature = "non_null_convenience", since = "1.80.0")]
    pub const unsafe fn offset(self, count: isize) -> Self
    where
        T: Sized,
    {
        // SAFETY: the caller must uphold the safety contract for `offset`.
        // Additionally safety contract of `offset` guarantees that the resulting pointer is
        // pointing to an allocation, there can't be an allocation at null, thus it's safe to
        // construct `NonNull`.
        unsafe { NonNull { pointer: intrinsics::offset(self.as_ptr(), count) } }
    }

    /// Calculates the offset from a pointer in bytes.
    ///
    /// `count` is in units of **bytes**.
    ///
    /// This is purely a convenience for casting to a `u8` pointer and
    /// using [offset][pointer::offset] on it. See that method for documentation
    /// and safety requirements.
    ///
    /// For non-`Sized` pointees this operation changes only the data pointer,
    /// leaving the metadata untouched.
    #[must_use]
    #[inline(always)]
    #[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
    #[stable(feature = "non_null_convenience", since = "1.80.0")]
    #[rustc_const_stable(feature = "non_null_convenience", since = "1.80.0")]
    pub const unsafe fn byte_offset(self, count: isize) -> Self {
        // SAFETY: the caller must uphold the safety contract for `offset` and `byte_offset` has
        // the same safety contract.
        // Additionally safety contract of `offset` guarantees that the resulting pointer is
        // pointing to an allocation, there can't be an allocation at null, thus it's safe to
        // construct `NonNull`.
        unsafe { NonNull { pointer: self.as_ptr().byte_offset(count) } }
    }

    /// Adds an offset to a pointer (convenience for `.offset(count as isize)`).
    ///
    /// `count` is in units of T; e.g., a `count` of 3 represents a pointer
    /// offset of `3 * size_of::<T>()` bytes.
    ///
    /// # Safety
    ///
    /// If any of the following conditions are violated, the result is Undefined Behavior:
    ///
    /// * The computed offset, `count * size_of::<T>()` bytes, must not overflow `isize`.
    ///
    /// * If the computed offset is non-zero, then `self` must be derived from a pointer to some
    ///   [allocation], and the entire memory range between `self` and the result must be in
    ///   bounds of that allocation. In particular, this range must not "wrap around" the edge
    ///   of the address space.
    ///
    /// Allocations can never be larger than `isize::MAX` bytes, so if the computed offset
    /// stays in bounds of the allocation, it is guaranteed to satisfy the first requirement.
    /// This implies, for instance, that `vec.as_ptr().add(vec.len())` (for `vec: Vec<T>`) is always
    /// safe.
    ///
    /// [allocation]: crate::ptr#allocation
    ///
    /// # Examples
    ///
    /// ```
    /// use std::ptr::NonNull;
    ///
    /// let s: &str = "123";
    /// let ptr: NonNull<u8> = NonNull::new(s.as_ptr().cast_mut()).unwrap();
    ///
    /// unsafe {
    ///     println!("{}", ptr.add(1).read() as char);
    ///     println!("{}", ptr.add(2).read() as char);
    /// }
    /// ```
    #[inline(always)]
    #[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
    #[must_use = "returns a new pointer rather than modifying its argument"]
    #[stable(feature = "non_null_convenience", since = "1.80.0")]
    #[rustc_const_stable(feature = "non_null_convenience", since = "1.80.0")]
    pub const unsafe fn add(self, count: usize) -> Self
    where
        T: Sized,
    {
        // SAFETY: the caller must uphold the safety contract for `offset`.
        // Additionally safety contract of `offset` guarantees that the resulting pointer is
        // pointing to an allocation, there can't be an allocation at null, thus it's safe to
        // construct `NonNull`.
        unsafe { NonNull { pointer: intrinsics::offset(self.as_ptr(), count) } }
    }

    /// Calculates the offset from a pointer in bytes (convenience for `.byte_offset(count as isize)`).
    ///
    /// `count` is in units of bytes.
    ///
    /// This is purely a convenience for casting to a `u8` pointer and
    /// using [`add`][NonNull::add] on it. See that method for documentation
    /// and safety requirements.
    ///
    /// For non-`Sized` pointees this operation changes only the data pointer,
    /// leaving the metadata untouched.
    #[must_use]
    #[inline(always)]
    #[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
    #[stable(feature = "non_null_convenience", since = "1.80.0")]
    #[rustc_const_stable(feature = "non_null_convenience", since = "1.80.0")]
    pub const unsafe fn byte_add(self, count: usize) -> Self {
        // SAFETY: the caller must uphold the safety contract for `add` and `byte_add` has the same
        // safety contract.
        // Additionally safety contract of `add` guarantees that the resulting pointer is pointing
        // to an allocation, there can't be an allocation at null, thus it's safe to construct
        // `NonNull`.
        unsafe { NonNull { pointer: self.as_ptr().byte_add(count) } }
    }

    /// Subtracts an offset from a pointer (convenience for
    /// `.offset((count as isize).wrapping_neg())`).
    ///
    /// `count` is in units of T; e.g., a `count` of 3 represents a pointer
    /// offset of `3 * size_of::<T>()` bytes.
    ///
    /// # Safety
    ///
    /// If any of the following conditions are violated, the result is Undefined Behavior:
    ///
    /// * The computed offset, `count * size_of::<T>()` bytes, must not overflow `isize`.
    ///
    /// * If the computed offset is non-zero, then `self` must be derived from a pointer to some
    ///   [allocation], and the entire memory range between `self` and the result must be in
    ///   bounds of that allocation. In particular, this range must not "wrap around" the edge
    ///   of the address space.
    ///
    /// Allocations can never be larger than `isize::MAX` bytes, so if the computed offset
    /// stays in bounds of the allocation, it is guaranteed to satisfy the first requirement.
    /// This implies, for instance, that `vec.as_ptr().add(vec.len())` (for `vec: Vec<T>`) is always
    /// safe.
    ///
    /// [allocation]: crate::ptr#allocation
    ///
    /// # Examples
    ///
    /// ```
    /// use std::ptr::NonNull;
    ///
    /// let s: &str = "123";
    ///
    /// unsafe {
    ///     let end: NonNull<u8> = NonNull::new(s.as_ptr().cast_mut()).unwrap().add(3);
    ///     println!("{}", end.sub(1).read() as char);
    ///     println!("{}", end.sub(2).read() as char);
    /// }
    /// ```
    #[inline(always)]
    #[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
    #[must_use = "returns a new pointer rather than modifying its argument"]
    #[stable(feature = "non_null_convenience", since = "1.80.0")]
    #[rustc_const_stable(feature = "non_null_convenience", since = "1.80.0")]
    pub const unsafe fn sub(self, count: usize) -> Self
    where
        T: Sized,
    {
        if T::IS_ZST {
            // Pointer arithmetic does nothing when the pointee is a ZST.
            self
        } else {
            // SAFETY: the caller must uphold the safety contract for `offset`.
            // Because the pointee is *not* a ZST, that means that `count` is
            // at most `isize::MAX`, and thus the negation cannot overflow.
            unsafe { self.offset((count as isize).unchecked_neg()) }
        }
    }

    /// Calculates the offset from a pointer in bytes (convenience for
    /// `.byte_offset((count as isize).wrapping_neg())`).
    ///
    /// `count` is in units of bytes.
    ///
    /// This is purely a convenience for casting to a `u8` pointer and
    /// using [`sub`][NonNull::sub] on it. See that method for documentation
    /// and safety requirements.
    ///
    /// For non-`Sized` pointees this operation changes only the data pointer,
    /// leaving the metadata untouched.
    #[must_use]
    #[inline(always)]
    #[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
    #[stable(feature = "non_null_convenience", since = "1.80.0")]
    #[rustc_const_stable(feature = "non_null_convenience", since = "1.80.0")]
    pub const unsafe fn byte_sub(self, count: usize) -> Self {
        // SAFETY: the caller must uphold the safety contract for `sub` and `byte_sub` has the same
        // safety contract.
        // Additionally safety contract of `sub` guarantees that the resulting pointer is pointing
        // to an allocation, there can't be an allocation at null, thus it's safe to construct
        // `NonNull`.
        unsafe { NonNull { pointer: self.as_ptr().byte_sub(count) } }
    }

    /// Calculates the distance between two pointers within the same allocation. The returned value is in
    /// units of T: the distance in bytes divided by `size_of::<T>()`.
    ///
    /// This is equivalent to `(self as isize - origin as isize) / (size_of::<T>() as isize)`,
    /// except that it has a lot more opportunities for UB, in exchange for the compiler
    /// better understanding what you are doing.
    ///
    /// The primary motivation of this method is for computing the `len` of an array/slice
    /// of `T` that you are currently representing as a "start" and "end" pointer
    /// (and "end" is "one past the end" of the array).
    /// In that case, `end.offset_from(start)` gets you the length of the array.
    ///
    /// All of the following safety requirements are trivially satisfied for this usecase.
    ///
    /// [`offset`]: #method.offset
    ///
    /// # Safety
    ///
    /// If any of the following conditions are violated, the result is Undefined Behavior:
    ///
    /// * `self` and `origin` must either
    ///
    ///   * point to the same address, or
    ///   * both be *derived from* a pointer to the same [allocation], and the memory range between
    ///     the two pointers must be in bounds of that object. (See below for an example.)
    ///
    /// * The distance between the pointers, in bytes, must be an exact multiple
    ///   of the size of `T`.
    ///
    /// As a consequence, the absolute distance between the pointers, in bytes, computed on
    /// mathematical integers (without "wrapping around"), cannot overflow an `isize`. This is
    /// implied by the in-bounds requirement, and the fact that no allocation can be larger
    /// than `isize::MAX` bytes.
    ///
    /// The requirement for pointers to be derived from the same allocation is primarily
    /// needed for `const`-compatibility: the distance between pointers into *different* allocated
    /// objects is not known at compile-time. However, the requirement also exists at
    /// runtime and may be exploited by optimizations. If you wish to compute the difference between
    /// pointers that are not guaranteed to be from the same allocation, use `(self as isize -
    /// origin as isize) / size_of::<T>()`.
    // FIXME: recommend `addr()` instead of `as usize` once that is stable.
    ///
    /// [`add`]: #method.add
    /// [allocation]: crate::ptr#allocation
    ///
    /// # Panics
    ///
    /// This function panics if `T` is a Zero-Sized Type ("ZST").
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// use std::ptr::NonNull;
    ///
    /// let a = [0; 5];
    /// let ptr1: NonNull<u32> = NonNull::from(&a[1]);
    /// let ptr2: NonNull<u32> = NonNull::from(&a[3]);
    /// unsafe {
    ///     assert_eq!(ptr2.offset_from(ptr1), 2);
    ///     assert_eq!(ptr1.offset_from(ptr2), -2);
    ///     assert_eq!(ptr1.offset(2), ptr2);
    ///     assert_eq!(ptr2.offset(-2), ptr1);
    /// }
    /// ```
    ///
    /// *Incorrect* usage:
    ///
    /// ```rust,no_run
    /// use std::ptr::NonNull;
    ///
    /// let ptr1 = NonNull::new(Box::into_raw(Box::new(0u8))).unwrap();
    /// let ptr2 = NonNull::new(Box::into_raw(Box::new(1u8))).unwrap();
    /// let diff = (ptr2.addr().get() as isize).wrapping_sub(ptr1.addr().get() as isize);
    /// // Make ptr2_other an "alias" of ptr2.add(1), but derived from ptr1.
    /// let diff_plus_1 = diff.wrapping_add(1);
    /// let ptr2_other = NonNull::new(ptr1.as_ptr().wrapping_byte_offset(diff_plus_1)).unwrap();
    /// assert_eq!(ptr2.addr(), ptr2_other.addr());
    /// // Since ptr2_other and ptr2 are derived from pointers to different objects,
    /// // computing their offset is undefined behavior, even though
    /// // they point to addresses that are in-bounds of the same object!
    ///
    /// let one = unsafe { ptr2_other.offset_from(ptr2) }; // Undefined Behavior! ⚠️
    /// ```
    #[inline]
    #[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
    #[stable(feature = "non_null_convenience", since = "1.80.0")]
    #[rustc_const_stable(feature = "non_null_convenience", since = "1.80.0")]
    pub const unsafe fn offset_from(self, origin: NonNull<T>) -> isize
    where
        T: Sized,
    {
        // SAFETY: the caller must uphold the safety contract for `offset_from`.
        unsafe { self.as_ptr().offset_from(origin.as_ptr()) }
    }

    /// Calculates the distance between two pointers within the same allocation. The returned value is in
    /// units of **bytes**.
    ///
    /// This is purely a convenience for casting to a `u8` pointer and
    /// using [`offset_from`][NonNull::offset_from] on it. See that method for
    /// documentation and safety requirements.
    ///
    /// For non-`Sized` pointees this operation considers only the data pointers,
    /// ignoring the metadata.
    #[inline(always)]
    #[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
    #[stable(feature = "non_null_convenience", since = "1.80.0")]
    #[rustc_const_stable(feature = "non_null_convenience", since = "1.80.0")]
    pub const unsafe fn byte_offset_from<U: ?Sized>(self, origin: NonNull<U>) -> isize {
        // SAFETY: the caller must uphold the safety contract for `byte_offset_from`.
        unsafe { self.as_ptr().byte_offset_from(origin.as_ptr()) }
    }

    // N.B. `wrapping_offset``, `wrapping_add`, etc are not implemented because they can wrap to null

    /// Calculates the distance between two pointers within the same allocation, *where it's known that
    /// `self` is equal to or greater than `origin`*. The returned value is in
    /// units of T: the distance in bytes is divided by `size_of::<T>()`.
    ///
    /// This computes the same value that [`offset_from`](#method.offset_from)
    /// would compute, but with the added precondition that the offset is
    /// guaranteed to be non-negative.  This method is equivalent to
    /// `usize::try_from(self.offset_from(origin)).unwrap_unchecked()`,
    /// but it provides slightly more information to the optimizer, which can
    /// sometimes allow it to optimize slightly better with some backends.
    ///
    /// This method can be though of as recovering the `count` that was passed
    /// to [`add`](#method.add) (or, with the parameters in the other order,
    /// to [`sub`](#method.sub)).  The following are all equivalent, assuming
    /// that their safety preconditions are met:
    /// ```rust
    /// # unsafe fn blah(ptr: std::ptr::NonNull<u32>, origin: std::ptr::NonNull<u32>, count: usize) -> bool { unsafe {
    /// ptr.offset_from_unsigned(origin) == count
    /// # &&
    /// origin.add(count) == ptr
    /// # &&
    /// ptr.sub(count) == origin
    /// # } }
    /// ```
    ///
    /// # Safety
    ///
    /// - The distance between the pointers must be non-negative (`self >= origin`)
    ///
    /// - *All* the safety conditions of [`offset_from`](#method.offset_from)
    ///   apply to this method as well; see it for the full details.
    ///
    /// Importantly, despite the return type of this method being able to represent
    /// a larger offset, it's still *not permitted* to pass pointers which differ
    /// by more than `isize::MAX` *bytes*.  As such, the result of this method will
    /// always be less than or equal to `isize::MAX as usize`.
    ///
    /// # Panics
    ///
    /// This function panics if `T` is a Zero-Sized Type ("ZST").
    ///
    /// # Examples
    ///
    /// ```
    /// use std::ptr::NonNull;
    ///
    /// let a = [0; 5];
    /// let ptr1: NonNull<u32> = NonNull::from(&a[1]);
    /// let ptr2: NonNull<u32> = NonNull::from(&a[3]);
    /// unsafe {
    ///     assert_eq!(ptr2.offset_from_unsigned(ptr1), 2);
    ///     assert_eq!(ptr1.add(2), ptr2);
    ///     assert_eq!(ptr2.sub(2), ptr1);
    ///     assert_eq!(ptr2.offset_from_unsigned(ptr2), 0);
    /// }
    ///
    /// // This would be incorrect, as the pointers are not correctly ordered:
    /// // ptr1.offset_from_unsigned(ptr2)
    /// ```
    #[inline]
    #[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
    #[stable(feature = "ptr_sub_ptr", since = "1.87.0")]
    #[rustc_const_stable(feature = "const_ptr_sub_ptr", since = "1.87.0")]
    pub const unsafe fn offset_from_unsigned(self, subtracted: NonNull<T>) -> usize
    where
        T: Sized,
    {
        // SAFETY: the caller must uphold the safety contract for `offset_from_unsigned`.
        unsafe { self.as_ptr().offset_from_unsigned(subtracted.as_ptr()) }
    }

    /// Calculates the distance between two pointers within the same allocation, *where it's known that
    /// `self` is equal to or greater than `origin`*. The returned value is in
    /// units of **bytes**.
    ///
    /// This is purely a convenience for casting to a `u8` pointer and
    /// using [`offset_from_unsigned`][NonNull::offset_from_unsigned] on it.
    /// See that method for documentation and safety requirements.
    ///
    /// For non-`Sized` pointees this operation considers only the data pointers,
    /// ignoring the metadata.
    #[inline(always)]
    #[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
    #[stable(feature = "ptr_sub_ptr", since = "1.87.0")]
    #[rustc_const_stable(feature = "const_ptr_sub_ptr", since = "1.87.0")]
    pub const unsafe fn byte_offset_from_unsigned<U: ?Sized>(self, origin: NonNull<U>) -> usize {
        // SAFETY: the caller must uphold the safety contract for `byte_offset_from_unsigned`.
        unsafe { self.as_ptr().byte_offset_from_unsigned(origin.as_ptr()) }
    }

    /// Reads the value from `self` without moving it. This leaves the
    /// memory in `self` unchanged.
    ///
    /// See [`ptr::read`] for safety concerns and examples.
    ///
    /// [`ptr::read`]: crate::ptr::read()
    #[inline]
    #[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
    #[stable(feature = "non_null_convenience", since = "1.80.0")]
    #[rustc_const_stable(feature = "non_null_convenience", since = "1.80.0")]
    pub const unsafe fn read(self) -> T
    where
        T: Sized,
    {
        // SAFETY: the caller must uphold the safety contract for `read`.
        unsafe { ptr::read(self.as_ptr()) }
    }

    /// Performs a volatile read of the value from `self` without moving it. This
    /// leaves the memory in `self` unchanged.
    ///
    /// Volatile operations are intended to act on I/O memory, and are guaranteed
    /// to not be elided or reordered by the compiler across other volatile
    /// operations.
    ///
    /// See [`ptr::read_volatile`] for safety concerns and examples.
    ///
    /// [`ptr::read_volatile`]: crate::ptr::read_volatile()
    #[inline]
    #[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
    #[stable(feature = "non_null_convenience", since = "1.80.0")]
    pub unsafe fn read_volatile(self) -> T
    where
        T: Sized,
    {
        // SAFETY: the caller must uphold the safety contract for `read_volatile`.
        unsafe { ptr::read_volatile(self.as_ptr()) }
    }

    /// Reads the value from `self` without moving it. This leaves the
    /// memory in `self` unchanged.
    ///
    /// Unlike `read`, the pointer may be unaligned.
    ///
    /// See [`ptr::read_unaligned`] for safety concerns and examples.
    ///
    /// [`ptr::read_unaligned`]: crate::ptr::read_unaligned()
    #[inline]
    #[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
    #[stable(feature = "non_null_convenience", since = "1.80.0")]
    #[rustc_const_stable(feature = "non_null_convenience", since = "1.80.0")]
    pub const unsafe fn read_unaligned(self) -> T
    where
        T: Sized,
    {
        // SAFETY: the caller must uphold the safety contract for `read_unaligned`.
        unsafe { ptr::read_unaligned(self.as_ptr()) }
    }

    /// Copies `count * size_of::<T>()` bytes from `self` to `dest`. The source
    /// and destination may overlap.
    ///
    /// NOTE: this has the *same* argument order as [`ptr::copy`].
    ///
    /// See [`ptr::copy`] for safety concerns and examples.
    ///
    /// [`ptr::copy`]: crate::ptr::copy()
    #[inline(always)]
    #[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
    #[stable(feature = "non_null_convenience", since = "1.80.0")]
    #[rustc_const_stable(feature = "const_intrinsic_copy", since = "1.83.0")]
    pub const unsafe fn copy_to(self, dest: NonNull<T>, count: usize)
    where
        T: Sized,
    {
        // SAFETY: the caller must uphold the safety contract for `copy`.
        unsafe { ptr::copy(self.as_ptr(), dest.as_ptr(), count) }
    }

    /// Copies `count * size_of::<T>()` bytes from `self` to `dest`. The source
    /// and destination may *not* overlap.
    ///
    /// NOTE: this has the *same* argument order as [`ptr::copy_nonoverlapping`].
    ///
    /// See [`ptr::copy_nonoverlapping`] for safety concerns and examples.
    ///
    /// [`ptr::copy_nonoverlapping`]: crate::ptr::copy_nonoverlapping()
    #[inline(always)]
    #[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
    #[stable(feature = "non_null_convenience", since = "1.80.0")]
    #[rustc_const_stable(feature = "const_intrinsic_copy", since = "1.83.0")]
    pub const unsafe fn copy_to_nonoverlapping(self, dest: NonNull<T>, count: usize)
    where
        T: Sized,
    {
        // SAFETY: the caller must uphold the safety contract for `copy_nonoverlapping`.
        unsafe { ptr::copy_nonoverlapping(self.as_ptr(), dest.as_ptr(), count) }
    }

    /// Copies `count * size_of::<T>()` bytes from `src` to `self`. The source
    /// and destination may overlap.
    ///
    /// NOTE: this has the *opposite* argument order of [`ptr::copy`].
    ///
    /// See [`ptr::copy`] for safety concerns and examples.
    ///
    /// [`ptr::copy`]: crate::ptr::copy()
    #[inline(always)]
    #[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
    #[stable(feature = "non_null_convenience", since = "1.80.0")]
    #[rustc_const_stable(feature = "const_intrinsic_copy", since = "1.83.0")]
    pub const unsafe fn copy_from(self, src: NonNull<T>, count: usize)
    where
        T: Sized,
    {
        // SAFETY: the caller must uphold the safety contract for `copy`.
        unsafe { ptr::copy(src.as_ptr(), self.as_ptr(), count) }
    }

    /// Copies `count * size_of::<T>()` bytes from `src` to `self`. The source
    /// and destination may *not* overlap.
    ///
    /// NOTE: this has the *opposite* argument order of [`ptr::copy_nonoverlapping`].
    ///
    /// See [`ptr::copy_nonoverlapping`] for safety concerns and examples.
    ///
    /// [`ptr::copy_nonoverlapping`]: crate::ptr::copy_nonoverlapping()
    #[inline(always)]
    #[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
    #[stable(feature = "non_null_convenience", since = "1.80.0")]
    #[rustc_const_stable(feature = "const_intrinsic_copy", since = "1.83.0")]
    pub const unsafe fn copy_from_nonoverlapping(self, src: NonNull<T>, count: usize)
    where
        T: Sized,
    {
        // SAFETY: the caller must uphold the safety contract for `copy_nonoverlapping`.
        unsafe { ptr::copy_nonoverlapping(src.as_ptr(), self.as_ptr(), count) }
    }

    /// Executes the destructor (if any) of the pointed-to value.
    ///
    /// See [`ptr::drop_in_place`] for safety concerns and examples.
    ///
    /// [`ptr::drop_in_place`]: crate::ptr::drop_in_place()
    #[inline(always)]
    #[stable(feature = "non_null_convenience", since = "1.80.0")]
    pub unsafe fn drop_in_place(self) {
        // SAFETY: the caller must uphold the safety contract for `drop_in_place`.
        unsafe { ptr::drop_in_place(self.as_ptr()) }
    }

    /// Overwrites a memory location with the given value without reading or
    /// dropping the old value.
    ///
    /// See [`ptr::write`] for safety concerns and examples.
    ///
    /// [`ptr::write`]: crate::ptr::write()
    #[inline(always)]
    #[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
    #[stable(feature = "non_null_convenience", since = "1.80.0")]
    #[rustc_const_stable(feature = "const_ptr_write", since = "1.83.0")]
    pub const unsafe fn write(self, val: T)
    where
        T: Sized,
    {
        // SAFETY: the caller must uphold the safety contract for `write`.
        unsafe { ptr::write(self.as_ptr(), val) }
    }

    /// Invokes memset on the specified pointer, setting `count * size_of::<T>()`
    /// bytes of memory starting at `self` to `val`.
    ///
    /// See [`ptr::write_bytes`] for safety concerns and examples.
    ///
    /// [`ptr::write_bytes`]: crate::ptr::write_bytes()
    #[inline(always)]
    #[doc(alias = "memset")]
    #[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
    #[stable(feature = "non_null_convenience", since = "1.80.0")]
    #[rustc_const_stable(feature = "const_ptr_write", since = "1.83.0")]
    pub const unsafe fn write_bytes(self, val: u8, count: usize)
    where
        T: Sized,
    {
        // SAFETY: the caller must uphold the safety contract for `write_bytes`.
        unsafe { ptr::write_bytes(self.as_ptr(), val, count) }
    }

    /// Performs a volatile write of a memory location with the given value without
    /// reading or dropping the old value.
    ///
    /// Volatile operations are intended to act on I/O memory, and are guaranteed
    /// to not be elided or reordered by the compiler across other volatile
    /// operations.
    ///
    /// See [`ptr::write_volatile`] for safety concerns and examples.
    ///
    /// [`ptr::write_volatile`]: crate::ptr::write_volatile()
    #[inline(always)]
    #[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
    #[stable(feature = "non_null_convenience", since = "1.80.0")]
    pub unsafe fn write_volatile(self, val: T)
    where
        T: Sized,
    {
        // SAFETY: the caller must uphold the safety contract for `write_volatile`.
        unsafe { ptr::write_volatile(self.as_ptr(), val) }
    }

    /// Overwrites a memory location with the given value without reading or
    /// dropping the old value.
    ///
    /// Unlike `write`, the pointer may be unaligned.
    ///
    /// See [`ptr::write_unaligned`] for safety concerns and examples.
    ///
    /// [`ptr::write_unaligned`]: crate::ptr::write_unaligned()
    #[inline(always)]
    #[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
    #[stable(feature = "non_null_convenience", since = "1.80.0")]
    #[rustc_const_stable(feature = "const_ptr_write", since = "1.83.0")]
    pub const unsafe fn write_unaligned(self, val: T)
    where
        T: Sized,
    {
        // SAFETY: the caller must uphold the safety contract for `write_unaligned`.
        unsafe { ptr::write_unaligned(self.as_ptr(), val) }
    }

    /// Replaces the value at `self` with `src`, returning the old
    /// value, without dropping either.
    ///
    /// See [`ptr::replace`] for safety concerns and examples.
    ///
    /// [`ptr::replace`]: crate::ptr::replace()
    #[inline(always)]
    #[stable(feature = "non_null_convenience", since = "1.80.0")]
    #[rustc_const_stable(feature = "const_inherent_ptr_replace", since = "1.88.0")]
    pub const unsafe fn replace(self, src: T) -> T
    where
        T: Sized,
    {
        // SAFETY: the caller must uphold the safety contract for `replace`.
        unsafe { ptr::replace(self.as_ptr(), src) }
    }

    /// Swaps the values at two mutable locations of the same type, without
    /// deinitializing either. They may overlap, unlike `mem::swap` which is
    /// otherwise equivalent.
    ///
    /// See [`ptr::swap`] for safety concerns and examples.
    ///
    /// [`ptr::swap`]: crate::ptr::swap()
    #[inline(always)]
    #[stable(feature = "non_null_convenience", since = "1.80.0")]
    #[rustc_const_stable(feature = "const_swap", since = "1.85.0")]
    pub const unsafe fn swap(self, with: NonNull<T>)
    where
        T: Sized,
    {
        // SAFETY: the caller must uphold the safety contract for `swap`.
        unsafe { ptr::swap(self.as_ptr(), with.as_ptr()) }
    }

    /// Computes the offset that needs to be applied to the pointer in order to make it aligned to
    /// `align`.
    ///
    /// If it is not possible to align the pointer, the implementation returns
    /// `usize::MAX`.
    ///
    /// The offset is expressed in number of `T` elements, and not bytes.
    ///
    /// There are no guarantees whatsoever that offsetting the pointer will not overflow or go
    /// beyond the allocation that the pointer points into. It is up to the caller to ensure that
    /// the returned offset is correct in all terms other than alignment.
    ///
    /// When this is called during compile-time evaluation (which is unstable), the implementation
    /// may return `usize::MAX` in cases where that can never happen at runtime. This is because the
    /// actual alignment of pointers is not known yet during compile-time, so an offset with
    /// guaranteed alignment can sometimes not be computed. For example, a buffer declared as `[u8;
    /// N]` might be allocated at an odd or an even address, but at compile-time this is not yet
    /// known, so the execution has to be correct for either choice. It is therefore impossible to
    /// find an offset that is guaranteed to be 2-aligned. (This behavior is subject to change, as usual
    /// for unstable APIs.)
    ///
    /// # Panics
    ///
    /// The function panics if `align` is not a power-of-two.
    ///
    /// # Examples
    ///
    /// Accessing adjacent `u8` as `u16`
    ///
    /// ```
    /// use std::ptr::NonNull;
    ///
    /// # unsafe {
    /// let x = [5_u8, 6, 7, 8, 9];
    /// let ptr = NonNull::new(x.as_ptr() as *mut u8).unwrap();
    /// let offset = ptr.align_offset(align_of::<u16>());
    ///
    /// if offset < x.len() - 1 {
    ///     let u16_ptr = ptr.add(offset).cast::<u16>();
    ///     assert!(u16_ptr.read() == u16::from_ne_bytes([5, 6]) || u16_ptr.read() == u16::from_ne_bytes([6, 7]));
    /// } else {
    ///     // while the pointer can be aligned via `offset`, it would point
    ///     // outside the allocation
    /// }
    /// # }
    /// ```
    #[inline]
    #[must_use]
    #[stable(feature = "non_null_convenience", since = "1.80.0")]
    pub fn align_offset(self, align: usize) -> usize
    where
        T: Sized,
    {
        if !align.is_power_of_two() {
            panic!("align_offset: align is not a power-of-two");
        }

        {
            // SAFETY: `align` has been checked to be a power of 2 above.
            unsafe { ptr::align_offset(self.as_ptr(), align) }
        }
    }

    /// Returns whether the pointer is properly aligned for `T`.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::ptr::NonNull;
    ///
    /// // On some platforms, the alignment of i32 is less than 4.
    /// #[repr(align(4))]
    /// struct AlignedI32(i32);
    ///
    /// let data = AlignedI32(42);
    /// let ptr = NonNull::<AlignedI32>::from(&data);
    ///
    /// assert!(ptr.is_aligned());
    /// assert!(!NonNull::new(ptr.as_ptr().wrapping_byte_add(1)).unwrap().is_aligned());
    /// ```
    #[inline]
    #[must_use]
    #[stable(feature = "pointer_is_aligned", since = "1.79.0")]
    pub fn is_aligned(self) -> bool
    where
        T: Sized,
    {
        self.as_ptr().is_aligned()
    }

    /// Returns whether the pointer is aligned to `align`.
    ///
    /// For non-`Sized` pointees this operation considers only the data pointer,
    /// ignoring the metadata.
    ///
    /// # Panics
    ///
    /// The function panics if `align` is not a power-of-two (this includes 0).
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(pointer_is_aligned_to)]
    ///
    /// // On some platforms, the alignment of i32 is less than 4.
    /// #[repr(align(4))]
    /// struct AlignedI32(i32);
    ///
    /// let data = AlignedI32(42);
    /// let ptr = &data as *const AlignedI32;
    ///
    /// assert!(ptr.is_aligned_to(1));
    /// assert!(ptr.is_aligned_to(2));
    /// assert!(ptr.is_aligned_to(4));
    ///
    /// assert!(ptr.wrapping_byte_add(2).is_aligned_to(2));
    /// assert!(!ptr.wrapping_byte_add(2).is_aligned_to(4));
    ///
    /// assert_ne!(ptr.is_aligned_to(8), ptr.wrapping_add(1).is_aligned_to(8));
    /// ```
    #[inline]
    #[must_use]
    #[unstable(feature = "pointer_is_aligned_to", issue = "96284")]
    pub fn is_aligned_to(self, align: usize) -> bool {
        self.as_ptr().is_aligned_to(align)
    }
}

impl<T> NonNull<[T]> {
    /// Creates a non-null raw slice from a thin pointer and a length.
    ///
    /// The `len` argument is the number of **elements**, not the number of bytes.
    ///
    /// This function is safe, but dereferencing the return value is unsafe.
    /// See the documentation of [`slice::from_raw_parts`] for slice safety requirements.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use std::ptr::NonNull;
    ///
    /// // create a slice pointer when starting out with a pointer to the first element
    /// let mut x = [5, 6, 7];
    /// let nonnull_pointer = NonNull::new(x.as_mut_ptr()).unwrap();
    /// let slice = NonNull::slice_from_raw_parts(nonnull_pointer, 3);
    /// assert_eq!(unsafe { slice.as_ref()[2] }, 7);
    /// ```
    ///
    /// (Note that this example artificially demonstrates a use of this method,
    /// but `let slice = NonNull::from(&x[..]);` would be a better way to write code like this.)
    #[stable(feature = "nonnull_slice_from_raw_parts", since = "1.70.0")]
    #[rustc_const_stable(feature = "const_slice_from_raw_parts_mut", since = "1.83.0")]
    #[must_use]
    #[inline]
    pub const fn slice_from_raw_parts(data: NonNull<T>, len: usize) -> Self {
        // SAFETY: `data` is a `NonNull` pointer which is necessarily non-null
        unsafe { Self::new_unchecked(super::slice_from_raw_parts_mut(data.as_ptr(), len)) }
    }

    /// Returns the length of a non-null raw slice.
    ///
    /// The returned value is the number of **elements**, not the number of bytes.
    ///
    /// This function is safe, even when the non-null raw slice cannot be dereferenced to a slice
    /// because the pointer does not have a valid address.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use std::ptr::NonNull;
    ///
    /// let slice: NonNull<[i8]> = NonNull::slice_from_raw_parts(NonNull::dangling(), 3);
    /// assert_eq!(slice.len(), 3);
    /// ```
    #[stable(feature = "slice_ptr_len_nonnull", since = "1.63.0")]
    #[rustc_const_stable(feature = "const_slice_ptr_len_nonnull", since = "1.63.0")]
    #[must_use]
    #[inline]
    pub const fn len(self) -> usize {
        self.as_ptr().len()
    }

    /// Returns `true` if the non-null raw slice has a length of 0.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use std::ptr::NonNull;
    ///
    /// let slice: NonNull<[i8]> = NonNull::slice_from_raw_parts(NonNull::dangling(), 3);
    /// assert!(!slice.is_empty());
    /// ```
    #[stable(feature = "slice_ptr_is_empty_nonnull", since = "1.79.0")]
    #[rustc_const_stable(feature = "const_slice_ptr_is_empty_nonnull", since = "1.79.0")]
    #[must_use]
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.len() == 0
    }

    /// Returns a non-null pointer to the slice's buffer.
    ///
    /// # Examples
    ///
    /// ```rust
    /// #![feature(slice_ptr_get)]
    /// use std::ptr::NonNull;
    ///
    /// let slice: NonNull<[i8]> = NonNull::slice_from_raw_parts(NonNull::dangling(), 3);
    /// assert_eq!(slice.as_non_null_ptr(), NonNull::<i8>::dangling());
    /// ```
    #[inline]
    #[must_use]
    #[unstable(feature = "slice_ptr_get", issue = "74265")]
    pub const fn as_non_null_ptr(self) -> NonNull<T> {
        self.cast()
    }

    /// Returns a raw pointer to the slice's buffer.
    ///
    /// # Examples
    ///
    /// ```rust
    /// #![feature(slice_ptr_get)]
    /// use std::ptr::NonNull;
    ///
    /// let slice: NonNull<[i8]> = NonNull::slice_from_raw_parts(NonNull::dangling(), 3);
    /// assert_eq!(slice.as_mut_ptr(), NonNull::<i8>::dangling().as_ptr());
    /// ```
    #[inline]
    #[must_use]
    #[unstable(feature = "slice_ptr_get", issue = "74265")]
    #[rustc_never_returns_null_ptr]
    pub const fn as_mut_ptr(self) -> *mut T {
        self.as_non_null_ptr().as_ptr()
    }

    /// Returns a shared reference to a slice of possibly uninitialized values. In contrast to
    /// [`as_ref`], this does not require that the value has to be initialized.
    ///
    /// For the mutable counterpart see [`as_uninit_slice_mut`].
    ///
    /// [`as_ref`]: NonNull::as_ref
    /// [`as_uninit_slice_mut`]: NonNull::as_uninit_slice_mut
    ///
    /// # Safety
    ///
    /// When calling this method, you have to ensure that all of the following is true:
    ///
    /// * The pointer must be [valid] for reads for `ptr.len() * size_of::<T>()` many bytes,
    ///   and it must be properly aligned. This means in particular:
    ///
    ///     * The entire memory range of this slice must be contained within a single allocation!
    ///       Slices can never span across multiple allocations.
    ///
    ///     * The pointer must be aligned even for zero-length slices. One
    ///       reason for this is that enum layout optimizations may rely on references
    ///       (including slices of any length) being aligned and non-null to distinguish
    ///       them from other data. You can obtain a pointer that is usable as `data`
    ///       for zero-length slices using [`NonNull::dangling()`].
    ///
    /// * The total size `ptr.len() * size_of::<T>()` of the slice must be no larger than `isize::MAX`.
    ///   See the safety documentation of [`pointer::offset`].
    ///
    /// * You must enforce Rust's aliasing rules, since the returned lifetime `'a` is
    ///   arbitrarily chosen and does not necessarily reflect the actual lifetime of the data.
    ///   In particular, while this reference exists, the memory the pointer points to must
    ///   not get mutated (except inside `UnsafeCell`).
    ///
    /// This applies even if the result of this method is unused!
    ///
    /// See also [`slice::from_raw_parts`].
    ///
    /// [valid]: crate::ptr#safety
    #[inline]
    #[must_use]
    #[unstable(feature = "ptr_as_uninit", issue = "75402")]
    pub const unsafe fn as_uninit_slice<'a>(self) -> &'a [MaybeUninit<T>] {
        // SAFETY: the caller must uphold the safety contract for `as_uninit_slice`.
        unsafe { slice::from_raw_parts(self.cast().as_ptr(), self.len()) }
    }

    /// Returns a unique reference to a slice of possibly uninitialized values. In contrast to
    /// [`as_mut`], this does not require that the value has to be initialized.
    ///
    /// For the shared counterpart see [`as_uninit_slice`].
    ///
    /// [`as_mut`]: NonNull::as_mut
    /// [`as_uninit_slice`]: NonNull::as_uninit_slice
    ///
    /// # Safety
    ///
    /// When calling this method, you have to ensure that all of the following is true:
    ///
    /// * The pointer must be [valid] for reads and writes for `ptr.len() * size_of::<T>()`
    ///   many bytes, and it must be properly aligned. This means in particular:
    ///
    ///     * The entire memory range of this slice must be contained within a single allocation!
    ///       Slices can never span across multiple allocations.
    ///
    ///     * The pointer must be aligned even for zero-length slices. One
    ///       reason for this is that enum layout optimizations may rely on references
    ///       (including slices of any length) being aligned and non-null to distinguish
    ///       them from other data. You can obtain a pointer that is usable as `data`
    ///       for zero-length slices using [`NonNull::dangling()`].
    ///
    /// * The total size `ptr.len() * size_of::<T>()` of the slice must be no larger than `isize::MAX`.
    ///   See the safety documentation of [`pointer::offset`].
    ///
    /// * You must enforce Rust's aliasing rules, since the returned lifetime `'a` is
    ///   arbitrarily chosen and does not necessarily reflect the actual lifetime of the data.
    ///   In particular, while this reference exists, the memory the pointer points to must
    ///   not get accessed (read or written) through any other pointer.
    ///
    /// This applies even if the result of this method is unused!
    ///
    /// See also [`slice::from_raw_parts_mut`].
    ///
    /// [valid]: crate::ptr#safety
    ///
    /// # Examples
    ///
    /// ```rust
    /// #![feature(allocator_api, ptr_as_uninit)]
    ///
    /// use std::alloc::{Allocator, Layout, Global};
    /// use std::mem::MaybeUninit;
    /// use std::ptr::NonNull;
    ///
    /// let memory: NonNull<[u8]> = Global.allocate(Layout::new::<[u8; 32]>())?;
    /// // This is safe as `memory` is valid for reads and writes for `memory.len()` many bytes.
    /// // Note that calling `memory.as_mut()` is not allowed here as the content may be uninitialized.
    /// # #[allow(unused_variables)]
    /// let slice: &mut [MaybeUninit<u8>] = unsafe { memory.as_uninit_slice_mut() };
    /// # // Prevent leaks for Miri.
    /// # unsafe { Global.deallocate(memory.cast(), Layout::new::<[u8; 32]>()); }
    /// # Ok::<_, std::alloc::AllocError>(())
    /// ```
    #[inline]
    #[must_use]
    #[unstable(feature = "ptr_as_uninit", issue = "75402")]
    pub const unsafe fn as_uninit_slice_mut<'a>(self) -> &'a mut [MaybeUninit<T>] {
        // SAFETY: the caller must uphold the safety contract for `as_uninit_slice_mut`.
        unsafe { slice::from_raw_parts_mut(self.cast().as_ptr(), self.len()) }
    }

    /// Returns a raw pointer to an element or subslice, without doing bounds
    /// checking.
    ///
    /// Calling this method with an out-of-bounds index or when `self` is not dereferenceable
    /// is *[undefined behavior]* even if the resulting pointer is not used.
    ///
    /// [undefined behavior]: https://doc.rust-lang.org/reference/behavior-considered-undefined.html
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(slice_ptr_get)]
    /// use std::ptr::NonNull;
    ///
    /// let x = &mut [1, 2, 4];
    /// let x = NonNull::slice_from_raw_parts(NonNull::new(x.as_mut_ptr()).unwrap(), x.len());
    ///
    /// unsafe {
    ///     assert_eq!(x.get_unchecked_mut(1).as_ptr(), x.as_non_null_ptr().as_ptr().add(1));
    /// }
    /// ```
    #[unstable(feature = "slice_ptr_get", issue = "74265")]
    #[inline]
    pub unsafe fn get_unchecked_mut<I>(self, index: I) -> NonNull<I::Output>
    where
        I: SliceIndex<[T]>,
    {
        // SAFETY: the caller ensures that `self` is dereferenceable and `index` in-bounds.
        // As a consequence, the resulting pointer cannot be null.
        unsafe { NonNull::new_unchecked(self.as_ptr().get_unchecked_mut(index)) }
    }
}

#[stable(feature = "nonnull", since = "1.25.0")]
impl<T: PointeeSized> Clone for NonNull<T> {
    #[inline(always)]
    fn clone(&self) -> Self {
        *self
    }
}

#[stable(feature = "nonnull", since = "1.25.0")]
impl<T: PointeeSized> Copy for NonNull<T> {}

#[unstable(feature = "coerce_unsized", issue = "18598")]
impl<T: PointeeSized, U: PointeeSized> CoerceUnsized<NonNull<U>> for NonNull<T> where T: Unsize<U> {}

#[unstable(feature = "dispatch_from_dyn", issue = "none")]
impl<T: PointeeSized, U: PointeeSized> DispatchFromDyn<NonNull<U>> for NonNull<T> where T: Unsize<U> {}

#[stable(feature = "pin", since = "1.33.0")]
unsafe impl<T: PointeeSized> PinCoerceUnsized for NonNull<T> {}

#[unstable(feature = "pointer_like_trait", issue = "none")]
impl<T> core::marker::PointerLike for NonNull<T> {}

#[stable(feature = "nonnull", since = "1.25.0")]
impl<T: PointeeSized> fmt::Debug for NonNull<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Pointer::fmt(&self.as_ptr(), f)
    }
}

#[stable(feature = "nonnull", since = "1.25.0")]
impl<T: PointeeSized> fmt::Pointer for NonNull<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Pointer::fmt(&self.as_ptr(), f)
    }
}

#[stable(feature = "nonnull", since = "1.25.0")]
impl<T: PointeeSized> Eq for NonNull<T> {}

#[stable(feature = "nonnull", since = "1.25.0")]
impl<T: PointeeSized> PartialEq for NonNull<T> {
    #[inline]
    #[allow(ambiguous_wide_pointer_comparisons)]
    fn eq(&self, other: &Self) -> bool {
        self.as_ptr() == other.as_ptr()
    }
}

#[stable(feature = "nonnull", since = "1.25.0")]
impl<T: PointeeSized> Ord for NonNull<T> {
    #[inline]
    #[allow(ambiguous_wide_pointer_comparisons)]
    fn cmp(&self, other: &Self) -> Ordering {
        self.as_ptr().cmp(&other.as_ptr())
    }
}

#[stable(feature = "nonnull", since = "1.25.0")]
impl<T: PointeeSized> PartialOrd for NonNull<T> {
    #[inline]
    #[allow(ambiguous_wide_pointer_comparisons)]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.as_ptr().partial_cmp(&other.as_ptr())
    }
}

#[stable(feature = "nonnull", since = "1.25.0")]
impl<T: PointeeSized> hash::Hash for NonNull<T> {
    #[inline]
    fn hash<H: hash::Hasher>(&self, state: &mut H) {
        self.as_ptr().hash(state)
    }
}

#[unstable(feature = "ptr_internals", issue = "none")]
impl<T: PointeeSized> From<Unique<T>> for NonNull<T> {
    #[inline]
    fn from(unique: Unique<T>) -> Self {
        unique.as_non_null_ptr()
    }
}

#[stable(feature = "nonnull", since = "1.25.0")]
impl<T: PointeeSized> From<&mut T> for NonNull<T> {
    /// Converts a `&mut T` to a `NonNull<T>`.
    ///
    /// This conversion is safe and infallible since references cannot be null.
    #[inline]
    fn from(r: &mut T) -> Self {
        NonNull::from_mut(r)
    }
}

#[stable(feature = "nonnull", since = "1.25.0")]
impl<T: PointeeSized> From<&T> for NonNull<T> {
    /// Converts a `&T` to a `NonNull<T>`.
    ///
    /// This conversion is safe and infallible since references cannot be null.
    #[inline]
    fn from(r: &T) -> Self {
        NonNull::from_ref(r)
    }
}
