use super::*;
use crate::cmp::Ordering::{self, Equal, Greater, Less};
use crate::intrinsics;

#[lang = "mut_ptr"]
impl<T: ?Sized> *mut T {
    /// Returns `true` if the pointer is null.
    ///
    /// Note that unsized types have many possible null pointers, as only the
    /// raw data pointer is considered, not their length, vtable, etc.
    /// Therefore, two pointers that are null may still not compare equal to
    /// each other.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// let mut s = [1, 2, 3];
    /// let ptr: *mut u32 = s.as_mut_ptr();
    /// assert!(!ptr.is_null());
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn is_null(self) -> bool {
        // Compare via a cast to a thin pointer, so fat pointers are only
        // considering their "data" part for null-ness.
        (self as *mut u8) == null_mut()
    }

    /// Casts to a pointer of another type.
    #[stable(feature = "ptr_cast", since = "1.38.0")]
    #[rustc_const_stable(feature = "const_ptr_cast", since = "1.38.0")]
    #[inline]
    pub const fn cast<U>(self) -> *mut U {
        self as _
    }

    /// Returns `None` if the pointer is null, or else returns a reference to
    /// the value wrapped in `Some`.
    ///
    /// # Safety
    ///
    /// While this method and its mutable counterpart are useful for
    /// null-safety, it is important to note that this is still an unsafe
    /// operation because the returned value could be pointing to invalid
    /// memory.
    ///
    /// When calling this method, you have to ensure that if the pointer is
    /// non-NULL, then it is properly aligned, dereferenceable (for the whole
    /// size of `T`) and points to an initialized instance of `T`. This applies
    /// even if the result of this method is unused!
    /// (The part about being initialized is not yet fully decided, but until
    /// it is, the only safe approach is to ensure that they are indeed initialized.)
    ///
    /// Additionally, the lifetime `'a` returned is arbitrarily chosen and does
    /// not necessarily reflect the actual lifetime of the data. It is up to the
    /// caller to ensure that for the duration of this lifetime, the memory this
    /// pointer points to does not get written to outside of `UnsafeCell<U>`.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// let ptr: *mut u8 = &mut 10u8 as *mut u8;
    ///
    /// unsafe {
    ///     if let Some(val_back) = ptr.as_ref() {
    ///         println!("We got back the value: {}!", val_back);
    ///     }
    /// }
    /// ```
    ///
    /// # Null-unchecked version
    ///
    /// If you are sure the pointer can never be null and are looking for some kind of
    /// `as_ref_unchecked` that returns the `&T` instead of `Option<&T>`, know that you can
    /// dereference the pointer directly.
    ///
    /// ```
    /// let ptr: *mut u8 = &mut 10u8 as *mut u8;
    ///
    /// unsafe {
    ///     let val_back = &*ptr;
    ///     println!("We got back the value: {}!", val_back);
    /// }
    /// ```
    #[stable(feature = "ptr_as_ref", since = "1.9.0")]
    #[inline]
    pub unsafe fn as_ref<'a>(self) -> Option<&'a T> {
        if self.is_null() { None } else { Some(&*self) }
    }

    /// Calculates the offset from a pointer.
    ///
    /// `count` is in units of T; e.g., a `count` of 3 represents a pointer
    /// offset of `3 * size_of::<T>()` bytes.
    ///
    /// # Safety
    ///
    /// If any of the following conditions are violated, the result is Undefined
    /// Behavior:
    ///
    /// * Both the starting and resulting pointer must be either in bounds or one
    ///   byte past the end of the same allocated object. Note that in Rust,
    ///   every (stack-allocated) variable is considered a separate allocated object.
    ///
    /// * The computed offset, **in bytes**, cannot overflow an `isize`.
    ///
    /// * The offset being in bounds cannot rely on "wrapping around" the address
    ///   space. That is, the infinite-precision sum, **in bytes** must fit in a usize.
    ///
    /// The compiler and standard library generally tries to ensure allocations
    /// never reach a size where an offset is a concern. For instance, `Vec`
    /// and `Box` ensure they never allocate more than `isize::MAX` bytes, so
    /// `vec.as_ptr().add(vec.len())` is always safe.
    ///
    /// Most platforms fundamentally can't even construct such an allocation.
    /// For instance, no known 64-bit platform can ever serve a request
    /// for 2<sup>63</sup> bytes due to page-table limitations or splitting the address space.
    /// However, some 32-bit and 16-bit platforms may successfully serve a request for
    /// more than `isize::MAX` bytes with things like Physical Address
    /// Extension. As such, memory acquired directly from allocators or memory
    /// mapped files *may* be too large to handle with this function.
    ///
    /// Consider using [`wrapping_offset`] instead if these constraints are
    /// difficult to satisfy. The only advantage of this method is that it
    /// enables more aggressive compiler optimizations.
    ///
    /// [`wrapping_offset`]: #method.wrapping_offset
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// let mut s = [1, 2, 3];
    /// let ptr: *mut u32 = s.as_mut_ptr();
    ///
    /// unsafe {
    ///     println!("{}", *ptr.offset(1));
    ///     println!("{}", *ptr.offset(2));
    /// }
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[must_use = "returns a new pointer rather than modifying its argument"]
    #[inline]
    pub unsafe fn offset(self, count: isize) -> *mut T
    where
        T: Sized,
    {
        intrinsics::offset(self, count) as *mut T
    }

    /// Calculates the offset from a pointer using wrapping arithmetic.
    /// `count` is in units of T; e.g., a `count` of 3 represents a pointer
    /// offset of `3 * size_of::<T>()` bytes.
    ///
    /// # Safety
    ///
    /// The resulting pointer does not need to be in bounds, but it is
    /// potentially hazardous to dereference (which requires `unsafe`).
    ///
    /// In particular, the resulting pointer remains attached to the same allocated
    /// object that `self` points to. It may *not* be used to access a
    /// different allocated object. Note that in Rust,
    /// every (stack-allocated) variable is considered a separate allocated object.
    ///
    /// In other words, `x.wrapping_offset(y.wrapping_offset_from(x))` is
    /// *not* the same as `y`, and dereferencing it is undefined behavior
    /// unless `x` and `y` point into the same allocated object.
    ///
    /// Compared to [`offset`], this method basically delays the requirement of staying
    /// within the same allocated object: [`offset`] is immediate Undefined Behavior when
    /// crossing object boundaries; `wrapping_offset` produces a pointer but still leads
    /// to Undefined Behavior if that pointer is dereferenced. [`offset`] can be optimized
    /// better and is thus preferable in performance-sensitive code.
    ///
    /// If you need to cross object boundaries, cast the pointer to an integer and
    /// do the arithmetic there.
    ///
    /// [`offset`]: #method.offset
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// // Iterate using a raw pointer in increments of two elements
    /// let mut data = [1u8, 2, 3, 4, 5];
    /// let mut ptr: *mut u8 = data.as_mut_ptr();
    /// let step = 2;
    /// let end_rounded_up = ptr.wrapping_offset(6);
    ///
    /// while ptr != end_rounded_up {
    ///     unsafe {
    ///         *ptr = 0;
    ///     }
    ///     ptr = ptr.wrapping_offset(step);
    /// }
    /// assert_eq!(&data, &[0, 2, 0, 4, 0]);
    /// ```
    #[stable(feature = "ptr_wrapping_offset", since = "1.16.0")]
    #[must_use = "returns a new pointer rather than modifying its argument"]
    #[inline]
    pub fn wrapping_offset(self, count: isize) -> *mut T
    where
        T: Sized,
    {
        // SAFETY: the `arith_offset` intrinsic has no prerequisites to be called.
        unsafe { intrinsics::arith_offset(self, count) as *mut T }
    }

    /// Returns `None` if the pointer is null, or else returns a mutable
    /// reference to the value wrapped in `Some`.
    ///
    /// # Safety
    ///
    /// As with [`as_ref`], this is unsafe because it cannot verify the validity
    /// of the returned pointer, nor can it ensure that the lifetime `'a`
    /// returned is indeed a valid lifetime for the contained data.
    ///
    /// When calling this method, you have to ensure that *either* the pointer is NULL *or*
    /// all of the following is true:
    /// - it is properly aligned
    /// - it must point to an initialized instance of T; in particular, the pointer must be
    ///   "dereferenceable" in the sense defined [here].
    ///
    /// This applies even if the result of this method is unused!
    /// (The part about being initialized is not yet fully decided, but until
    /// it is the only safe approach is to ensure that they are indeed initialized.)
    ///
    /// Additionally, the lifetime `'a` returned is arbitrarily chosen and does
    /// not necessarily reflect the actual lifetime of the data. *You* must enforce
    /// Rust's aliasing rules. In particular, for the duration of this lifetime,
    /// the memory this pointer points to must not get accessed (read or written)
    /// through any other pointer.
    ///
    /// [here]: crate::ptr#safety
    /// [`as_ref`]: #method.as_ref
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// let mut s = [1, 2, 3];
    /// let ptr: *mut u32 = s.as_mut_ptr();
    /// let first_value = unsafe { ptr.as_mut().unwrap() };
    /// *first_value = 4;
    /// println!("{:?}", s); // It'll print: "[4, 2, 3]".
    /// ```
    ///
    /// # Null-unchecked version
    ///
    /// If you are sure the pointer can never be null and are looking for some kind of
    /// `as_mut_unchecked` that returns the `&mut T` instead of `Option<&mut T>`, know that
    /// you can dereference the pointer directly.
    ///
    /// ```
    /// let mut s = [1, 2, 3];
    /// let ptr: *mut u32 = s.as_mut_ptr();
    /// let first_value = unsafe { &mut *ptr };
    /// *first_value = 4;
    /// println!("{:?}", s); // It'll print: "[4, 2, 3]".
    /// ```
    #[stable(feature = "ptr_as_ref", since = "1.9.0")]
    #[inline]
    pub unsafe fn as_mut<'a>(self) -> Option<&'a mut T> {
        if self.is_null() { None } else { Some(&mut *self) }
    }

    /// Calculates the distance between two pointers. The returned value is in
    /// units of T: the distance in bytes is divided by `mem::size_of::<T>()`.
    ///
    /// This function is the inverse of [`offset`].
    ///
    /// [`offset`]: #method.offset-1
    /// [`wrapping_offset_from`]: #method.wrapping_offset_from-1
    ///
    /// # Safety
    ///
    /// If any of the following conditions are violated, the result is Undefined
    /// Behavior:
    ///
    /// * Both the starting and other pointer must be either in bounds or one
    ///   byte past the end of the same allocated object. Note that in Rust,
    ///   every (stack-allocated) variable is considered a separate allocated object.
    ///
    /// * The distance between the pointers, **in bytes**, cannot overflow an `isize`.
    ///
    /// * The distance between the pointers, in bytes, must be an exact multiple
    ///   of the size of `T`.
    ///
    /// * The distance being in bounds cannot rely on "wrapping around" the address space.
    ///
    /// The compiler and standard library generally try to ensure allocations
    /// never reach a size where an offset is a concern. For instance, `Vec`
    /// and `Box` ensure they never allocate more than `isize::MAX` bytes, so
    /// `ptr_into_vec.offset_from(vec.as_ptr())` is always safe.
    ///
    /// Most platforms fundamentally can't even construct such an allocation.
    /// For instance, no known 64-bit platform can ever serve a request
    /// for 2<sup>63</sup> bytes due to page-table limitations or splitting the address space.
    /// However, some 32-bit and 16-bit platforms may successfully serve a request for
    /// more than `isize::MAX` bytes with things like Physical Address
    /// Extension. As such, memory acquired directly from allocators or memory
    /// mapped files *may* be too large to handle with this function.
    ///
    /// Consider using [`wrapping_offset_from`] instead if these constraints are
    /// difficult to satisfy. The only advantage of this method is that it
    /// enables more aggressive compiler optimizations.
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
    /// #![feature(ptr_offset_from)]
    ///
    /// let mut a = [0; 5];
    /// let ptr1: *mut i32 = &mut a[1];
    /// let ptr2: *mut i32 = &mut a[3];
    /// unsafe {
    ///     assert_eq!(ptr2.offset_from(ptr1), 2);
    ///     assert_eq!(ptr1.offset_from(ptr2), -2);
    ///     assert_eq!(ptr1.offset(2), ptr2);
    ///     assert_eq!(ptr2.offset(-2), ptr1);
    /// }
    /// ```
    #[unstable(feature = "ptr_offset_from", issue = "41079")]
    #[rustc_const_unstable(feature = "const_ptr_offset_from", issue = "41079")]
    #[inline]
    pub const unsafe fn offset_from(self, origin: *const T) -> isize
    where
        T: Sized,
    {
        (self as *const T).offset_from(origin)
    }

    /// Calculates the distance between two pointers. The returned value is in
    /// units of T: the distance in bytes is divided by `mem::size_of::<T>()`.
    ///
    /// If the address different between the two pointers is not a multiple of
    /// `mem::size_of::<T>()` then the result of the division is rounded towards
    /// zero.
    ///
    /// Though this method is safe for any two pointers, note that its result
    /// will be mostly useless if the two pointers aren't into the same allocated
    /// object, for example if they point to two different local variables.
    ///
    /// # Panics
    ///
    /// This function panics if `T` is a zero-sized type.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// #![feature(ptr_wrapping_offset_from)]
    ///
    /// let mut a = [0; 5];
    /// let ptr1: *mut i32 = &mut a[1];
    /// let ptr2: *mut i32 = &mut a[3];
    /// assert_eq!(ptr2.wrapping_offset_from(ptr1), 2);
    /// assert_eq!(ptr1.wrapping_offset_from(ptr2), -2);
    /// assert_eq!(ptr1.wrapping_offset(2), ptr2);
    /// assert_eq!(ptr2.wrapping_offset(-2), ptr1);
    ///
    /// let ptr1: *mut i32 = 3 as _;
    /// let ptr2: *mut i32 = 13 as _;
    /// assert_eq!(ptr2.wrapping_offset_from(ptr1), 2);
    /// ```
    #[unstable(feature = "ptr_wrapping_offset_from", issue = "41079")]
    #[inline]
    pub fn wrapping_offset_from(self, origin: *const T) -> isize
    where
        T: Sized,
    {
        (self as *const T).wrapping_offset_from(origin)
    }

    /// Calculates the offset from a pointer (convenience for `.offset(count as isize)`).
    ///
    /// `count` is in units of T; e.g., a `count` of 3 represents a pointer
    /// offset of `3 * size_of::<T>()` bytes.
    ///
    /// # Safety
    ///
    /// If any of the following conditions are violated, the result is Undefined
    /// Behavior:
    ///
    /// * Both the starting and resulting pointer must be either in bounds or one
    ///   byte past the end of the same allocated object. Note that in Rust,
    ///   every (stack-allocated) variable is considered a separate allocated object.
    ///
    /// * The computed offset, **in bytes**, cannot overflow an `isize`.
    ///
    /// * The offset being in bounds cannot rely on "wrapping around" the address
    ///   space. That is, the infinite-precision sum must fit in a `usize`.
    ///
    /// The compiler and standard library generally tries to ensure allocations
    /// never reach a size where an offset is a concern. For instance, `Vec`
    /// and `Box` ensure they never allocate more than `isize::MAX` bytes, so
    /// `vec.as_ptr().add(vec.len())` is always safe.
    ///
    /// Most platforms fundamentally can't even construct such an allocation.
    /// For instance, no known 64-bit platform can ever serve a request
    /// for 2<sup>63</sup> bytes due to page-table limitations or splitting the address space.
    /// However, some 32-bit and 16-bit platforms may successfully serve a request for
    /// more than `isize::MAX` bytes with things like Physical Address
    /// Extension. As such, memory acquired directly from allocators or memory
    /// mapped files *may* be too large to handle with this function.
    ///
    /// Consider using [`wrapping_add`] instead if these constraints are
    /// difficult to satisfy. The only advantage of this method is that it
    /// enables more aggressive compiler optimizations.
    ///
    /// [`wrapping_add`]: #method.wrapping_add
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// let s: &str = "123";
    /// let ptr: *const u8 = s.as_ptr();
    ///
    /// unsafe {
    ///     println!("{}", *ptr.add(1) as char);
    ///     println!("{}", *ptr.add(2) as char);
    /// }
    /// ```
    #[stable(feature = "pointer_methods", since = "1.26.0")]
    #[must_use = "returns a new pointer rather than modifying its argument"]
    #[inline]
    pub unsafe fn add(self, count: usize) -> Self
    where
        T: Sized,
    {
        self.offset(count as isize)
    }

    /// Calculates the offset from a pointer (convenience for
    /// `.offset((count as isize).wrapping_neg())`).
    ///
    /// `count` is in units of T; e.g., a `count` of 3 represents a pointer
    /// offset of `3 * size_of::<T>()` bytes.
    ///
    /// # Safety
    ///
    /// If any of the following conditions are violated, the result is Undefined
    /// Behavior:
    ///
    /// * Both the starting and resulting pointer must be either in bounds or one
    ///   byte past the end of the same allocated object. Note that in Rust,
    ///   every (stack-allocated) variable is considered a separate allocated object.
    ///
    /// * The computed offset cannot exceed `isize::MAX` **bytes**.
    ///
    /// * The offset being in bounds cannot rely on "wrapping around" the address
    ///   space. That is, the infinite-precision sum must fit in a usize.
    ///
    /// The compiler and standard library generally tries to ensure allocations
    /// never reach a size where an offset is a concern. For instance, `Vec`
    /// and `Box` ensure they never allocate more than `isize::MAX` bytes, so
    /// `vec.as_ptr().add(vec.len()).sub(vec.len())` is always safe.
    ///
    /// Most platforms fundamentally can't even construct such an allocation.
    /// For instance, no known 64-bit platform can ever serve a request
    /// for 2<sup>63</sup> bytes due to page-table limitations or splitting the address space.
    /// However, some 32-bit and 16-bit platforms may successfully serve a request for
    /// more than `isize::MAX` bytes with things like Physical Address
    /// Extension. As such, memory acquired directly from allocators or memory
    /// mapped files *may* be too large to handle with this function.
    ///
    /// Consider using [`wrapping_sub`] instead if these constraints are
    /// difficult to satisfy. The only advantage of this method is that it
    /// enables more aggressive compiler optimizations.
    ///
    /// [`wrapping_sub`]: #method.wrapping_sub
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// let s: &str = "123";
    ///
    /// unsafe {
    ///     let end: *const u8 = s.as_ptr().add(3);
    ///     println!("{}", *end.sub(1) as char);
    ///     println!("{}", *end.sub(2) as char);
    /// }
    /// ```
    #[stable(feature = "pointer_methods", since = "1.26.0")]
    #[must_use = "returns a new pointer rather than modifying its argument"]
    #[inline]
    pub unsafe fn sub(self, count: usize) -> Self
    where
        T: Sized,
    {
        self.offset((count as isize).wrapping_neg())
    }

    /// Calculates the offset from a pointer using wrapping arithmetic.
    /// (convenience for `.wrapping_offset(count as isize)`)
    ///
    /// `count` is in units of T; e.g., a `count` of 3 represents a pointer
    /// offset of `3 * size_of::<T>()` bytes.
    ///
    /// # Safety
    ///
    /// The resulting pointer does not need to be in bounds, but it is
    /// potentially hazardous to dereference (which requires `unsafe`).
    ///
    /// In particular, the resulting pointer remains attached to the same allocated
    /// object that `self` points to. It may *not* be used to access a
    /// different allocated object. Note that in Rust,
    /// every (stack-allocated) variable is considered a separate allocated object.
    ///
    /// Compared to [`add`], this method basically delays the requirement of staying
    /// within the same allocated object: [`add`] is immediate Undefined Behavior when
    /// crossing object boundaries; `wrapping_add` produces a pointer but still leads
    /// to Undefined Behavior if that pointer is dereferenced. [`add`] can be optimized
    /// better and is thus preferable in performance-sensitive code.
    ///
    /// If you need to cross object boundaries, cast the pointer to an integer and
    /// do the arithmetic there.
    ///
    /// [`add`]: #method.add
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// // Iterate using a raw pointer in increments of two elements
    /// let data = [1u8, 2, 3, 4, 5];
    /// let mut ptr: *const u8 = data.as_ptr();
    /// let step = 2;
    /// let end_rounded_up = ptr.wrapping_add(6);
    ///
    /// // This loop prints "1, 3, 5, "
    /// while ptr != end_rounded_up {
    ///     unsafe {
    ///         print!("{}, ", *ptr);
    ///     }
    ///     ptr = ptr.wrapping_add(step);
    /// }
    /// ```
    #[stable(feature = "pointer_methods", since = "1.26.0")]
    #[must_use = "returns a new pointer rather than modifying its argument"]
    #[inline]
    pub fn wrapping_add(self, count: usize) -> Self
    where
        T: Sized,
    {
        self.wrapping_offset(count as isize)
    }

    /// Calculates the offset from a pointer using wrapping arithmetic.
    /// (convenience for `.wrapping_offset((count as isize).wrapping_sub())`)
    ///
    /// `count` is in units of T; e.g., a `count` of 3 represents a pointer
    /// offset of `3 * size_of::<T>()` bytes.
    ///
    /// # Safety
    ///
    /// The resulting pointer does not need to be in bounds, but it is
    /// potentially hazardous to dereference (which requires `unsafe`).
    ///
    /// In particular, the resulting pointer remains attached to the same allocated
    /// object that `self` points to. It may *not* be used to access a
    /// different allocated object. Note that in Rust,
    /// every (stack-allocated) variable is considered a separate allocated object.
    ///
    /// Compared to [`sub`], this method basically delays the requirement of staying
    /// within the same allocated object: [`sub`] is immediate Undefined Behavior when
    /// crossing object boundaries; `wrapping_sub` produces a pointer but still leads
    /// to Undefined Behavior if that pointer is dereferenced. [`sub`] can be optimized
    /// better and is thus preferable in performance-sensitive code.
    ///
    /// If you need to cross object boundaries, cast the pointer to an integer and
    /// do the arithmetic there.
    ///
    /// [`sub`]: #method.sub
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// // Iterate using a raw pointer in increments of two elements (backwards)
    /// let data = [1u8, 2, 3, 4, 5];
    /// let mut ptr: *const u8 = data.as_ptr();
    /// let start_rounded_down = ptr.wrapping_sub(2);
    /// ptr = ptr.wrapping_add(4);
    /// let step = 2;
    /// // This loop prints "5, 3, 1, "
    /// while ptr != start_rounded_down {
    ///     unsafe {
    ///         print!("{}, ", *ptr);
    ///     }
    ///     ptr = ptr.wrapping_sub(step);
    /// }
    /// ```
    #[stable(feature = "pointer_methods", since = "1.26.0")]
    #[must_use = "returns a new pointer rather than modifying its argument"]
    #[inline]
    pub fn wrapping_sub(self, count: usize) -> Self
    where
        T: Sized,
    {
        self.wrapping_offset((count as isize).wrapping_neg())
    }

    /// Reads the value from `self` without moving it. This leaves the
    /// memory in `self` unchanged.
    ///
    /// See [`ptr::read`] for safety concerns and examples.
    ///
    /// [`ptr::read`]: ./ptr/fn.read.html
    #[stable(feature = "pointer_methods", since = "1.26.0")]
    #[inline]
    pub unsafe fn read(self) -> T
    where
        T: Sized,
    {
        read(self)
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
    /// [`ptr::read_volatile`]: ./ptr/fn.read_volatile.html
    #[stable(feature = "pointer_methods", since = "1.26.0")]
    #[inline]
    pub unsafe fn read_volatile(self) -> T
    where
        T: Sized,
    {
        read_volatile(self)
    }

    /// Reads the value from `self` without moving it. This leaves the
    /// memory in `self` unchanged.
    ///
    /// Unlike `read`, the pointer may be unaligned.
    ///
    /// See [`ptr::read_unaligned`] for safety concerns and examples.
    ///
    /// [`ptr::read_unaligned`]: ./ptr/fn.read_unaligned.html
    #[stable(feature = "pointer_methods", since = "1.26.0")]
    #[inline]
    pub unsafe fn read_unaligned(self) -> T
    where
        T: Sized,
    {
        read_unaligned(self)
    }

    /// Copies `count * size_of<T>` bytes from `self` to `dest`. The source
    /// and destination may overlap.
    ///
    /// NOTE: this has the *same* argument order as [`ptr::copy`].
    ///
    /// See [`ptr::copy`] for safety concerns and examples.
    ///
    /// [`ptr::copy`]: ./ptr/fn.copy.html
    #[stable(feature = "pointer_methods", since = "1.26.0")]
    #[inline]
    pub unsafe fn copy_to(self, dest: *mut T, count: usize)
    where
        T: Sized,
    {
        copy(self, dest, count)
    }

    /// Copies `count * size_of<T>` bytes from `self` to `dest`. The source
    /// and destination may *not* overlap.
    ///
    /// NOTE: this has the *same* argument order as [`ptr::copy_nonoverlapping`].
    ///
    /// See [`ptr::copy_nonoverlapping`] for safety concerns and examples.
    ///
    /// [`ptr::copy_nonoverlapping`]: ./ptr/fn.copy_nonoverlapping.html
    #[stable(feature = "pointer_methods", since = "1.26.0")]
    #[inline]
    pub unsafe fn copy_to_nonoverlapping(self, dest: *mut T, count: usize)
    where
        T: Sized,
    {
        copy_nonoverlapping(self, dest, count)
    }

    /// Copies `count * size_of<T>` bytes from `src` to `self`. The source
    /// and destination may overlap.
    ///
    /// NOTE: this has the *opposite* argument order of [`ptr::copy`].
    ///
    /// See [`ptr::copy`] for safety concerns and examples.
    ///
    /// [`ptr::copy`]: ./ptr/fn.copy.html
    #[stable(feature = "pointer_methods", since = "1.26.0")]
    #[inline]
    pub unsafe fn copy_from(self, src: *const T, count: usize)
    where
        T: Sized,
    {
        copy(src, self, count)
    }

    /// Copies `count * size_of<T>` bytes from `src` to `self`. The source
    /// and destination may *not* overlap.
    ///
    /// NOTE: this has the *opposite* argument order of [`ptr::copy_nonoverlapping`].
    ///
    /// See [`ptr::copy_nonoverlapping`] for safety concerns and examples.
    ///
    /// [`ptr::copy_nonoverlapping`]: ./ptr/fn.copy_nonoverlapping.html
    #[stable(feature = "pointer_methods", since = "1.26.0")]
    #[inline]
    pub unsafe fn copy_from_nonoverlapping(self, src: *const T, count: usize)
    where
        T: Sized,
    {
        copy_nonoverlapping(src, self, count)
    }

    /// Executes the destructor (if any) of the pointed-to value.
    ///
    /// See [`ptr::drop_in_place`] for safety concerns and examples.
    ///
    /// [`ptr::drop_in_place`]: ./ptr/fn.drop_in_place.html
    #[stable(feature = "pointer_methods", since = "1.26.0")]
    #[inline]
    pub unsafe fn drop_in_place(self) {
        drop_in_place(self)
    }

    /// Overwrites a memory location with the given value without reading or
    /// dropping the old value.
    ///
    /// See [`ptr::write`] for safety concerns and examples.
    ///
    /// [`ptr::write`]: ./ptr/fn.write.html
    #[stable(feature = "pointer_methods", since = "1.26.0")]
    #[inline]
    pub unsafe fn write(self, val: T)
    where
        T: Sized,
    {
        write(self, val)
    }

    /// Invokes memset on the specified pointer, setting `count * size_of::<T>()`
    /// bytes of memory starting at `self` to `val`.
    ///
    /// See [`ptr::write_bytes`] for safety concerns and examples.
    ///
    /// [`ptr::write_bytes`]: ./ptr/fn.write_bytes.html
    #[stable(feature = "pointer_methods", since = "1.26.0")]
    #[inline]
    pub unsafe fn write_bytes(self, val: u8, count: usize)
    where
        T: Sized,
    {
        write_bytes(self, val, count)
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
    /// [`ptr::write_volatile`]: ./ptr/fn.write_volatile.html
    #[stable(feature = "pointer_methods", since = "1.26.0")]
    #[inline]
    pub unsafe fn write_volatile(self, val: T)
    where
        T: Sized,
    {
        write_volatile(self, val)
    }

    /// Overwrites a memory location with the given value without reading or
    /// dropping the old value.
    ///
    /// Unlike `write`, the pointer may be unaligned.
    ///
    /// See [`ptr::write_unaligned`] for safety concerns and examples.
    ///
    /// [`ptr::write_unaligned`]: ./ptr/fn.write_unaligned.html
    #[stable(feature = "pointer_methods", since = "1.26.0")]
    #[inline]
    pub unsafe fn write_unaligned(self, val: T)
    where
        T: Sized,
    {
        write_unaligned(self, val)
    }

    /// Replaces the value at `self` with `src`, returning the old
    /// value, without dropping either.
    ///
    /// See [`ptr::replace`] for safety concerns and examples.
    ///
    /// [`ptr::replace`]: ./ptr/fn.replace.html
    #[stable(feature = "pointer_methods", since = "1.26.0")]
    #[inline]
    pub unsafe fn replace(self, src: T) -> T
    where
        T: Sized,
    {
        replace(self, src)
    }

    /// Swaps the values at two mutable locations of the same type, without
    /// deinitializing either. They may overlap, unlike `mem::swap` which is
    /// otherwise equivalent.
    ///
    /// See [`ptr::swap`] for safety concerns and examples.
    ///
    /// [`ptr::swap`]: ./ptr/fn.swap.html
    #[stable(feature = "pointer_methods", since = "1.26.0")]
    #[inline]
    pub unsafe fn swap(self, with: *mut T)
    where
        T: Sized,
    {
        swap(self, with)
    }

    /// Computes the offset that needs to be applied to the pointer in order to make it aligned to
    /// `align`.
    ///
    /// If it is not possible to align the pointer, the implementation returns
    /// `usize::MAX`. It is permissible for the implementation to *always*
    /// return `usize::MAX`. Only your algorithm's performance can depend
    /// on getting a usable offset here, not its correctness.
    ///
    /// The offset is expressed in number of `T` elements, and not bytes. The value returned can be
    /// used with the `wrapping_add` method.
    ///
    /// There are no guarantees whatsoever that offsetting the pointer will not overflow or go
    /// beyond the allocation that the pointer points into. It is up to the caller to ensure that
    /// the returned offset is correct in all terms other than alignment.
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
    /// # fn foo(n: usize) {
    /// # use std::mem::align_of;
    /// # unsafe {
    /// let x = [5u8, 6u8, 7u8, 8u8, 9u8];
    /// let ptr = &x[n] as *const u8;
    /// let offset = ptr.align_offset(align_of::<u16>());
    /// if offset < x.len() - n - 1 {
    ///     let u16_ptr = ptr.add(offset) as *const u16;
    ///     assert_ne!(*u16_ptr, 500);
    /// } else {
    ///     // while the pointer can be aligned via `offset`, it would point
    ///     // outside the allocation
    /// }
    /// # } }
    /// ```
    #[stable(feature = "align_offset", since = "1.36.0")]
    pub fn align_offset(self, align: usize) -> usize
    where
        T: Sized,
    {
        if !align.is_power_of_two() {
            panic!("align_offset: align is not a power-of-two");
        }
        // SAFETY: `align` has been checked to be a power of 2 above
        unsafe { align_offset(self, align) }
    }
}

#[lang = "mut_slice_ptr"]
impl<T> *mut [T] {
    /// Returns the length of a raw slice.
    ///
    /// The returned value is the number of **elements**, not the number of bytes.
    ///
    /// This function is safe, even when the raw slice cannot be cast to a slice
    /// reference because the pointer is null or unaligned.
    ///
    /// # Examples
    ///
    /// ```rust
    /// #![feature(slice_ptr_len)]
    ///
    /// use std::ptr;
    ///
    /// let slice: *mut [i8] = ptr::slice_from_raw_parts_mut(ptr::null_mut(), 3);
    /// assert_eq!(slice.len(), 3);
    /// ```
    #[inline]
    #[unstable(feature = "slice_ptr_len", issue = "71146")]
    #[rustc_const_unstable(feature = "const_slice_ptr_len", issue = "71146")]
    pub const fn len(self) -> usize {
        // SAFETY: this is safe because `*const [T]` and `FatPtr<T>` have the same layout.
        // Only `std` can make this guarantee.
        unsafe { Repr { rust_mut: self }.raw }.len
    }
}

// Equality for pointers
#[stable(feature = "rust1", since = "1.0.0")]
impl<T: ?Sized> PartialEq for *mut T {
    #[inline]
    fn eq(&self, other: &*mut T) -> bool {
        *self == *other
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: ?Sized> Eq for *mut T {}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: ?Sized> Ord for *mut T {
    #[inline]
    fn cmp(&self, other: &*mut T) -> Ordering {
        if self < other {
            Less
        } else if self == other {
            Equal
        } else {
            Greater
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: ?Sized> PartialOrd for *mut T {
    #[inline]
    fn partial_cmp(&self, other: &*mut T) -> Option<Ordering> {
        Some(self.cmp(other))
    }

    #[inline]
    fn lt(&self, other: &*mut T) -> bool {
        *self < *other
    }

    #[inline]
    fn le(&self, other: &*mut T) -> bool {
        *self <= *other
    }

    #[inline]
    fn gt(&self, other: &*mut T) -> bool {
        *self > *other
    }

    #[inline]
    fn ge(&self, other: &*mut T) -> bool {
        *self >= *other
    }
}
