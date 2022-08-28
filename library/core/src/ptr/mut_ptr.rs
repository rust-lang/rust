use super::*;
use crate::cmp::Ordering::{self, Equal, Greater, Less};
use crate::intrinsics;
use crate::slice::{self, SliceIndex};

impl<T: ?Sized> *mut T {
    /// Returns `true` if the pointer is null.
    ///
    /// Note that unsized types have many possible null pointers, as only the
    /// raw data pointer is considered, not their length, vtable, etc.
    /// Therefore, two pointers that are null may still not compare equal to
    /// each other.
    ///
    /// ## Behavior during const evaluation
    ///
    /// When this function is used during const evaluation, it may return `false` for pointers
    /// that turn out to be null at runtime. Specifically, when a pointer to some memory
    /// is offset beyond its bounds in such a way that the resulting pointer is null,
    /// the function will still return `false`. There is no way for CTFE to know
    /// the absolute position of that memory, so we cannot tell if the pointer is
    /// null or not.
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
    #[rustc_const_unstable(feature = "const_ptr_is_null", issue = "74939")]
    #[inline]
    pub const fn is_null(self) -> bool {
        // Compare via a cast to a thin pointer, so fat pointers are only
        // considering their "data" part for null-ness.
        (self as *mut u8).guaranteed_eq(null_mut())
    }

    /// Casts to a pointer of another type.
    #[stable(feature = "ptr_cast", since = "1.38.0")]
    #[rustc_const_stable(feature = "const_ptr_cast", since = "1.38.0")]
    #[inline(always)]
    pub const fn cast<U>(self) -> *mut U {
        self as _
    }

    /// Use the pointer value in a new pointer of another type.
    ///
    /// In case `val` is a (fat) pointer to an unsized type, this operation
    /// will ignore the pointer part, whereas for (thin) pointers to sized
    /// types, this has the same effect as a simple cast.
    ///
    /// The resulting pointer will have provenance of `self`, i.e., for a fat
    /// pointer, this operation is semantically the same as creating a new
    /// fat pointer with the data pointer value of `self` but the metadata of
    /// `val`.
    ///
    /// # Examples
    ///
    /// This function is primarily useful for allowing byte-wise pointer
    /// arithmetic on potentially fat pointers:
    ///
    /// ```
    /// #![feature(set_ptr_value)]
    /// # use core::fmt::Debug;
    /// let mut arr: [i32; 3] = [1, 2, 3];
    /// let mut ptr = arr.as_mut_ptr() as *mut dyn Debug;
    /// let thin = ptr as *mut u8;
    /// unsafe {
    ///     ptr = thin.add(8).with_metadata_of(ptr);
    ///     # assert_eq!(*(ptr as *mut i32), 3);
    ///     println!("{:?}", &*ptr); // will print "3"
    /// }
    /// ```
    #[unstable(feature = "set_ptr_value", issue = "75091")]
    #[must_use = "returns a new pointer rather than modifying its argument"]
    #[inline]
    pub fn with_metadata_of<U>(self, mut val: *mut U) -> *mut U
    where
        U: ?Sized,
    {
        let target = &mut val as *mut *mut U as *mut *mut u8;
        // SAFETY: In case of a thin pointer, this operations is identical
        // to a simple assignment. In case of a fat pointer, with the current
        // fat pointer layout implementation, the first field of such a
        // pointer is always the data pointer, which is likewise assigned.
        unsafe { *target = self as *mut u8 };
        val
    }

    /// Changes constness without changing the type.
    ///
    /// This is a bit safer than `as` because it wouldn't silently change the type if the code is
    /// refactored.
    ///
    /// While not strictly required (`*mut T` coerces to `*const T`), this is provided for symmetry
    /// with [`cast_mut`] on `*const T` and may have documentation value if used instead of implicit
    /// coercion.
    ///
    /// [`cast_mut`]: #method.cast_mut
    #[stable(feature = "ptr_const_cast", since = "CURRENT_RUSTC_VERSION")]
    #[rustc_const_stable(feature = "ptr_const_cast", since = "CURRENT_RUSTC_VERSION")]
    pub const fn cast_const(self) -> *const T {
        self as _
    }

    /// Casts a pointer to its raw bits.
    ///
    /// This is equivalent to `as usize`, but is more specific to enhance readability.
    /// The inverse method is [`from_bits`](#method.from_bits-1).
    ///
    /// In particular, `*p as usize` and `p as usize` will both compile for
    /// pointers to numeric types but do very different things, so using this
    /// helps emphasize that reading the bits was intentional.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(ptr_to_from_bits)]
    /// let mut array = [13, 42];
    /// let mut it = array.iter_mut();
    /// let p0: *mut i32 = it.next().unwrap();
    /// assert_eq!(<*mut _>::from_bits(p0.to_bits()), p0);
    /// let p1: *mut i32 = it.next().unwrap();
    /// assert_eq!(p1.to_bits() - p0.to_bits(), 4);
    /// ```
    #[unstable(feature = "ptr_to_from_bits", issue = "91126")]
    pub fn to_bits(self) -> usize
    where
        T: Sized,
    {
        self as usize
    }

    /// Creates a pointer from its raw bits.
    ///
    /// This is equivalent to `as *mut T`, but is more specific to enhance readability.
    /// The inverse method is [`to_bits`](#method.to_bits-1).
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(ptr_to_from_bits)]
    /// use std::ptr::NonNull;
    /// let dangling: *mut u8 = NonNull::dangling().as_ptr();
    /// assert_eq!(<*mut u8>::from_bits(1), dangling);
    /// ```
    #[unstable(feature = "ptr_to_from_bits", issue = "91126")]
    pub fn from_bits(bits: usize) -> Self
    where
        T: Sized,
    {
        bits as Self
    }

    /// Gets the "address" portion of the pointer.
    ///
    /// This is similar to `self as usize`, which semantically discards *provenance* and
    /// *address-space* information. However, unlike `self as usize`, casting the returned address
    /// back to a pointer yields [`invalid`][], which is undefined behavior to dereference. To
    /// properly restore the lost information and obtain a dereferencable pointer, use
    /// [`with_addr`][pointer::with_addr] or [`map_addr`][pointer::map_addr].
    ///
    /// If using those APIs is not possible because there is no way to preserve a pointer with the
    /// required provenance, use [`expose_addr`][pointer::expose_addr] and
    /// [`from_exposed_addr_mut`][from_exposed_addr_mut] instead. However, note that this makes
    /// your code less portable and less amenable to tools that check for compliance with the Rust
    /// memory model.
    ///
    /// On most platforms this will produce a value with the same bytes as the original
    /// pointer, because all the bytes are dedicated to describing the address.
    /// Platforms which need to store additional information in the pointer may
    /// perform a change of representation to produce a value containing only the address
    /// portion of the pointer. What that means is up to the platform to define.
    ///
    /// This API and its claimed semantics are part of the Strict Provenance experiment, and as such
    /// might change in the future (including possibly weakening this so it becomes wholly
    /// equivalent to `self as usize`). See the [module documentation][crate::ptr] for details.
    #[must_use]
    #[inline]
    #[unstable(feature = "strict_provenance", issue = "95228")]
    pub fn addr(self) -> usize
    where
        T: Sized,
    {
        // FIXME(strict_provenance_magic): I am magic and should be a compiler intrinsic.
        // SAFETY: Pointer-to-integer transmutes are valid (if you are okay with losing the
        // provenance).
        unsafe { mem::transmute(self) }
    }

    /// Gets the "address" portion of the pointer, and 'exposes' the "provenance" part for future
    /// use in [`from_exposed_addr`][].
    ///
    /// This is equivalent to `self as usize`, which semantically discards *provenance* and
    /// *address-space* information. Furthermore, this (like the `as` cast) has the implicit
    /// side-effect of marking the provenance as 'exposed', so on platforms that support it you can
    /// later call [`from_exposed_addr_mut`][] to reconstitute the original pointer including its
    /// provenance. (Reconstructing address space information, if required, is your responsibility.)
    ///
    /// Using this method means that code is *not* following Strict Provenance rules. Supporting
    /// [`from_exposed_addr_mut`][] complicates specification and reasoning and may not be supported
    /// by tools that help you to stay conformant with the Rust memory model, so it is recommended
    /// to use [`addr`][pointer::addr] wherever possible.
    ///
    /// On most platforms this will produce a value with the same bytes as the original pointer,
    /// because all the bytes are dedicated to describing the address. Platforms which need to store
    /// additional information in the pointer may not support this operation, since the 'expose'
    /// side-effect which is required for [`from_exposed_addr_mut`][] to work is typically not
    /// available.
    ///
    /// This API and its claimed semantics are part of the Strict Provenance experiment, see the
    /// [module documentation][crate::ptr] for details.
    ///
    /// [`from_exposed_addr_mut`]: from_exposed_addr_mut
    #[must_use]
    #[inline]
    #[unstable(feature = "strict_provenance", issue = "95228")]
    pub fn expose_addr(self) -> usize
    where
        T: Sized,
    {
        // FIXME(strict_provenance_magic): I am magic and should be a compiler intrinsic.
        self as usize
    }

    /// Creates a new pointer with the given address.
    ///
    /// This performs the same operation as an `addr as ptr` cast, but copies
    /// the *address-space* and *provenance* of `self` to the new pointer.
    /// This allows us to dynamically preserve and propagate this important
    /// information in a way that is otherwise impossible with a unary cast.
    ///
    /// This is equivalent to using [`wrapping_offset`][pointer::wrapping_offset] to offset
    /// `self` to the given address, and therefore has all the same capabilities and restrictions.
    ///
    /// This API and its claimed semantics are part of the Strict Provenance experiment,
    /// see the [module documentation][crate::ptr] for details.
    #[must_use]
    #[inline]
    #[unstable(feature = "strict_provenance", issue = "95228")]
    pub fn with_addr(self, addr: usize) -> Self
    where
        T: Sized,
    {
        // FIXME(strict_provenance_magic): I am magic and should be a compiler intrinsic.
        //
        // In the mean-time, this operation is defined to be "as if" it was
        // a wrapping_offset, so we can emulate it as such. This should properly
        // restore pointer provenance even under today's compiler.
        let self_addr = self.addr() as isize;
        let dest_addr = addr as isize;
        let offset = dest_addr.wrapping_sub(self_addr);

        // This is the canonical desugarring of this operation
        self.cast::<u8>().wrapping_offset(offset).cast::<T>()
    }

    /// Creates a new pointer by mapping `self`'s address to a new one.
    ///
    /// This is a convenience for [`with_addr`][pointer::with_addr], see that method for details.
    ///
    /// This API and its claimed semantics are part of the Strict Provenance experiment,
    /// see the [module documentation][crate::ptr] for details.
    #[must_use]
    #[inline]
    #[unstable(feature = "strict_provenance", issue = "95228")]
    pub fn map_addr(self, f: impl FnOnce(usize) -> usize) -> Self
    where
        T: Sized,
    {
        self.with_addr(f(self.addr()))
    }

    /// Decompose a (possibly wide) pointer into its address and metadata components.
    ///
    /// The pointer can be later reconstructed with [`from_raw_parts_mut`].
    #[unstable(feature = "ptr_metadata", issue = "81513")]
    #[rustc_const_unstable(feature = "ptr_metadata", issue = "81513")]
    #[inline]
    pub const fn to_raw_parts(self) -> (*mut (), <T as super::Pointee>::Metadata) {
        (self.cast(), super::metadata(self))
    }

    /// Returns `None` if the pointer is null, or else returns a shared reference to
    /// the value wrapped in `Some`. If the value may be uninitialized, [`as_uninit_ref`]
    /// must be used instead.
    ///
    /// For the mutable counterpart see [`as_mut`].
    ///
    /// [`as_uninit_ref`]: #method.as_uninit_ref-1
    /// [`as_mut`]: #method.as_mut
    ///
    /// # Safety
    ///
    /// When calling this method, you have to ensure that *either* the pointer is null *or*
    /// all of the following is true:
    ///
    /// * The pointer must be properly aligned.
    ///
    /// * It must be "dereferenceable" in the sense defined in [the module documentation].
    ///
    /// * The pointer must point to an initialized instance of `T`.
    ///
    /// * You must enforce Rust's aliasing rules, since the returned lifetime `'a` is
    ///   arbitrarily chosen and does not necessarily reflect the actual lifetime of the data.
    ///   In particular, while this reference exists, the memory the pointer points to must
    ///   not get mutated (except inside `UnsafeCell`).
    ///
    /// This applies even if the result of this method is unused!
    /// (The part about being initialized is not yet fully decided, but until
    /// it is, the only safe approach is to ensure that they are indeed initialized.)
    ///
    /// [the module documentation]: crate::ptr#safety
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
    ///         println!("We got back the value: {val_back}!");
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
    ///     println!("We got back the value: {val_back}!");
    /// }
    /// ```
    #[stable(feature = "ptr_as_ref", since = "1.9.0")]
    #[rustc_const_unstable(feature = "const_ptr_as_ref", issue = "91822")]
    #[inline]
    pub const unsafe fn as_ref<'a>(self) -> Option<&'a T> {
        // SAFETY: the caller must guarantee that `self` is valid for a
        // reference if it isn't null.
        if self.is_null() { None } else { unsafe { Some(&*self) } }
    }

    /// Returns `None` if the pointer is null, or else returns a shared reference to
    /// the value wrapped in `Some`. In contrast to [`as_ref`], this does not require
    /// that the value has to be initialized.
    ///
    /// For the mutable counterpart see [`as_uninit_mut`].
    ///
    /// [`as_ref`]: #method.as_ref-1
    /// [`as_uninit_mut`]: #method.as_uninit_mut
    ///
    /// # Safety
    ///
    /// When calling this method, you have to ensure that *either* the pointer is null *or*
    /// all of the following is true:
    ///
    /// * The pointer must be properly aligned.
    ///
    /// * It must be "dereferenceable" in the sense defined in [the module documentation].
    ///
    /// * You must enforce Rust's aliasing rules, since the returned lifetime `'a` is
    ///   arbitrarily chosen and does not necessarily reflect the actual lifetime of the data.
    ///   In particular, while this reference exists, the memory the pointer points to must
    ///   not get mutated (except inside `UnsafeCell`).
    ///
    /// This applies even if the result of this method is unused!
    ///
    /// [the module documentation]: crate::ptr#safety
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// #![feature(ptr_as_uninit)]
    ///
    /// let ptr: *mut u8 = &mut 10u8 as *mut u8;
    ///
    /// unsafe {
    ///     if let Some(val_back) = ptr.as_uninit_ref() {
    ///         println!("We got back the value: {}!", val_back.assume_init());
    ///     }
    /// }
    /// ```
    #[inline]
    #[unstable(feature = "ptr_as_uninit", issue = "75402")]
    #[rustc_const_unstable(feature = "const_ptr_as_ref", issue = "91822")]
    pub const unsafe fn as_uninit_ref<'a>(self) -> Option<&'a MaybeUninit<T>>
    where
        T: Sized,
    {
        // SAFETY: the caller must guarantee that `self` meets all the
        // requirements for a reference.
        if self.is_null() { None } else { Some(unsafe { &*(self as *const MaybeUninit<T>) }) }
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
    ///   byte past the end of the same [allocated object].
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
    /// [allocated object]: crate::ptr#allocated-object
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
    #[rustc_const_stable(feature = "const_ptr_offset", since = "1.61.0")]
    #[inline(always)]
    #[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
    pub const unsafe fn offset(self, count: isize) -> *mut T
    where
        T: Sized,
    {
        // SAFETY: the caller must uphold the safety contract for `offset`.
        // The obtained pointer is valid for writes since the caller must
        // guarantee that it points to the same allocated object as `self`.
        unsafe { intrinsics::offset(self, count) as *mut T }
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
    #[unstable(feature = "pointer_byte_offsets", issue = "96283")]
    #[rustc_const_unstable(feature = "const_pointer_byte_offsets", issue = "96283")]
    #[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
    pub const unsafe fn byte_offset(self, count: isize) -> Self {
        // SAFETY: the caller must uphold the safety contract for `offset`.
        let this = unsafe { self.cast::<u8>().offset(count).cast::<()>() };
        from_raw_parts_mut::<T>(this, metadata(self))
    }

    /// Calculates the offset from a pointer using wrapping arithmetic.
    /// `count` is in units of T; e.g., a `count` of 3 represents a pointer
    /// offset of `3 * size_of::<T>()` bytes.
    ///
    /// # Safety
    ///
    /// This operation itself is always safe, but using the resulting pointer is not.
    ///
    /// The resulting pointer "remembers" the [allocated object] that `self` points to; it must not
    /// be used to read or write other allocated objects.
    ///
    /// In other words, `let z = x.wrapping_offset((y as isize) - (x as isize))` does *not* make `z`
    /// the same as `y` even if we assume `T` has size `1` and there is no overflow: `z` is still
    /// attached to the object `x` is attached to, and dereferencing it is Undefined Behavior unless
    /// `x` and `y` point into the same allocated object.
    ///
    /// Compared to [`offset`], this method basically delays the requirement of staying within the
    /// same allocated object: [`offset`] is immediate Undefined Behavior when crossing object
    /// boundaries; `wrapping_offset` produces a pointer but still leads to Undefined Behavior if a
    /// pointer is dereferenced when it is out-of-bounds of the object it is attached to. [`offset`]
    /// can be optimized better and is thus preferable in performance-sensitive code.
    ///
    /// The delayed check only considers the value of the pointer that was dereferenced, not the
    /// intermediate values used during the computation of the final result. For example,
    /// `x.wrapping_offset(o).wrapping_offset(o.wrapping_neg())` is always the same as `x`. In other
    /// words, leaving the allocated object and then re-entering it later is permitted.
    ///
    /// [`offset`]: #method.offset
    /// [allocated object]: crate::ptr#allocated-object
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
    #[rustc_const_stable(feature = "const_ptr_offset", since = "1.61.0")]
    #[inline(always)]
    pub const fn wrapping_offset(self, count: isize) -> *mut T
    where
        T: Sized,
    {
        // SAFETY: the `arith_offset` intrinsic has no prerequisites to be called.
        unsafe { intrinsics::arith_offset(self, count) as *mut T }
    }

    /// Calculates the offset from a pointer in bytes using wrapping arithmetic.
    ///
    /// `count` is in units of **bytes**.
    ///
    /// This is purely a convenience for casting to a `u8` pointer and
    /// using [wrapping_offset][pointer::wrapping_offset] on it. See that method
    /// for documentation.
    ///
    /// For non-`Sized` pointees this operation changes only the data pointer,
    /// leaving the metadata untouched.
    #[must_use]
    #[inline(always)]
    #[unstable(feature = "pointer_byte_offsets", issue = "96283")]
    #[rustc_const_unstable(feature = "const_pointer_byte_offsets", issue = "96283")]
    pub const fn wrapping_byte_offset(self, count: isize) -> Self {
        from_raw_parts_mut::<T>(
            self.cast::<u8>().wrapping_offset(count).cast::<()>(),
            metadata(self),
        )
    }

    /// Masks out bits of the pointer according to a mask.
    ///
    /// This is convenience for `ptr.map_addr(|a| a & mask)`.
    ///
    /// For non-`Sized` pointees this operation changes only the data pointer,
    /// leaving the metadata untouched.
    #[cfg(not(bootstrap))]
    #[unstable(feature = "ptr_mask", issue = "98290")]
    #[must_use = "returns a new pointer rather than modifying its argument"]
    #[inline(always)]
    pub fn mask(self, mask: usize) -> *mut T {
        let this = intrinsics::ptr_mask(self.cast::<()>(), mask) as *mut ();
        from_raw_parts_mut::<T>(this, metadata(self))
    }

    /// Returns `None` if the pointer is null, or else returns a unique reference to
    /// the value wrapped in `Some`. If the value may be uninitialized, [`as_uninit_mut`]
    /// must be used instead.
    ///
    /// For the shared counterpart see [`as_ref`].
    ///
    /// [`as_uninit_mut`]: #method.as_uninit_mut
    /// [`as_ref`]: #method.as_ref-1
    ///
    /// # Safety
    ///
    /// When calling this method, you have to ensure that *either* the pointer is null *or*
    /// all of the following is true:
    ///
    /// * The pointer must be properly aligned.
    ///
    /// * It must be "dereferenceable" in the sense defined in [the module documentation].
    ///
    /// * The pointer must point to an initialized instance of `T`.
    ///
    /// * You must enforce Rust's aliasing rules, since the returned lifetime `'a` is
    ///   arbitrarily chosen and does not necessarily reflect the actual lifetime of the data.
    ///   In particular, while this reference exists, the memory the pointer points to must
    ///   not get accessed (read or written) through any other pointer.
    ///
    /// This applies even if the result of this method is unused!
    /// (The part about being initialized is not yet fully decided, but until
    /// it is, the only safe approach is to ensure that they are indeed initialized.)
    ///
    /// [the module documentation]: crate::ptr#safety
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
    /// # assert_eq!(s, [4, 2, 3]);
    /// println!("{s:?}"); // It'll print: "[4, 2, 3]".
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
    /// # assert_eq!(s, [4, 2, 3]);
    /// println!("{s:?}"); // It'll print: "[4, 2, 3]".
    /// ```
    #[stable(feature = "ptr_as_ref", since = "1.9.0")]
    #[rustc_const_unstable(feature = "const_ptr_as_ref", issue = "91822")]
    #[inline]
    pub const unsafe fn as_mut<'a>(self) -> Option<&'a mut T> {
        // SAFETY: the caller must guarantee that `self` is be valid for
        // a mutable reference if it isn't null.
        if self.is_null() { None } else { unsafe { Some(&mut *self) } }
    }

    /// Returns `None` if the pointer is null, or else returns a unique reference to
    /// the value wrapped in `Some`. In contrast to [`as_mut`], this does not require
    /// that the value has to be initialized.
    ///
    /// For the shared counterpart see [`as_uninit_ref`].
    ///
    /// [`as_mut`]: #method.as_mut
    /// [`as_uninit_ref`]: #method.as_uninit_ref-1
    ///
    /// # Safety
    ///
    /// When calling this method, you have to ensure that *either* the pointer is null *or*
    /// all of the following is true:
    ///
    /// * The pointer must be properly aligned.
    ///
    /// * It must be "dereferenceable" in the sense defined in [the module documentation].
    ///
    /// * You must enforce Rust's aliasing rules, since the returned lifetime `'a` is
    ///   arbitrarily chosen and does not necessarily reflect the actual lifetime of the data.
    ///   In particular, while this reference exists, the memory the pointer points to must
    ///   not get accessed (read or written) through any other pointer.
    ///
    /// This applies even if the result of this method is unused!
    ///
    /// [the module documentation]: crate::ptr#safety
    #[inline]
    #[unstable(feature = "ptr_as_uninit", issue = "75402")]
    #[rustc_const_unstable(feature = "const_ptr_as_ref", issue = "91822")]
    pub const unsafe fn as_uninit_mut<'a>(self) -> Option<&'a mut MaybeUninit<T>>
    where
        T: Sized,
    {
        // SAFETY: the caller must guarantee that `self` meets all the
        // requirements for a reference.
        if self.is_null() { None } else { Some(unsafe { &mut *(self as *mut MaybeUninit<T>) }) }
    }

    /// Returns whether two pointers are guaranteed to be equal.
    ///
    /// At runtime this function behaves like `self == other`.
    /// However, in some contexts (e.g., compile-time evaluation),
    /// it is not always possible to determine equality of two pointers, so this function may
    /// spuriously return `false` for pointers that later actually turn out to be equal.
    /// But when it returns `true`, the pointers are guaranteed to be equal.
    ///
    /// This function is the mirror of [`guaranteed_ne`], but not its inverse. There are pointer
    /// comparisons for which both functions return `false`.
    ///
    /// [`guaranteed_ne`]: #method.guaranteed_ne
    ///
    /// The return value may change depending on the compiler version and unsafe code might not
    /// rely on the result of this function for soundness. It is suggested to only use this function
    /// for performance optimizations where spurious `false` return values by this function do not
    /// affect the outcome, but just the performance.
    /// The consequences of using this method to make runtime and compile-time code behave
    /// differently have not been explored. This method should not be used to introduce such
    /// differences, and it should also not be stabilized before we have a better understanding
    /// of this issue.
    #[unstable(feature = "const_raw_ptr_comparison", issue = "53020")]
    #[rustc_const_unstable(feature = "const_raw_ptr_comparison", issue = "53020")]
    #[inline]
    pub const fn guaranteed_eq(self, other: *mut T) -> bool
    where
        T: Sized,
    {
        intrinsics::ptr_guaranteed_eq(self as *const _, other as *const _)
    }

    /// Returns whether two pointers are guaranteed to be unequal.
    ///
    /// At runtime this function behaves like `self != other`.
    /// However, in some contexts (e.g., compile-time evaluation),
    /// it is not always possible to determine the inequality of two pointers, so this function may
    /// spuriously return `false` for pointers that later actually turn out to be unequal.
    /// But when it returns `true`, the pointers are guaranteed to be unequal.
    ///
    /// This function is the mirror of [`guaranteed_eq`], but not its inverse. There are pointer
    /// comparisons for which both functions return `false`.
    ///
    /// [`guaranteed_eq`]: #method.guaranteed_eq
    ///
    /// The return value may change depending on the compiler version and unsafe code might not
    /// rely on the result of this function for soundness. It is suggested to only use this function
    /// for performance optimizations where spurious `false` return values by this function do not
    /// affect the outcome, but just the performance.
    /// The consequences of using this method to make runtime and compile-time code behave
    /// differently have not been explored. This method should not be used to introduce such
    /// differences, and it should also not be stabilized before we have a better understanding
    /// of this issue.
    #[unstable(feature = "const_raw_ptr_comparison", issue = "53020")]
    #[rustc_const_unstable(feature = "const_raw_ptr_comparison", issue = "53020")]
    #[inline]
    pub const unsafe fn guaranteed_ne(self, other: *mut T) -> bool
    where
        T: Sized,
    {
        intrinsics::ptr_guaranteed_ne(self as *const _, other as *const _)
    }

    /// Calculates the distance between two pointers. The returned value is in
    /// units of T: the distance in bytes divided by `mem::size_of::<T>()`.
    ///
    /// This function is the inverse of [`offset`].
    ///
    /// [`offset`]: #method.offset-1
    ///
    /// # Safety
    ///
    /// If any of the following conditions are violated, the result is Undefined
    /// Behavior:
    ///
    /// * Both the starting and other pointer must be either in bounds or one
    ///   byte past the end of the same [allocated object].
    ///
    /// * Both pointers must be *derived from* a pointer to the same object.
    ///   (See below for an example.)
    ///
    /// * The distance between the pointers, in bytes, must be an exact multiple
    ///   of the size of `T`.
    ///
    /// * The distance between the pointers, **in bytes**, cannot overflow an `isize`.
    ///
    /// * The distance being in bounds cannot rely on "wrapping around" the address space.
    ///
    /// Rust types are never larger than `isize::MAX` and Rust allocations never wrap around the
    /// address space, so two pointers within some value of any Rust type `T` will always satisfy
    /// the last two conditions. The standard library also generally ensures that allocations
    /// never reach a size where an offset is a concern. For instance, `Vec` and `Box` ensure they
    /// never allocate more than `isize::MAX` bytes, so `ptr_into_vec.offset_from(vec.as_ptr())`
    /// always satisfies the last two conditions.
    ///
    /// Most platforms fundamentally can't even construct such a large allocation.
    /// For instance, no known 64-bit platform can ever serve a request
    /// for 2<sup>63</sup> bytes due to page-table limitations or splitting the address space.
    /// However, some 32-bit and 16-bit platforms may successfully serve a request for
    /// more than `isize::MAX` bytes with things like Physical Address
    /// Extension. As such, memory acquired directly from allocators or memory
    /// mapped files *may* be too large to handle with this function.
    /// (Note that [`offset`] and [`add`] also have a similar limitation and hence cannot be used on
    /// such large allocations either.)
    ///
    /// [`add`]: #method.add
    /// [allocated object]: crate::ptr#allocated-object
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
    ///
    /// *Incorrect* usage:
    ///
    /// ```rust,no_run
    /// let ptr1 = Box::into_raw(Box::new(0u8));
    /// let ptr2 = Box::into_raw(Box::new(1u8));
    /// let diff = (ptr2 as isize).wrapping_sub(ptr1 as isize);
    /// // Make ptr2_other an "alias" of ptr2, but derived from ptr1.
    /// let ptr2_other = (ptr1 as *mut u8).wrapping_offset(diff);
    /// assert_eq!(ptr2 as usize, ptr2_other as usize);
    /// // Since ptr2_other and ptr2 are derived from pointers to different objects,
    /// // computing their offset is undefined behavior, even though
    /// // they point to the same address!
    /// unsafe {
    ///     let zero = ptr2_other.offset_from(ptr2); // Undefined Behavior
    /// }
    /// ```
    #[stable(feature = "ptr_offset_from", since = "1.47.0")]
    #[rustc_const_stable(feature = "const_ptr_offset_from", since = "1.65.0")]
    #[inline(always)]
    #[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
    pub const unsafe fn offset_from(self, origin: *const T) -> isize
    where
        T: Sized,
    {
        // SAFETY: the caller must uphold the safety contract for `offset_from`.
        unsafe { (self as *const T).offset_from(origin) }
    }

    /// Calculates the distance between two pointers. The returned value is in
    /// units of **bytes**.
    ///
    /// This is purely a convenience for casting to a `u8` pointer and
    /// using [offset_from][pointer::offset_from] on it. See that method for
    /// documentation and safety requirements.
    ///
    /// For non-`Sized` pointees this operation considers only the data pointers,
    /// ignoring the metadata.
    #[inline(always)]
    #[unstable(feature = "pointer_byte_offsets", issue = "96283")]
    #[rustc_const_unstable(feature = "const_pointer_byte_offsets", issue = "96283")]
    #[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
    pub const unsafe fn byte_offset_from(self, origin: *const T) -> isize {
        // SAFETY: the caller must uphold the safety contract for `offset_from`.
        unsafe { self.cast::<u8>().offset_from(origin.cast::<u8>()) }
    }

    /// Calculates the distance between two pointers, *where it's known that
    /// `self` is equal to or greater than `origin`*. The returned value is in
    /// units of T: the distance in bytes is divided by `mem::size_of::<T>()`.
    ///
    /// This computes the same value that [`offset_from`](#method.offset_from)
    /// would compute, but with the added precondition that that the offset is
    /// guaranteed to be non-negative.  This method is equivalent to
    /// `usize::from(self.offset_from(origin)).unwrap_unchecked()`,
    /// but it provides slightly more information to the optimizer, which can
    /// sometimes allow it to optimize slightly better with some backends.
    ///
    /// This method can be though of as recovering the `count` that was passed
    /// to [`add`](#method.add) (or, with the parameters in the other order,
    /// to [`sub`](#method.sub)).  The following are all equivalent, assuming
    /// that their safety preconditions are met:
    /// ```rust
    /// # #![feature(ptr_sub_ptr)]
    /// # unsafe fn blah(ptr: *mut i32, origin: *mut i32, count: usize) -> bool {
    /// ptr.sub_ptr(origin) == count
    /// # &&
    /// origin.add(count) == ptr
    /// # &&
    /// ptr.sub(count) == origin
    /// # }
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
    /// #![feature(ptr_sub_ptr)]
    ///
    /// let mut a = [0; 5];
    /// let p: *mut i32 = a.as_mut_ptr();
    /// unsafe {
    ///     let ptr1: *mut i32 = p.add(1);
    ///     let ptr2: *mut i32 = p.add(3);
    ///
    ///     assert_eq!(ptr2.sub_ptr(ptr1), 2);
    ///     assert_eq!(ptr1.add(2), ptr2);
    ///     assert_eq!(ptr2.sub(2), ptr1);
    ///     assert_eq!(ptr2.sub_ptr(ptr2), 0);
    /// }
    ///
    /// // This would be incorrect, as the pointers are not correctly ordered:
    /// // ptr1.offset_from(ptr2)
    #[unstable(feature = "ptr_sub_ptr", issue = "95892")]
    #[rustc_const_unstable(feature = "const_ptr_sub_ptr", issue = "95892")]
    #[inline]
    #[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
    pub const unsafe fn sub_ptr(self, origin: *const T) -> usize
    where
        T: Sized,
    {
        // SAFETY: the caller must uphold the safety contract for `sub_ptr`.
        unsafe { (self as *const T).sub_ptr(origin) }
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
    ///   byte past the end of the same [allocated object].
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
    /// [allocated object]: crate::ptr#allocated-object
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
    #[rustc_const_stable(feature = "const_ptr_offset", since = "1.61.0")]
    #[inline(always)]
    #[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
    pub const unsafe fn add(self, count: usize) -> Self
    where
        T: Sized,
    {
        // SAFETY: the caller must uphold the safety contract for `offset`.
        unsafe { self.offset(count as isize) }
    }

    /// Calculates the offset from a pointer in bytes (convenience for `.byte_offset(count as isize)`).
    ///
    /// `count` is in units of bytes.
    ///
    /// This is purely a convenience for casting to a `u8` pointer and
    /// using [add][pointer::add] on it. See that method for documentation
    /// and safety requirements.
    ///
    /// For non-`Sized` pointees this operation changes only the data pointer,
    /// leaving the metadata untouched.
    #[must_use]
    #[inline(always)]
    #[unstable(feature = "pointer_byte_offsets", issue = "96283")]
    #[rustc_const_unstable(feature = "const_pointer_byte_offsets", issue = "96283")]
    #[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
    pub const unsafe fn byte_add(self, count: usize) -> Self {
        // SAFETY: the caller must uphold the safety contract for `add`.
        let this = unsafe { self.cast::<u8>().add(count).cast::<()>() };
        from_raw_parts_mut::<T>(this, metadata(self))
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
    ///   byte past the end of the same [allocated object].
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
    /// [allocated object]: crate::ptr#allocated-object
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
    #[rustc_const_stable(feature = "const_ptr_offset", since = "1.61.0")]
    #[inline]
    #[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
    pub const unsafe fn sub(self, count: usize) -> Self
    where
        T: Sized,
    {
        // SAFETY: the caller must uphold the safety contract for `offset`.
        unsafe { self.offset((count as isize).wrapping_neg()) }
    }

    /// Calculates the offset from a pointer in bytes (convenience for
    /// `.byte_offset((count as isize).wrapping_neg())`).
    ///
    /// `count` is in units of bytes.
    ///
    /// This is purely a convenience for casting to a `u8` pointer and
    /// using [sub][pointer::sub] on it. See that method for documentation
    /// and safety requirements.
    ///
    /// For non-`Sized` pointees this operation changes only the data pointer,
    /// leaving the metadata untouched.
    #[must_use]
    #[inline(always)]
    #[unstable(feature = "pointer_byte_offsets", issue = "96283")]
    #[rustc_const_unstable(feature = "const_pointer_byte_offsets", issue = "96283")]
    #[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
    pub const unsafe fn byte_sub(self, count: usize) -> Self {
        // SAFETY: the caller must uphold the safety contract for `sub`.
        let this = unsafe { self.cast::<u8>().sub(count).cast::<()>() };
        from_raw_parts_mut::<T>(this, metadata(self))
    }

    /// Calculates the offset from a pointer using wrapping arithmetic.
    /// (convenience for `.wrapping_offset(count as isize)`)
    ///
    /// `count` is in units of T; e.g., a `count` of 3 represents a pointer
    /// offset of `3 * size_of::<T>()` bytes.
    ///
    /// # Safety
    ///
    /// This operation itself is always safe, but using the resulting pointer is not.
    ///
    /// The resulting pointer "remembers" the [allocated object] that `self` points to; it must not
    /// be used to read or write other allocated objects.
    ///
    /// In other words, `let z = x.wrapping_add((y as usize) - (x as usize))` does *not* make `z`
    /// the same as `y` even if we assume `T` has size `1` and there is no overflow: `z` is still
    /// attached to the object `x` is attached to, and dereferencing it is Undefined Behavior unless
    /// `x` and `y` point into the same allocated object.
    ///
    /// Compared to [`add`], this method basically delays the requirement of staying within the
    /// same allocated object: [`add`] is immediate Undefined Behavior when crossing object
    /// boundaries; `wrapping_add` produces a pointer but still leads to Undefined Behavior if a
    /// pointer is dereferenced when it is out-of-bounds of the object it is attached to. [`add`]
    /// can be optimized better and is thus preferable in performance-sensitive code.
    ///
    /// The delayed check only considers the value of the pointer that was dereferenced, not the
    /// intermediate values used during the computation of the final result. For example,
    /// `x.wrapping_add(o).wrapping_sub(o)` is always the same as `x`. In other words, leaving the
    /// allocated object and then re-entering it later is permitted.
    ///
    /// [`add`]: #method.add
    /// [allocated object]: crate::ptr#allocated-object
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
    #[rustc_const_stable(feature = "const_ptr_offset", since = "1.61.0")]
    #[inline(always)]
    pub const fn wrapping_add(self, count: usize) -> Self
    where
        T: Sized,
    {
        self.wrapping_offset(count as isize)
    }

    /// Calculates the offset from a pointer in bytes using wrapping arithmetic.
    /// (convenience for `.wrapping_byte_offset(count as isize)`)
    ///
    /// `count` is in units of bytes.
    ///
    /// This is purely a convenience for casting to a `u8` pointer and
    /// using [wrapping_add][pointer::wrapping_add] on it. See that method for documentation.
    ///
    /// For non-`Sized` pointees this operation changes only the data pointer,
    /// leaving the metadata untouched.
    #[must_use]
    #[inline(always)]
    #[unstable(feature = "pointer_byte_offsets", issue = "96283")]
    #[rustc_const_unstable(feature = "const_pointer_byte_offsets", issue = "96283")]
    pub const fn wrapping_byte_add(self, count: usize) -> Self {
        from_raw_parts_mut::<T>(self.cast::<u8>().wrapping_add(count).cast::<()>(), metadata(self))
    }

    /// Calculates the offset from a pointer using wrapping arithmetic.
    /// (convenience for `.wrapping_offset((count as isize).wrapping_neg())`)
    ///
    /// `count` is in units of T; e.g., a `count` of 3 represents a pointer
    /// offset of `3 * size_of::<T>()` bytes.
    ///
    /// # Safety
    ///
    /// This operation itself is always safe, but using the resulting pointer is not.
    ///
    /// The resulting pointer "remembers" the [allocated object] that `self` points to; it must not
    /// be used to read or write other allocated objects.
    ///
    /// In other words, `let z = x.wrapping_sub((x as usize) - (y as usize))` does *not* make `z`
    /// the same as `y` even if we assume `T` has size `1` and there is no overflow: `z` is still
    /// attached to the object `x` is attached to, and dereferencing it is Undefined Behavior unless
    /// `x` and `y` point into the same allocated object.
    ///
    /// Compared to [`sub`], this method basically delays the requirement of staying within the
    /// same allocated object: [`sub`] is immediate Undefined Behavior when crossing object
    /// boundaries; `wrapping_sub` produces a pointer but still leads to Undefined Behavior if a
    /// pointer is dereferenced when it is out-of-bounds of the object it is attached to. [`sub`]
    /// can be optimized better and is thus preferable in performance-sensitive code.
    ///
    /// The delayed check only considers the value of the pointer that was dereferenced, not the
    /// intermediate values used during the computation of the final result. For example,
    /// `x.wrapping_add(o).wrapping_sub(o)` is always the same as `x`. In other words, leaving the
    /// allocated object and then re-entering it later is permitted.
    ///
    /// [`sub`]: #method.sub
    /// [allocated object]: crate::ptr#allocated-object
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
    #[rustc_const_stable(feature = "const_ptr_offset", since = "1.61.0")]
    #[inline]
    pub const fn wrapping_sub(self, count: usize) -> Self
    where
        T: Sized,
    {
        self.wrapping_offset((count as isize).wrapping_neg())
    }

    /// Calculates the offset from a pointer in bytes using wrapping arithmetic.
    /// (convenience for `.wrapping_offset((count as isize).wrapping_neg())`)
    ///
    /// `count` is in units of bytes.
    ///
    /// This is purely a convenience for casting to a `u8` pointer and
    /// using [wrapping_sub][pointer::wrapping_sub] on it. See that method for documentation.
    ///
    /// For non-`Sized` pointees this operation changes only the data pointer,
    /// leaving the metadata untouched.
    #[must_use]
    #[inline(always)]
    #[unstable(feature = "pointer_byte_offsets", issue = "96283")]
    #[rustc_const_unstable(feature = "const_pointer_byte_offsets", issue = "96283")]
    pub const fn wrapping_byte_sub(self, count: usize) -> Self {
        from_raw_parts_mut::<T>(self.cast::<u8>().wrapping_sub(count).cast::<()>(), metadata(self))
    }

    /// Reads the value from `self` without moving it. This leaves the
    /// memory in `self` unchanged.
    ///
    /// See [`ptr::read`] for safety concerns and examples.
    ///
    /// [`ptr::read`]: crate::ptr::read()
    #[stable(feature = "pointer_methods", since = "1.26.0")]
    #[rustc_const_unstable(feature = "const_ptr_read", issue = "80377")]
    #[inline(always)]
    #[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
    pub const unsafe fn read(self) -> T
    where
        T: Sized,
    {
        // SAFETY: the caller must uphold the safety contract for ``.
        unsafe { read(self) }
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
    #[stable(feature = "pointer_methods", since = "1.26.0")]
    #[inline(always)]
    #[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
    pub unsafe fn read_volatile(self) -> T
    where
        T: Sized,
    {
        // SAFETY: the caller must uphold the safety contract for `read_volatile`.
        unsafe { read_volatile(self) }
    }

    /// Reads the value from `self` without moving it. This leaves the
    /// memory in `self` unchanged.
    ///
    /// Unlike `read`, the pointer may be unaligned.
    ///
    /// See [`ptr::read_unaligned`] for safety concerns and examples.
    ///
    /// [`ptr::read_unaligned`]: crate::ptr::read_unaligned()
    #[stable(feature = "pointer_methods", since = "1.26.0")]
    #[rustc_const_unstable(feature = "const_ptr_read", issue = "80377")]
    #[inline(always)]
    #[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
    pub const unsafe fn read_unaligned(self) -> T
    where
        T: Sized,
    {
        // SAFETY: the caller must uphold the safety contract for `read_unaligned`.
        unsafe { read_unaligned(self) }
    }

    /// Copies `count * size_of<T>` bytes from `self` to `dest`. The source
    /// and destination may overlap.
    ///
    /// NOTE: this has the *same* argument order as [`ptr::copy`].
    ///
    /// See [`ptr::copy`] for safety concerns and examples.
    ///
    /// [`ptr::copy`]: crate::ptr::copy()
    #[rustc_const_stable(feature = "const_intrinsic_copy", since = "1.63.0")]
    #[stable(feature = "pointer_methods", since = "1.26.0")]
    #[inline(always)]
    #[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
    pub const unsafe fn copy_to(self, dest: *mut T, count: usize)
    where
        T: Sized,
    {
        // SAFETY: the caller must uphold the safety contract for `copy`.
        unsafe { copy(self, dest, count) }
    }

    /// Copies `count * size_of<T>` bytes from `self` to `dest`. The source
    /// and destination may *not* overlap.
    ///
    /// NOTE: this has the *same* argument order as [`ptr::copy_nonoverlapping`].
    ///
    /// See [`ptr::copy_nonoverlapping`] for safety concerns and examples.
    ///
    /// [`ptr::copy_nonoverlapping`]: crate::ptr::copy_nonoverlapping()
    #[rustc_const_stable(feature = "const_intrinsic_copy", since = "1.63.0")]
    #[stable(feature = "pointer_methods", since = "1.26.0")]
    #[inline(always)]
    #[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
    pub const unsafe fn copy_to_nonoverlapping(self, dest: *mut T, count: usize)
    where
        T: Sized,
    {
        // SAFETY: the caller must uphold the safety contract for `copy_nonoverlapping`.
        unsafe { copy_nonoverlapping(self, dest, count) }
    }

    /// Copies `count * size_of<T>` bytes from `src` to `self`. The source
    /// and destination may overlap.
    ///
    /// NOTE: this has the *opposite* argument order of [`ptr::copy`].
    ///
    /// See [`ptr::copy`] for safety concerns and examples.
    ///
    /// [`ptr::copy`]: crate::ptr::copy()
    #[rustc_const_stable(feature = "const_intrinsic_copy", since = "1.63.0")]
    #[stable(feature = "pointer_methods", since = "1.26.0")]
    #[inline(always)]
    #[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
    pub const unsafe fn copy_from(self, src: *const T, count: usize)
    where
        T: Sized,
    {
        // SAFETY: the caller must uphold the safety contract for `copy`.
        unsafe { copy(src, self, count) }
    }

    /// Copies `count * size_of<T>` bytes from `src` to `self`. The source
    /// and destination may *not* overlap.
    ///
    /// NOTE: this has the *opposite* argument order of [`ptr::copy_nonoverlapping`].
    ///
    /// See [`ptr::copy_nonoverlapping`] for safety concerns and examples.
    ///
    /// [`ptr::copy_nonoverlapping`]: crate::ptr::copy_nonoverlapping()
    #[rustc_const_stable(feature = "const_intrinsic_copy", since = "1.63.0")]
    #[stable(feature = "pointer_methods", since = "1.26.0")]
    #[inline(always)]
    #[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
    pub const unsafe fn copy_from_nonoverlapping(self, src: *const T, count: usize)
    where
        T: Sized,
    {
        // SAFETY: the caller must uphold the safety contract for `copy_nonoverlapping`.
        unsafe { copy_nonoverlapping(src, self, count) }
    }

    /// Executes the destructor (if any) of the pointed-to value.
    ///
    /// See [`ptr::drop_in_place`] for safety concerns and examples.
    ///
    /// [`ptr::drop_in_place`]: crate::ptr::drop_in_place()
    #[stable(feature = "pointer_methods", since = "1.26.0")]
    #[inline(always)]
    pub unsafe fn drop_in_place(self) {
        // SAFETY: the caller must uphold the safety contract for `drop_in_place`.
        unsafe { drop_in_place(self) }
    }

    /// Overwrites a memory location with the given value without reading or
    /// dropping the old value.
    ///
    /// See [`ptr::write`] for safety concerns and examples.
    ///
    /// [`ptr::write`]: crate::ptr::write()
    #[stable(feature = "pointer_methods", since = "1.26.0")]
    #[rustc_const_unstable(feature = "const_ptr_write", issue = "86302")]
    #[inline(always)]
    #[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
    pub const unsafe fn write(self, val: T)
    where
        T: Sized,
    {
        // SAFETY: the caller must uphold the safety contract for `write`.
        unsafe { write(self, val) }
    }

    /// Invokes memset on the specified pointer, setting `count * size_of::<T>()`
    /// bytes of memory starting at `self` to `val`.
    ///
    /// See [`ptr::write_bytes`] for safety concerns and examples.
    ///
    /// [`ptr::write_bytes`]: crate::ptr::write_bytes()
    #[doc(alias = "memset")]
    #[stable(feature = "pointer_methods", since = "1.26.0")]
    #[rustc_const_unstable(feature = "const_ptr_write", issue = "86302")]
    #[inline(always)]
    #[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
    pub const unsafe fn write_bytes(self, val: u8, count: usize)
    where
        T: Sized,
    {
        // SAFETY: the caller must uphold the safety contract for `write_bytes`.
        unsafe { write_bytes(self, val, count) }
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
    #[stable(feature = "pointer_methods", since = "1.26.0")]
    #[inline(always)]
    #[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
    pub unsafe fn write_volatile(self, val: T)
    where
        T: Sized,
    {
        // SAFETY: the caller must uphold the safety contract for `write_volatile`.
        unsafe { write_volatile(self, val) }
    }

    /// Overwrites a memory location with the given value without reading or
    /// dropping the old value.
    ///
    /// Unlike `write`, the pointer may be unaligned.
    ///
    /// See [`ptr::write_unaligned`] for safety concerns and examples.
    ///
    /// [`ptr::write_unaligned`]: crate::ptr::write_unaligned()
    #[stable(feature = "pointer_methods", since = "1.26.0")]
    #[rustc_const_unstable(feature = "const_ptr_write", issue = "86302")]
    #[inline(always)]
    #[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
    pub const unsafe fn write_unaligned(self, val: T)
    where
        T: Sized,
    {
        // SAFETY: the caller must uphold the safety contract for `write_unaligned`.
        unsafe { write_unaligned(self, val) }
    }

    /// Replaces the value at `self` with `src`, returning the old
    /// value, without dropping either.
    ///
    /// See [`ptr::replace`] for safety concerns and examples.
    ///
    /// [`ptr::replace`]: crate::ptr::replace()
    #[stable(feature = "pointer_methods", since = "1.26.0")]
    #[inline(always)]
    pub unsafe fn replace(self, src: T) -> T
    where
        T: Sized,
    {
        // SAFETY: the caller must uphold the safety contract for `replace`.
        unsafe { replace(self, src) }
    }

    /// Swaps the values at two mutable locations of the same type, without
    /// deinitializing either. They may overlap, unlike `mem::swap` which is
    /// otherwise equivalent.
    ///
    /// See [`ptr::swap`] for safety concerns and examples.
    ///
    /// [`ptr::swap`]: crate::ptr::swap()
    #[stable(feature = "pointer_methods", since = "1.26.0")]
    #[rustc_const_unstable(feature = "const_swap", issue = "83163")]
    #[inline(always)]
    pub const unsafe fn swap(self, with: *mut T)
    where
        T: Sized,
    {
        // SAFETY: the caller must uphold the safety contract for `swap`.
        unsafe { swap(self, with) }
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
    /// use std::mem::align_of;
    ///
    /// # unsafe {
    /// let mut x = [5_u8, 6, 7, 8, 9];
    /// let ptr = x.as_mut_ptr();
    /// let offset = ptr.align_offset(align_of::<u16>());
    ///
    /// if offset < x.len() - 1 {
    ///     let u16_ptr = ptr.add(offset).cast::<u16>();
    ///     *u16_ptr = 0;
    ///
    ///     assert!(x == [0, 0, 7, 8, 9] || x == [5, 0, 0, 8, 9]);
    /// } else {
    ///     // while the pointer can be aligned via `offset`, it would point
    ///     // outside the allocation
    /// }
    /// # }
    /// ```
    #[stable(feature = "align_offset", since = "1.36.0")]
    #[rustc_const_unstable(feature = "const_align_offset", issue = "90962")]
    pub const fn align_offset(self, align: usize) -> usize
    where
        T: Sized,
    {
        if !align.is_power_of_two() {
            panic!("align_offset: align is not a power-of-two");
        }

        fn rt_impl<T>(p: *mut T, align: usize) -> usize {
            // SAFETY: `align` has been checked to be a power of 2 above
            unsafe { align_offset(p, align) }
        }

        const fn ctfe_impl<T>(_: *mut T, _: usize) -> usize {
            usize::MAX
        }

        // SAFETY:
        // It is permissible for `align_offset` to always return `usize::MAX`,
        // algorithm correctness can not depend on `align_offset` returning non-max values.
        //
        // As such the behaviour can't change after replacing `align_offset` with `usize::MAX`, only performance can.
        unsafe { intrinsics::const_eval_select((self, align), ctfe_impl, rt_impl) }
    }

    /// Returns whether the pointer is properly aligned for `T`.
    #[must_use]
    #[inline]
    #[unstable(feature = "pointer_is_aligned", issue = "96284")]
    pub fn is_aligned(self) -> bool
    where
        T: Sized,
    {
        self.is_aligned_to(core::mem::align_of::<T>())
    }

    /// Returns whether the pointer is aligned to `align`.
    ///
    /// For non-`Sized` pointees this operation considers only the data pointer,
    /// ignoring the metadata.
    ///
    /// # Panics
    ///
    /// The function panics if `align` is not a power-of-two (this includes 0).
    #[must_use]
    #[inline]
    #[unstable(feature = "pointer_is_aligned", issue = "96284")]
    pub fn is_aligned_to(self, align: usize) -> bool {
        if !align.is_power_of_two() {
            panic!("is_aligned_to: align is not a power-of-two");
        }

        // Cast is needed for `T: !Sized`
        self.cast::<u8>().addr() & align - 1 == 0
    }
}

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
    /// use std::ptr;
    ///
    /// let slice: *mut [i8] = ptr::slice_from_raw_parts_mut(ptr::null_mut(), 3);
    /// assert_eq!(slice.len(), 3);
    /// ```
    #[inline(always)]
    #[unstable(feature = "slice_ptr_len", issue = "71146")]
    #[rustc_const_unstable(feature = "const_slice_ptr_len", issue = "71146")]
    pub const fn len(self) -> usize {
        metadata(self)
    }

    /// Returns `true` if the raw slice has a length of 0.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(slice_ptr_len)]
    ///
    /// let mut a = [1, 2, 3];
    /// let ptr = &mut a as *mut [_];
    /// assert!(!ptr.is_empty());
    /// ```
    #[inline(always)]
    #[unstable(feature = "slice_ptr_len", issue = "71146")]
    #[rustc_const_unstable(feature = "const_slice_ptr_len", issue = "71146")]
    pub const fn is_empty(self) -> bool {
        self.len() == 0
    }

    /// Divides one mutable raw slice into two at an index.
    ///
    /// The first will contain all indices from `[0, mid)` (excluding
    /// the index `mid` itself) and the second will contain all
    /// indices from `[mid, len)` (excluding the index `len` itself).
    ///
    /// # Panics
    ///
    /// Panics if `mid > len`.
    ///
    /// # Safety
    ///
    /// `mid` must be [in-bounds] of the underlying [allocated object].
    /// Which means `self` must be dereferenceable and span a single allocation
    /// that is at least `mid * size_of::<T>()` bytes long. Not upholding these
    /// requirements is *[undefined behavior]* even if the resulting pointers are not used.
    ///
    /// Since `len` being in-bounds it is not a safety invariant of `*mut [T]` the
    /// safety requirements of this method are the same as for [`split_at_mut_unchecked`].
    /// The explicit bounds check is only as useful as `len` is correct.
    ///
    /// [`split_at_mut_unchecked`]: #method.split_at_mut_unchecked
    /// [in-bounds]: #method.add
    /// [allocated object]: crate::ptr#allocated-object
    /// [undefined behavior]: https://doc.rust-lang.org/reference/behavior-considered-undefined.html
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(raw_slice_split)]
    /// #![feature(slice_ptr_get)]
    ///
    /// let mut v = [1, 0, 3, 0, 5, 6];
    /// let ptr = &mut v as *mut [_];
    /// unsafe {
    ///     let (left, right) = ptr.split_at_mut(2);
    ///     assert_eq!(&*left, [1, 0]);
    ///     assert_eq!(&*right, [3, 0, 5, 6]);
    /// }
    /// ```
    #[inline(always)]
    #[track_caller]
    #[unstable(feature = "raw_slice_split", issue = "95595")]
    pub unsafe fn split_at_mut(self, mid: usize) -> (*mut [T], *mut [T]) {
        assert!(mid <= self.len());
        // SAFETY: The assert above is only a safety-net as long as `self.len()` is correct
        // The actual safety requirements of this function are the same as for `split_at_mut_unchecked`
        unsafe { self.split_at_mut_unchecked(mid) }
    }

    /// Divides one mutable raw slice into two at an index, without doing bounds checking.
    ///
    /// The first will contain all indices from `[0, mid)` (excluding
    /// the index `mid` itself) and the second will contain all
    /// indices from `[mid, len)` (excluding the index `len` itself).
    ///
    /// # Safety
    ///
    /// `mid` must be [in-bounds] of the underlying [allocated object].
    /// Which means `self` must be dereferenceable and span a single allocation
    /// that is at least `mid * size_of::<T>()` bytes long. Not upholding these
    /// requirements is *[undefined behavior]* even if the resulting pointers are not used.
    ///
    /// [in-bounds]: #method.add
    /// [out-of-bounds index]: #method.add
    /// [undefined behavior]: https://doc.rust-lang.org/reference/behavior-considered-undefined.html
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(raw_slice_split)]
    ///
    /// let mut v = [1, 0, 3, 0, 5, 6];
    /// // scoped to restrict the lifetime of the borrows
    /// unsafe {
    ///     let ptr = &mut v as *mut [_];
    ///     let (left, right) = ptr.split_at_mut_unchecked(2);
    ///     assert_eq!(&*left, [1, 0]);
    ///     assert_eq!(&*right, [3, 0, 5, 6]);
    ///     (&mut *left)[1] = 2;
    ///     (&mut *right)[1] = 4;
    /// }
    /// assert_eq!(v, [1, 2, 3, 4, 5, 6]);
    /// ```
    #[inline(always)]
    #[unstable(feature = "raw_slice_split", issue = "95595")]
    pub unsafe fn split_at_mut_unchecked(self, mid: usize) -> (*mut [T], *mut [T]) {
        let len = self.len();
        let ptr = self.as_mut_ptr();

        // SAFETY: Caller must pass a valid pointer and an index that is in-bounds.
        let tail = unsafe { ptr.add(mid) };
        (
            crate::ptr::slice_from_raw_parts_mut(ptr, mid),
            crate::ptr::slice_from_raw_parts_mut(tail, len - mid),
        )
    }

    /// Returns a raw pointer to the slice's buffer.
    ///
    /// This is equivalent to casting `self` to `*mut T`, but more type-safe.
    ///
    /// # Examples
    ///
    /// ```rust
    /// #![feature(slice_ptr_get)]
    /// use std::ptr;
    ///
    /// let slice: *mut [i8] = ptr::slice_from_raw_parts_mut(ptr::null_mut(), 3);
    /// assert_eq!(slice.as_mut_ptr(), ptr::null_mut());
    /// ```
    #[inline(always)]
    #[unstable(feature = "slice_ptr_get", issue = "74265")]
    #[rustc_const_unstable(feature = "slice_ptr_get", issue = "74265")]
    pub const fn as_mut_ptr(self) -> *mut T {
        self as *mut T
    }

    /// Returns a raw pointer to an element or subslice, without doing bounds
    /// checking.
    ///
    /// Calling this method with an [out-of-bounds index] or when `self` is not dereferenceable
    /// is *[undefined behavior]* even if the resulting pointer is not used.
    ///
    /// [out-of-bounds index]: #method.add
    /// [undefined behavior]: https://doc.rust-lang.org/reference/behavior-considered-undefined.html
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(slice_ptr_get)]
    ///
    /// let x = &mut [1, 2, 4] as *mut [i32];
    ///
    /// unsafe {
    ///     assert_eq!(x.get_unchecked_mut(1), x.as_mut_ptr().add(1));
    /// }
    /// ```
    #[unstable(feature = "slice_ptr_get", issue = "74265")]
    #[rustc_const_unstable(feature = "const_slice_index", issue = "none")]
    #[inline(always)]
    pub const unsafe fn get_unchecked_mut<I>(self, index: I) -> *mut I::Output
    where
        I: ~const SliceIndex<[T]>,
    {
        // SAFETY: the caller ensures that `self` is dereferenceable and `index` in-bounds.
        unsafe { index.get_unchecked_mut(self) }
    }

    /// Returns `None` if the pointer is null, or else returns a shared slice to
    /// the value wrapped in `Some`. In contrast to [`as_ref`], this does not require
    /// that the value has to be initialized.
    ///
    /// For the mutable counterpart see [`as_uninit_slice_mut`].
    ///
    /// [`as_ref`]: #method.as_ref-1
    /// [`as_uninit_slice_mut`]: #method.as_uninit_slice_mut
    ///
    /// # Safety
    ///
    /// When calling this method, you have to ensure that *either* the pointer is null *or*
    /// all of the following is true:
    ///
    /// * The pointer must be [valid] for reads for `ptr.len() * mem::size_of::<T>()` many bytes,
    ///   and it must be properly aligned. This means in particular:
    ///
    ///     * The entire memory range of this slice must be contained within a single [allocated object]!
    ///       Slices can never span across multiple allocated objects.
    ///
    ///     * The pointer must be aligned even for zero-length slices. One
    ///       reason for this is that enum layout optimizations may rely on references
    ///       (including slices of any length) being aligned and non-null to distinguish
    ///       them from other data. You can obtain a pointer that is usable as `data`
    ///       for zero-length slices using [`NonNull::dangling()`].
    ///
    /// * The total size `ptr.len() * mem::size_of::<T>()` of the slice must be no larger than `isize::MAX`.
    ///   See the safety documentation of [`pointer::offset`].
    ///
    /// * You must enforce Rust's aliasing rules, since the returned lifetime `'a` is
    ///   arbitrarily chosen and does not necessarily reflect the actual lifetime of the data.
    ///   In particular, while this reference exists, the memory the pointer points to must
    ///   not get mutated (except inside `UnsafeCell`).
    ///
    /// This applies even if the result of this method is unused!
    ///
    /// See also [`slice::from_raw_parts`][].
    ///
    /// [valid]: crate::ptr#safety
    /// [allocated object]: crate::ptr#allocated-object
    #[inline]
    #[unstable(feature = "ptr_as_uninit", issue = "75402")]
    #[rustc_const_unstable(feature = "const_ptr_as_ref", issue = "91822")]
    pub const unsafe fn as_uninit_slice<'a>(self) -> Option<&'a [MaybeUninit<T>]> {
        if self.is_null() {
            None
        } else {
            // SAFETY: the caller must uphold the safety contract for `as_uninit_slice`.
            Some(unsafe { slice::from_raw_parts(self as *const MaybeUninit<T>, self.len()) })
        }
    }

    /// Returns `None` if the pointer is null, or else returns a unique slice to
    /// the value wrapped in `Some`. In contrast to [`as_mut`], this does not require
    /// that the value has to be initialized.
    ///
    /// For the shared counterpart see [`as_uninit_slice`].
    ///
    /// [`as_mut`]: #method.as_mut
    /// [`as_uninit_slice`]: #method.as_uninit_slice-1
    ///
    /// # Safety
    ///
    /// When calling this method, you have to ensure that *either* the pointer is null *or*
    /// all of the following is true:
    ///
    /// * The pointer must be [valid] for reads and writes for `ptr.len() * mem::size_of::<T>()`
    ///   many bytes, and it must be properly aligned. This means in particular:
    ///
    ///     * The entire memory range of this slice must be contained within a single [allocated object]!
    ///       Slices can never span across multiple allocated objects.
    ///
    ///     * The pointer must be aligned even for zero-length slices. One
    ///       reason for this is that enum layout optimizations may rely on references
    ///       (including slices of any length) being aligned and non-null to distinguish
    ///       them from other data. You can obtain a pointer that is usable as `data`
    ///       for zero-length slices using [`NonNull::dangling()`].
    ///
    /// * The total size `ptr.len() * mem::size_of::<T>()` of the slice must be no larger than `isize::MAX`.
    ///   See the safety documentation of [`pointer::offset`].
    ///
    /// * You must enforce Rust's aliasing rules, since the returned lifetime `'a` is
    ///   arbitrarily chosen and does not necessarily reflect the actual lifetime of the data.
    ///   In particular, while this reference exists, the memory the pointer points to must
    ///   not get accessed (read or written) through any other pointer.
    ///
    /// This applies even if the result of this method is unused!
    ///
    /// See also [`slice::from_raw_parts_mut`][].
    ///
    /// [valid]: crate::ptr#safety
    /// [allocated object]: crate::ptr#allocated-object
    #[inline]
    #[unstable(feature = "ptr_as_uninit", issue = "75402")]
    #[rustc_const_unstable(feature = "const_ptr_as_ref", issue = "91822")]
    pub const unsafe fn as_uninit_slice_mut<'a>(self) -> Option<&'a mut [MaybeUninit<T>]> {
        if self.is_null() {
            None
        } else {
            // SAFETY: the caller must uphold the safety contract for `as_uninit_slice_mut`.
            Some(unsafe { slice::from_raw_parts_mut(self as *mut MaybeUninit<T>, self.len()) })
        }
    }
}

// Equality for pointers
#[stable(feature = "rust1", since = "1.0.0")]
impl<T: ?Sized> PartialEq for *mut T {
    #[inline(always)]
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
    #[inline(always)]
    fn partial_cmp(&self, other: &*mut T) -> Option<Ordering> {
        Some(self.cmp(other))
    }

    #[inline(always)]
    fn lt(&self, other: &*mut T) -> bool {
        *self < *other
    }

    #[inline(always)]
    fn le(&self, other: &*mut T) -> bool {
        *self <= *other
    }

    #[inline(always)]
    fn gt(&self, other: &*mut T) -> bool {
        *self > *other
    }

    #[inline(always)]
    fn ge(&self, other: &*mut T) -> bool {
        *self >= *other
    }
}
