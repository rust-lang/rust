use super::*;
use crate::cmp::Ordering::{self, Equal, Greater, Less};
use crate::intrinsics::{self, const_eval_select};
use crate::mem;
use crate::slice::{self, SliceIndex};

impl<T: ?Sized> *const T {
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
    /// ```
    /// let s: &str = "Follow the rabbit";
    /// let ptr: *const u8 = s.as_ptr();
    /// assert!(!ptr.is_null());
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[rustc_const_unstable(feature = "const_ptr_is_null", issue = "74939")]
    #[inline]
    pub const fn is_null(self) -> bool {
        #[inline]
        fn runtime_impl(ptr: *const u8) -> bool {
            ptr.addr() == 0
        }

        #[inline]
        const fn const_impl(ptr: *const u8) -> bool {
            // Compare via a cast to a thin pointer, so fat pointers are only
            // considering their "data" part for null-ness.
            match (ptr).guaranteed_eq(null_mut()) {
                None => false,
                Some(res) => res,
            }
        }

        // SAFETY: The two versions are equivalent at runtime.
        unsafe { const_eval_select((self as *const u8,), const_impl, runtime_impl) }
    }

    /// Casts to a pointer of another type.
    #[stable(feature = "ptr_cast", since = "1.38.0")]
    #[rustc_const_stable(feature = "const_ptr_cast", since = "1.38.0")]
    #[inline(always)]
    pub const fn cast<U>(self) -> *const U {
        self as _
    }

    /// Use the pointer value in a new pointer of another type.
    ///
    /// In case `meta` is a (fat) pointer to an unsized type, this operation
    /// will ignore the pointer part, whereas for (thin) pointers to sized
    /// types, this has the same effect as a simple cast.
    ///
    /// The resulting pointer will have provenance of `self`, i.e., for a fat
    /// pointer, this operation is semantically the same as creating a new
    /// fat pointer with the data pointer value of `self` but the metadata of
    /// `meta`.
    ///
    /// # Examples
    ///
    /// This function is primarily useful for allowing byte-wise pointer
    /// arithmetic on potentially fat pointers:
    ///
    /// ```
    /// #![feature(set_ptr_value)]
    /// # use core::fmt::Debug;
    /// let arr: [i32; 3] = [1, 2, 3];
    /// let mut ptr = arr.as_ptr() as *const dyn Debug;
    /// let thin = ptr as *const u8;
    /// unsafe {
    ///     ptr = thin.add(8).with_metadata_of(ptr);
    ///     # assert_eq!(*(ptr as *const i32), 3);
    ///     println!("{:?}", &*ptr); // will print "3"
    /// }
    /// ```
    #[unstable(feature = "set_ptr_value", issue = "75091")]
    #[rustc_const_unstable(feature = "set_ptr_value", issue = "75091")]
    #[must_use = "returns a new pointer rather than modifying its argument"]
    #[inline]
    pub const fn with_metadata_of<U>(self, meta: *const U) -> *const U
    where
        U: ?Sized,
    {
        from_raw_parts::<U>(self as *const (), metadata(meta))
    }

    /// Changes constness without changing the type.
    ///
    /// This is a bit safer than `as` because it wouldn't silently change the type if the code is
    /// refactored.
    #[stable(feature = "ptr_const_cast", since = "1.65.0")]
    #[rustc_const_stable(feature = "ptr_const_cast", since = "1.65.0")]
    #[inline(always)]
    pub const fn cast_mut(self) -> *mut T {
        self as _
    }

    /// Casts a pointer to its raw bits.
    ///
    /// This is equivalent to `as usize`, but is more specific to enhance readability.
    /// The inverse method is [`from_bits`](#method.from_bits).
    ///
    /// In particular, `*p as usize` and `p as usize` will both compile for
    /// pointers to numeric types but do very different things, so using this
    /// helps emphasize that reading the bits was intentional.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(ptr_to_from_bits)]
    /// # #[cfg(not(miri))] { // doctest does not work with strict provenance
    /// let array = [13, 42];
    /// let p0: *const i32 = &array[0];
    /// assert_eq!(<*const _>::from_bits(p0.to_bits()), p0);
    /// let p1: *const i32 = &array[1];
    /// assert_eq!(p1.to_bits() - p0.to_bits(), 4);
    /// # }
    /// ```
    #[unstable(feature = "ptr_to_from_bits", issue = "91126")]
    #[deprecated(
        since = "1.67",
        note = "replaced by the `exposed_addr` method, or update your code \
            to follow the strict provenance rules using its APIs"
    )]
    #[inline(always)]
    pub fn to_bits(self) -> usize
    where
        T: Sized,
    {
        self as usize
    }

    /// Creates a pointer from its raw bits.
    ///
    /// This is equivalent to `as *const T`, but is more specific to enhance readability.
    /// The inverse method is [`to_bits`](#method.to_bits).
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(ptr_to_from_bits)]
    /// # #[cfg(not(miri))] { // doctest does not work with strict provenance
    /// use std::ptr::NonNull;
    /// let dangling: *const u8 = NonNull::dangling().as_ptr();
    /// assert_eq!(<*const u8>::from_bits(1), dangling);
    /// # }
    /// ```
    #[unstable(feature = "ptr_to_from_bits", issue = "91126")]
    #[deprecated(
        since = "1.67",
        note = "replaced by the `ptr::from_exposed_addr` function, or update \
            your code to follow the strict provenance rules using its APIs"
    )]
    #[allow(fuzzy_provenance_casts)] // this is an unstable and semi-deprecated cast function
    #[inline(always)]
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
    /// properly restore the lost information and obtain a dereferenceable pointer, use
    /// [`with_addr`][pointer::with_addr] or [`map_addr`][pointer::map_addr].
    ///
    /// If using those APIs is not possible because there is no way to preserve a pointer with the
    /// required provenance, use [`expose_addr`][pointer::expose_addr] and
    /// [`from_exposed_addr`][from_exposed_addr] instead. However, note that this makes
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
    #[inline(always)]
    #[unstable(feature = "strict_provenance", issue = "95228")]
    pub fn addr(self) -> usize {
        // FIXME(strict_provenance_magic): I am magic and should be a compiler intrinsic.
        // SAFETY: Pointer-to-integer transmutes are valid (if you are okay with losing the
        // provenance).
        unsafe { mem::transmute(self.cast::<()>()) }
    }

    /// Gets the "address" portion of the pointer, and 'exposes' the "provenance" part for future
    /// use in [`from_exposed_addr`][].
    ///
    /// This is equivalent to `self as usize`, which semantically discards *provenance* and
    /// *address-space* information. Furthermore, this (like the `as` cast) has the implicit
    /// side-effect of marking the provenance as 'exposed', so on platforms that support it you can
    /// later call [`from_exposed_addr`][] to reconstitute the original pointer including its
    /// provenance. (Reconstructing address space information, if required, is your responsibility.)
    ///
    /// Using this method means that code is *not* following Strict Provenance rules. Supporting
    /// [`from_exposed_addr`][] complicates specification and reasoning and may not be supported by
    /// tools that help you to stay conformant with the Rust memory model, so it is recommended to
    /// use [`addr`][pointer::addr] wherever possible.
    ///
    /// On most platforms this will produce a value with the same bytes as the original pointer,
    /// because all the bytes are dedicated to describing the address. Platforms which need to store
    /// additional information in the pointer may not support this operation, since the 'expose'
    /// side-effect which is required for [`from_exposed_addr`][] to work is typically not
    /// available.
    ///
    /// This API and its claimed semantics are part of the Strict Provenance experiment, see the
    /// [module documentation][crate::ptr] for details.
    ///
    /// [`from_exposed_addr`]: from_exposed_addr
    #[must_use]
    #[inline(always)]
    #[unstable(feature = "strict_provenance", issue = "95228")]
    pub fn expose_addr(self) -> usize {
        // FIXME(strict_provenance_magic): I am magic and should be a compiler intrinsic.
        self.cast::<()>() as usize
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
    pub fn with_addr(self, addr: usize) -> Self {
        // FIXME(strict_provenance_magic): I am magic and should be a compiler intrinsic.
        //
        // In the mean-time, this operation is defined to be "as if" it was
        // a wrapping_offset, so we can emulate it as such. This should properly
        // restore pointer provenance even under today's compiler.
        let self_addr = self.addr() as isize;
        let dest_addr = addr as isize;
        let offset = dest_addr.wrapping_sub(self_addr);

        // This is the canonical desugarring of this operation
        self.wrapping_byte_offset(offset)
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
    pub fn map_addr(self, f: impl FnOnce(usize) -> usize) -> Self {
        self.with_addr(f(self.addr()))
    }

    /// Decompose a (possibly wide) pointer into its address and metadata components.
    ///
    /// The pointer can be later reconstructed with [`from_raw_parts`].
    #[unstable(feature = "ptr_metadata", issue = "81513")]
    #[rustc_const_unstable(feature = "ptr_metadata", issue = "81513")]
    #[inline]
    pub const fn to_raw_parts(self) -> (*const (), <T as super::Pointee>::Metadata) {
        (self.cast(), metadata(self))
    }

    /// Returns `None` if the pointer is null, or else returns a shared reference to
    /// the value wrapped in `Some`. If the value may be uninitialized, [`as_uninit_ref`]
    /// must be used instead.
    ///
    /// [`as_uninit_ref`]: #method.as_uninit_ref
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
    /// ```
    /// let ptr: *const u8 = &10u8 as *const u8;
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
    /// let ptr: *const u8 = &10u8 as *const u8;
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
        // SAFETY: the caller must guarantee that `self` is valid
        // for a reference if it isn't null.
        if self.is_null() { None } else { unsafe { Some(&*self) } }
    }

    /// Returns `None` if the pointer is null, or else returns a shared reference to
    /// the value wrapped in `Some`. In contrast to [`as_ref`], this does not require
    /// that the value has to be initialized.
    ///
    /// [`as_ref`]: #method.as_ref
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
    /// ```
    /// #![feature(ptr_as_uninit)]
    ///
    /// let ptr: *const u8 = &10u8 as *const u8;
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
    /// ```
    /// let s: &str = "123";
    /// let ptr: *const u8 = s.as_ptr();
    ///
    /// unsafe {
    ///     println!("{}", *ptr.offset(1) as char);
    ///     println!("{}", *ptr.offset(2) as char);
    /// }
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[must_use = "returns a new pointer rather than modifying its argument"]
    #[rustc_const_stable(feature = "const_ptr_offset", since = "1.61.0")]
    #[inline(always)]
    #[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
    pub const unsafe fn offset(self, count: isize) -> *const T
    where
        T: Sized,
    {
        // SAFETY: the caller must uphold the safety contract for `offset`.
        unsafe { intrinsics::offset(self, count) }
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
        unsafe { self.cast::<u8>().offset(count).with_metadata_of(self) }
    }

    /// Calculates the offset from a pointer using wrapping arithmetic.
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
    /// ```
    /// // Iterate using a raw pointer in increments of two elements
    /// let data = [1u8, 2, 3, 4, 5];
    /// let mut ptr: *const u8 = data.as_ptr();
    /// let step = 2;
    /// let end_rounded_up = ptr.wrapping_offset(6);
    ///
    /// // This loop prints "1, 3, 5, "
    /// while ptr != end_rounded_up {
    ///     unsafe {
    ///         print!("{}, ", *ptr);
    ///     }
    ///     ptr = ptr.wrapping_offset(step);
    /// }
    /// ```
    #[stable(feature = "ptr_wrapping_offset", since = "1.16.0")]
    #[must_use = "returns a new pointer rather than modifying its argument"]
    #[rustc_const_stable(feature = "const_ptr_offset", since = "1.61.0")]
    #[inline(always)]
    pub const fn wrapping_offset(self, count: isize) -> *const T
    where
        T: Sized,
    {
        // SAFETY: the `arith_offset` intrinsic has no prerequisites to be called.
        unsafe { intrinsics::arith_offset(self, count) }
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
        self.cast::<u8>().wrapping_offset(count).with_metadata_of(self)
    }

    /// Masks out bits of the pointer according to a mask.
    ///
    /// This is convenience for `ptr.map_addr(|a| a & mask)`.
    ///
    /// For non-`Sized` pointees this operation changes only the data pointer,
    /// leaving the metadata untouched.
    ///
    /// ## Examples
    ///
    /// ```
    /// #![feature(ptr_mask, strict_provenance)]
    /// let v = 17_u32;
    /// let ptr: *const u32 = &v;
    ///
    /// // `u32` is 4 bytes aligned,
    /// // which means that lower 2 bits are always 0.
    /// let tag_mask = 0b11;
    /// let ptr_mask = !tag_mask;
    ///
    /// // We can store something in these lower bits
    /// let tagged_ptr = ptr.map_addr(|a| a | 0b10);
    ///
    /// // Get the "tag" back
    /// let tag = tagged_ptr.addr() & tag_mask;
    /// assert_eq!(tag, 0b10);
    ///
    /// // Note that `tagged_ptr` is unaligned, it's UB to read from it.
    /// // To get original pointer `mask` can be used:
    /// let masked_ptr = tagged_ptr.mask(ptr_mask);
    /// assert_eq!(unsafe { *masked_ptr }, 17);
    /// ```
    #[unstable(feature = "ptr_mask", issue = "98290")]
    #[must_use = "returns a new pointer rather than modifying its argument"]
    #[inline(always)]
    pub fn mask(self, mask: usize) -> *const T {
        intrinsics::ptr_mask(self.cast::<()>(), mask).with_metadata_of(self)
    }

    /// Calculates the distance between two pointers. The returned value is in
    /// units of T: the distance in bytes divided by `mem::size_of::<T>()`.
    ///
    /// This function is the inverse of [`offset`].
    ///
    /// [`offset`]: #method.offset
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
    /// let a = [0; 5];
    /// let ptr1: *const i32 = &a[1];
    /// let ptr2: *const i32 = &a[3];
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
    /// let ptr1 = Box::into_raw(Box::new(0u8)) as *const u8;
    /// let ptr2 = Box::into_raw(Box::new(1u8)) as *const u8;
    /// let diff = (ptr2 as isize).wrapping_sub(ptr1 as isize);
    /// // Make ptr2_other an "alias" of ptr2, but derived from ptr1.
    /// let ptr2_other = (ptr1 as *const u8).wrapping_offset(diff);
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
    #[inline]
    #[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
    pub const unsafe fn offset_from(self, origin: *const T) -> isize
    where
        T: Sized,
    {
        let pointee_size = mem::size_of::<T>();
        assert!(0 < pointee_size && pointee_size <= isize::MAX as usize);
        // SAFETY: the caller must uphold the safety contract for `ptr_offset_from`.
        unsafe { intrinsics::ptr_offset_from(self, origin) }
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
    pub const unsafe fn byte_offset_from<U: ?Sized>(self, origin: *const U) -> isize {
        // SAFETY: the caller must uphold the safety contract for `offset_from`.
        unsafe { self.cast::<u8>().offset_from(origin.cast::<u8>()) }
    }

    /// Calculates the distance between two pointers, *where it's known that
    /// `self` is equal to or greater than `origin`*. The returned value is in
    /// units of T: the distance in bytes is divided by `mem::size_of::<T>()`.
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
    /// # #![feature(ptr_sub_ptr)]
    /// # unsafe fn blah(ptr: *const i32, origin: *const i32, count: usize) -> bool {
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
    /// let a = [0; 5];
    /// let ptr1: *const i32 = &a[1];
    /// let ptr2: *const i32 = &a[3];
    /// unsafe {
    ///     assert_eq!(ptr2.sub_ptr(ptr1), 2);
    ///     assert_eq!(ptr1.add(2), ptr2);
    ///     assert_eq!(ptr2.sub(2), ptr1);
    ///     assert_eq!(ptr2.sub_ptr(ptr2), 0);
    /// }
    ///
    /// // This would be incorrect, as the pointers are not correctly ordered:
    /// // ptr1.sub_ptr(ptr2)
    /// ```
    #[unstable(feature = "ptr_sub_ptr", issue = "95892")]
    #[rustc_const_unstable(feature = "const_ptr_sub_ptr", issue = "95892")]
    #[inline]
    #[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
    pub const unsafe fn sub_ptr(self, origin: *const T) -> usize
    where
        T: Sized,
    {
        let this = self;
        // SAFETY: The comparison has no side-effects, and the intrinsic
        // does this check internally in the CTFE implementation.
        unsafe {
            assert_unsafe_precondition!(
                "ptr::sub_ptr requires `this >= origin`",
                [T](this: *const T, origin: *const T) => this >= origin
            )
        };

        let pointee_size = mem::size_of::<T>();
        assert!(0 < pointee_size && pointee_size <= isize::MAX as usize);
        // SAFETY: the caller must uphold the safety contract for `ptr_offset_from_unsigned`.
        unsafe { intrinsics::ptr_offset_from_unsigned(self, origin) }
    }

    /// Returns whether two pointers are guaranteed to be equal.
    ///
    /// At runtime this function behaves like `Some(self == other)`.
    /// However, in some contexts (e.g., compile-time evaluation),
    /// it is not always possible to determine equality of two pointers, so this function may
    /// spuriously return `None` for pointers that later actually turn out to have its equality known.
    /// But when it returns `Some`, the pointers' equality is guaranteed to be known.
    ///
    /// The return value may change from `Some` to `None` and vice versa depending on the compiler
    /// version and unsafe code must not
    /// rely on the result of this function for soundness. It is suggested to only use this function
    /// for performance optimizations where spurious `None` return values by this function do not
    /// affect the outcome, but just the performance.
    /// The consequences of using this method to make runtime and compile-time code behave
    /// differently have not been explored. This method should not be used to introduce such
    /// differences, and it should also not be stabilized before we have a better understanding
    /// of this issue.
    #[unstable(feature = "const_raw_ptr_comparison", issue = "53020")]
    #[rustc_const_unstable(feature = "const_raw_ptr_comparison", issue = "53020")]
    #[inline]
    pub const fn guaranteed_eq(self, other: *const T) -> Option<bool>
    where
        T: Sized,
    {
        match intrinsics::ptr_guaranteed_cmp(self as _, other as _) {
            2 => None,
            other => Some(other == 1),
        }
    }

    /// Returns whether two pointers are guaranteed to be inequal.
    ///
    /// At runtime this function behaves like `Some(self != other)`.
    /// However, in some contexts (e.g., compile-time evaluation),
    /// it is not always possible to determine inequality of two pointers, so this function may
    /// spuriously return `None` for pointers that later actually turn out to have its inequality known.
    /// But when it returns `Some`, the pointers' inequality is guaranteed to be known.
    ///
    /// The return value may change from `Some` to `None` and vice versa depending on the compiler
    /// version and unsafe code must not
    /// rely on the result of this function for soundness. It is suggested to only use this function
    /// for performance optimizations where spurious `None` return values by this function do not
    /// affect the outcome, but just the performance.
    /// The consequences of using this method to make runtime and compile-time code behave
    /// differently have not been explored. This method should not be used to introduce such
    /// differences, and it should also not be stabilized before we have a better understanding
    /// of this issue.
    #[unstable(feature = "const_raw_ptr_comparison", issue = "53020")]
    #[rustc_const_unstable(feature = "const_raw_ptr_comparison", issue = "53020")]
    #[inline]
    pub const fn guaranteed_ne(self, other: *const T) -> Option<bool>
    where
        T: Sized,
    {
        match self.guaranteed_eq(other) {
            None => None,
            Some(eq) => Some(!eq),
        }
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
        unsafe { self.cast::<u8>().add(count).with_metadata_of(self) }
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
    #[inline(always)]
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
        unsafe { self.cast::<u8>().sub(count).with_metadata_of(self) }
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
        self.cast::<u8>().wrapping_add(count).with_metadata_of(self)
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
    #[inline(always)]
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
        self.cast::<u8>().wrapping_sub(count).with_metadata_of(self)
    }

    /// Reads the value from `self` without moving it. This leaves the
    /// memory in `self` unchanged.
    ///
    /// See [`ptr::read`] for safety concerns and examples.
    ///
    /// [`ptr::read`]: crate::ptr::read()
    #[stable(feature = "pointer_methods", since = "1.26.0")]
    #[rustc_const_unstable(feature = "const_ptr_read", issue = "80377")]
    #[inline]
    #[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
    pub const unsafe fn read(self) -> T
    where
        T: Sized,
    {
        // SAFETY: the caller must uphold the safety contract for `read`.
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
    #[inline]
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
    #[inline]
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
    #[inline]
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
    #[inline]
    #[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
    pub const unsafe fn copy_to_nonoverlapping(self, dest: *mut T, count: usize)
    where
        T: Sized,
    {
        // SAFETY: the caller must uphold the safety contract for `copy_nonoverlapping`.
        unsafe { copy_nonoverlapping(self, dest, count) }
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
    /// let x = [5_u8, 6, 7, 8, 9];
    /// let ptr = x.as_ptr();
    /// let offset = ptr.align_offset(align_of::<u16>());
    ///
    /// if offset < x.len() - 1 {
    ///     let u16_ptr = ptr.add(offset).cast::<u16>();
    ///     assert!(*u16_ptr == u16::from_ne_bytes([5, 6]) || *u16_ptr == u16::from_ne_bytes([6, 7]));
    /// } else {
    ///     // while the pointer can be aligned via `offset`, it would point
    ///     // outside the allocation
    /// }
    /// # }
    /// ```
    #[must_use]
    #[inline]
    #[stable(feature = "align_offset", since = "1.36.0")]
    #[rustc_const_unstable(feature = "const_align_offset", issue = "90962")]
    pub const fn align_offset(self, align: usize) -> usize
    where
        T: Sized,
    {
        if !align.is_power_of_two() {
            panic!("align_offset: align is not a power-of-two");
        }

        {
            // SAFETY: `align` has been checked to be a power of 2 above
            unsafe { align_offset(self, align) }
        }
    }

    /// Returns whether the pointer is properly aligned for `T`.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(pointer_is_aligned)]
    /// #![feature(pointer_byte_offsets)]
    ///
    /// // On some platforms, the alignment of i32 is less than 4.
    /// #[repr(align(4))]
    /// struct AlignedI32(i32);
    ///
    /// let data = AlignedI32(42);
    /// let ptr = &data as *const AlignedI32;
    ///
    /// assert!(ptr.is_aligned());
    /// assert!(!ptr.wrapping_byte_add(1).is_aligned());
    /// ```
    ///
    /// # At compiletime
    /// **Note: Alignment at compiletime is experimental and subject to change. See the
    /// [tracking issue] for details.**
    ///
    /// At compiletime, the compiler may not know where a value will end up in memory.
    /// Calling this function on a pointer created from a reference at compiletime will only
    /// return `true` if the pointer is guaranteed to be aligned. This means that the pointer
    /// is never aligned if cast to a type with a stricter alignment than the reference's
    /// underlying allocation.
    ///
    /// ```
    /// #![feature(pointer_is_aligned)]
    /// #![feature(const_pointer_is_aligned)]
    ///
    /// // On some platforms, the alignment of primitives is less than their size.
    /// #[repr(align(4))]
    /// struct AlignedI32(i32);
    /// #[repr(align(8))]
    /// struct AlignedI64(i64);
    ///
    /// const _: () = {
    ///     let data = AlignedI32(42);
    ///     let ptr = &data as *const AlignedI32;
    ///     assert!(ptr.is_aligned());
    ///
    ///     // At runtime either `ptr1` or `ptr2` would be aligned, but at compiletime neither is aligned.
    ///     let ptr1 = ptr.cast::<AlignedI64>();
    ///     let ptr2 = ptr.wrapping_add(1).cast::<AlignedI64>();
    ///     assert!(!ptr1.is_aligned());
    ///     assert!(!ptr2.is_aligned());
    /// };
    /// ```
    ///
    /// Due to this behavior, it is possible that a runtime pointer derived from a compiletime
    /// pointer is aligned, even if the compiletime pointer wasn't aligned.
    ///
    /// ```
    /// #![feature(pointer_is_aligned)]
    /// #![feature(const_pointer_is_aligned)]
    ///
    /// // On some platforms, the alignment of primitives is less than their size.
    /// #[repr(align(4))]
    /// struct AlignedI32(i32);
    /// #[repr(align(8))]
    /// struct AlignedI64(i64);
    ///
    /// // At compiletime, neither `COMPTIME_PTR` nor `COMPTIME_PTR + 1` is aligned.
    /// const COMPTIME_PTR: *const AlignedI32 = &AlignedI32(42);
    /// const _: () = assert!(!COMPTIME_PTR.cast::<AlignedI64>().is_aligned());
    /// const _: () = assert!(!COMPTIME_PTR.wrapping_add(1).cast::<AlignedI64>().is_aligned());
    ///
    /// // At runtime, either `runtime_ptr` or `runtime_ptr + 1` is aligned.
    /// let runtime_ptr = COMPTIME_PTR;
    /// assert_ne!(
    ///     runtime_ptr.cast::<AlignedI64>().is_aligned(),
    ///     runtime_ptr.wrapping_add(1).cast::<AlignedI64>().is_aligned(),
    /// );
    /// ```
    ///
    /// If a pointer is created from a fixed address, this function behaves the same during
    /// runtime and compiletime.
    ///
    /// ```
    /// #![feature(pointer_is_aligned)]
    /// #![feature(const_pointer_is_aligned)]
    ///
    /// // On some platforms, the alignment of primitives is less than their size.
    /// #[repr(align(4))]
    /// struct AlignedI32(i32);
    /// #[repr(align(8))]
    /// struct AlignedI64(i64);
    ///
    /// const _: () = {
    ///     let ptr = 40 as *const AlignedI32;
    ///     assert!(ptr.is_aligned());
    ///
    ///     // For pointers with a known address, runtime and compiletime behavior are identical.
    ///     let ptr1 = ptr.cast::<AlignedI64>();
    ///     let ptr2 = ptr.wrapping_add(1).cast::<AlignedI64>();
    ///     assert!(ptr1.is_aligned());
    ///     assert!(!ptr2.is_aligned());
    /// };
    /// ```
    ///
    /// [tracking issue]: https://github.com/rust-lang/rust/issues/104203
    #[must_use]
    #[inline]
    #[unstable(feature = "pointer_is_aligned", issue = "96284")]
    #[rustc_const_unstable(feature = "const_pointer_is_aligned", issue = "104203")]
    pub const fn is_aligned(self) -> bool
    where
        T: Sized,
    {
        self.is_aligned_to(mem::align_of::<T>())
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
    /// #![feature(pointer_is_aligned)]
    /// #![feature(pointer_byte_offsets)]
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
    ///
    /// # At compiletime
    /// **Note: Alignment at compiletime is experimental and subject to change. See the
    /// [tracking issue] for details.**
    ///
    /// At compiletime, the compiler may not know where a value will end up in memory.
    /// Calling this function on a pointer created from a reference at compiletime will only
    /// return `true` if the pointer is guaranteed to be aligned. This means that the pointer
    /// cannot be stricter aligned than the reference's underlying allocation.
    ///
    /// ```
    /// #![feature(pointer_is_aligned)]
    /// #![feature(const_pointer_is_aligned)]
    ///
    /// // On some platforms, the alignment of i32 is less than 4.
    /// #[repr(align(4))]
    /// struct AlignedI32(i32);
    ///
    /// const _: () = {
    ///     let data = AlignedI32(42);
    ///     let ptr = &data as *const AlignedI32;
    ///
    ///     assert!(ptr.is_aligned_to(1));
    ///     assert!(ptr.is_aligned_to(2));
    ///     assert!(ptr.is_aligned_to(4));
    ///
    ///     // At compiletime, we know for sure that the pointer isn't aligned to 8.
    ///     assert!(!ptr.is_aligned_to(8));
    ///     assert!(!ptr.wrapping_add(1).is_aligned_to(8));
    /// };
    /// ```
    ///
    /// Due to this behavior, it is possible that a runtime pointer derived from a compiletime
    /// pointer is aligned, even if the compiletime pointer wasn't aligned.
    ///
    /// ```
    /// #![feature(pointer_is_aligned)]
    /// #![feature(const_pointer_is_aligned)]
    ///
    /// // On some platforms, the alignment of i32 is less than 4.
    /// #[repr(align(4))]
    /// struct AlignedI32(i32);
    ///
    /// // At compiletime, neither `COMPTIME_PTR` nor `COMPTIME_PTR + 1` is aligned.
    /// const COMPTIME_PTR: *const AlignedI32 = &AlignedI32(42);
    /// const _: () = assert!(!COMPTIME_PTR.is_aligned_to(8));
    /// const _: () = assert!(!COMPTIME_PTR.wrapping_add(1).is_aligned_to(8));
    ///
    /// // At runtime, either `runtime_ptr` or `runtime_ptr + 1` is aligned.
    /// let runtime_ptr = COMPTIME_PTR;
    /// assert_ne!(
    ///     runtime_ptr.is_aligned_to(8),
    ///     runtime_ptr.wrapping_add(1).is_aligned_to(8),
    /// );
    /// ```
    ///
    /// If a pointer is created from a fixed address, this function behaves the same during
    /// runtime and compiletime.
    ///
    /// ```
    /// #![feature(pointer_is_aligned)]
    /// #![feature(const_pointer_is_aligned)]
    ///
    /// const _: () = {
    ///     let ptr = 40 as *const u8;
    ///     assert!(ptr.is_aligned_to(1));
    ///     assert!(ptr.is_aligned_to(2));
    ///     assert!(ptr.is_aligned_to(4));
    ///     assert!(ptr.is_aligned_to(8));
    ///     assert!(!ptr.is_aligned_to(16));
    /// };
    /// ```
    ///
    /// [tracking issue]: https://github.com/rust-lang/rust/issues/104203
    #[must_use]
    #[inline]
    #[unstable(feature = "pointer_is_aligned", issue = "96284")]
    #[rustc_const_unstable(feature = "const_pointer_is_aligned", issue = "104203")]
    pub const fn is_aligned_to(self, align: usize) -> bool {
        if !align.is_power_of_two() {
            panic!("is_aligned_to: align is not a power-of-two");
        }

        #[inline]
        fn runtime_impl(ptr: *const (), align: usize) -> bool {
            ptr.addr() & (align - 1) == 0
        }

        #[inline]
        const fn const_impl(ptr: *const (), align: usize) -> bool {
            // We can't use the address of `self` in a `const fn`, so we use `align_offset` instead.
            // The cast to `()` is used to
            //   1. deal with fat pointers; and
            //   2. ensure that `align_offset` doesn't actually try to compute an offset.
            ptr.align_offset(align) == 0
        }

        // SAFETY: The two versions are equivalent at runtime.
        unsafe { const_eval_select((self.cast::<()>(), align), const_impl, runtime_impl) }
    }
}

impl<T> *const [T] {
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
    /// let slice: *const [i8] = ptr::slice_from_raw_parts(ptr::null(), 3);
    /// assert_eq!(slice.len(), 3);
    /// ```
    #[inline]
    #[unstable(feature = "slice_ptr_len", issue = "71146")]
    #[rustc_const_unstable(feature = "const_slice_ptr_len", issue = "71146")]
    pub const fn len(self) -> usize {
        metadata(self)
    }

    /// Returns a raw pointer to the slice's buffer.
    ///
    /// This is equivalent to casting `self` to `*const T`, but more type-safe.
    ///
    /// # Examples
    ///
    /// ```rust
    /// #![feature(slice_ptr_get)]
    /// use std::ptr;
    ///
    /// let slice: *const [i8] = ptr::slice_from_raw_parts(ptr::null(), 3);
    /// assert_eq!(slice.as_ptr(), ptr::null());
    /// ```
    #[inline]
    #[unstable(feature = "slice_ptr_get", issue = "74265")]
    #[rustc_const_unstable(feature = "slice_ptr_get", issue = "74265")]
    pub const fn as_ptr(self) -> *const T {
        self as *const T
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
    ///
    /// let x = &[1, 2, 4] as *const [i32];
    ///
    /// unsafe {
    ///     assert_eq!(x.get_unchecked(1), x.as_ptr().add(1));
    /// }
    /// ```
    #[unstable(feature = "slice_ptr_get", issue = "74265")]
    #[rustc_const_unstable(feature = "const_slice_index", issue = "none")]
    #[inline]
    pub const unsafe fn get_unchecked<I>(self, index: I) -> *const I::Output
    where
        I: ~const SliceIndex<[T]>,
    {
        // SAFETY: the caller ensures that `self` is dereferenceable and `index` in-bounds.
        unsafe { index.get_unchecked(self) }
    }

    /// Returns `None` if the pointer is null, or else returns a shared slice to
    /// the value wrapped in `Some`. In contrast to [`as_ref`], this does not require
    /// that the value has to be initialized.
    ///
    /// [`as_ref`]: #method.as_ref
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
}

// Equality for pointers
#[stable(feature = "rust1", since = "1.0.0")]
impl<T: ?Sized> PartialEq for *const T {
    #[inline]
    fn eq(&self, other: &*const T) -> bool {
        *self == *other
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: ?Sized> Eq for *const T {}

// Comparison for pointers
#[stable(feature = "rust1", since = "1.0.0")]
impl<T: ?Sized> Ord for *const T {
    #[inline]
    fn cmp(&self, other: &*const T) -> Ordering {
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
impl<T: ?Sized> PartialOrd for *const T {
    #[inline]
    fn partial_cmp(&self, other: &*const T) -> Option<Ordering> {
        Some(self.cmp(other))
    }

    #[inline]
    fn lt(&self, other: &*const T) -> bool {
        *self < *other
    }

    #[inline]
    fn le(&self, other: &*const T) -> bool {
        *self <= *other
    }

    #[inline]
    fn gt(&self, other: &*const T) -> bool {
        *self > *other
    }

    #[inline]
    fn ge(&self, other: &*const T) -> bool {
        *self >= *other
    }
}
