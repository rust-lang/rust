// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// FIXME: talk about offset, copy_memory, copy_nonoverlapping_memory

//! Raw, unsafe pointers, `*const T`, and `*mut T`.
//!
//! *[See also the pointer primitive types](../../std/primitive.pointer.html).*

#![stable(feature = "rust1", since = "1.0.0")]

use intrinsics;
use ops::CoerceUnsized;
use fmt;
use hash;
use marker::{PhantomData, Unsize};
use mem;
use nonzero::NonZero;

use cmp::Ordering::{self, Less, Equal, Greater};

// FIXME #19649: intrinsic docs don't render, so these have no docs :(

#[stable(feature = "rust1", since = "1.0.0")]
pub use intrinsics::copy_nonoverlapping;

#[stable(feature = "rust1", since = "1.0.0")]
pub use intrinsics::copy;

#[stable(feature = "rust1", since = "1.0.0")]
pub use intrinsics::write_bytes;

/// Executes the destructor (if any) of the pointed-to value.
///
/// This has two use cases:
///
/// * It is *required* to use `drop_in_place` to drop unsized types like
///   trait objects, because they can't be read out onto the stack and
///   dropped normally.
///
/// * It is friendlier to the optimizer to do this over `ptr::read` when
///   dropping manually allocated memory (e.g. when writing Box/Rc/Vec),
///   as the compiler doesn't need to prove that it's sound to elide the
///   copy.
///
/// # Undefined Behavior
///
/// This has all the same safety problems as `ptr::read` with respect to
/// invalid pointers, types, and double drops.
#[stable(feature = "drop_in_place", since = "1.8.0")]
#[lang="drop_in_place"]
#[allow(unconditional_recursion)]
pub unsafe fn drop_in_place<T: ?Sized>(to_drop: *mut T) {
    // Code here does not matter - this is replaced by the
    // real drop glue by the compiler.
    drop_in_place(to_drop);
}

/// Creates a null raw pointer.
///
/// # Examples
///
/// ```
/// use std::ptr;
///
/// let p: *const i32 = ptr::null();
/// assert!(p.is_null());
/// ```
#[inline]
#[stable(feature = "rust1", since = "1.0.0")]
pub const fn null<T>() -> *const T { 0 as *const T }

/// Creates a null mutable raw pointer.
///
/// # Examples
///
/// ```
/// use std::ptr;
///
/// let p: *mut i32 = ptr::null_mut();
/// assert!(p.is_null());
/// ```
#[inline]
#[stable(feature = "rust1", since = "1.0.0")]
pub const fn null_mut<T>() -> *mut T { 0 as *mut T }

/// Swaps the values at two mutable locations of the same type, without
/// deinitializing either. They may overlap, unlike `mem::swap` which is
/// otherwise equivalent.
///
/// # Safety
///
/// This function copies the memory through the raw pointers passed to it
/// as arguments.
///
/// Ensure that these pointers are valid before calling `swap`.
#[inline]
#[stable(feature = "rust1", since = "1.0.0")]
pub unsafe fn swap<T>(x: *mut T, y: *mut T) {
    // Give ourselves some scratch space to work with
    let mut tmp: T = mem::uninitialized();

    // Perform the swap
    copy_nonoverlapping(x, &mut tmp, 1);
    copy(y, x, 1); // `x` and `y` may overlap
    copy_nonoverlapping(&tmp, y, 1);

    // y and t now point to the same thing, but we need to completely forget `tmp`
    // because it's no longer relevant.
    mem::forget(tmp);
}

/// Swaps a sequence of values at two mutable locations of the same type.
///
/// # Safety
///
/// The two arguments must each point to the beginning of `count` locations
/// of valid memory, and the two memory ranges must not overlap.
///
/// # Examples
///
/// Basic usage:
///
/// ```
/// #![feature(swap_nonoverlapping)]
///
/// use std::ptr;
///
/// let mut x = [1, 2, 3, 4];
/// let mut y = [7, 8, 9];
///
/// unsafe {
///     ptr::swap_nonoverlapping(x.as_mut_ptr(), y.as_mut_ptr(), 2);
/// }
///
/// assert_eq!(x, [7, 8, 3, 4]);
/// assert_eq!(y, [1, 2, 9]);
/// ```
#[inline]
#[unstable(feature = "swap_nonoverlapping", issue = "42818")]
pub unsafe fn swap_nonoverlapping<T>(x: *mut T, y: *mut T, count: usize) {
    let x = x as *mut u8;
    let y = y as *mut u8;
    let len = mem::size_of::<T>() * count;
    swap_nonoverlapping_bytes(x, y, len)
}

#[inline]
unsafe fn swap_nonoverlapping_bytes(x: *mut u8, y: *mut u8, len: usize) {
    // The approach here is to utilize simd to swap x & y efficiently. Testing reveals
    // that swapping either 32 bytes or 64 bytes at a time is most efficient for intel
    // Haswell E processors. LLVM is more able to optimize if we give a struct a
    // #[repr(simd)], even if we don't actually use this struct directly.
    //
    // FIXME repr(simd) broken on emscripten and redox
    // It's also broken on big-endian powerpc64 and s390x.  #42778
    #[cfg_attr(not(any(target_os = "emscripten", target_os = "redox",
                       target_endian = "big")),
               repr(simd))]
    struct Block(u64, u64, u64, u64);
    struct UnalignedBlock(u64, u64, u64, u64);

    let block_size = mem::size_of::<Block>();

    // Loop through x & y, copying them `Block` at a time
    // The optimizer should unroll the loop fully for most types
    // N.B. We can't use a for loop as the `range` impl calls `mem::swap` recursively
    let mut i = 0;
    while i + block_size <= len {
        // Create some uninitialized memory as scratch space
        // Declaring `t` here avoids aligning the stack when this loop is unused
        let mut t: Block = mem::uninitialized();
        let t = &mut t as *mut _ as *mut u8;
        let x = x.offset(i as isize);
        let y = y.offset(i as isize);

        // Swap a block of bytes of x & y, using t as a temporary buffer
        // This should be optimized into efficient SIMD operations where available
        copy_nonoverlapping(x, t, block_size);
        copy_nonoverlapping(y, x, block_size);
        copy_nonoverlapping(t, y, block_size);
        i += block_size;
    }

    if i < len {
        // Swap any remaining bytes
        let mut t: UnalignedBlock = mem::uninitialized();
        let rem = len - i;

        let t = &mut t as *mut _ as *mut u8;
        let x = x.offset(i as isize);
        let y = y.offset(i as isize);

        copy_nonoverlapping(x, t, rem);
        copy_nonoverlapping(y, x, rem);
        copy_nonoverlapping(t, y, rem);
    }
}

/// Replaces the value at `dest` with `src`, returning the old
/// value, without dropping either.
///
/// # Safety
///
/// This is only unsafe because it accepts a raw pointer.
/// Otherwise, this operation is identical to `mem::replace`.
#[inline]
#[stable(feature = "rust1", since = "1.0.0")]
pub unsafe fn replace<T>(dest: *mut T, mut src: T) -> T {
    mem::swap(&mut *dest, &mut src); // cannot overlap
    src
}

/// Reads the value from `src` without moving it. This leaves the
/// memory in `src` unchanged.
///
/// # Safety
///
/// Beyond accepting a raw pointer, this is unsafe because it semantically
/// moves the value out of `src` without preventing further usage of `src`.
/// If `T` is not `Copy`, then care must be taken to ensure that the value at
/// `src` is not used before the data is overwritten again (e.g. with `write`,
/// `zero_memory`, or `copy_memory`). Note that `*src = foo` counts as a use
/// because it will attempt to drop the value previously at `*src`.
///
/// The pointer must be aligned; use `read_unaligned` if that is not the case.
///
/// # Examples
///
/// Basic usage:
///
/// ```
/// let x = 12;
/// let y = &x as *const i32;
///
/// unsafe {
///     assert_eq!(std::ptr::read(y), 12);
/// }
/// ```
#[inline(always)]
#[stable(feature = "rust1", since = "1.0.0")]
pub unsafe fn read<T>(src: *const T) -> T {
    let mut tmp: T = mem::uninitialized();
    copy_nonoverlapping(src, &mut tmp, 1);
    tmp
}

/// Reads the value from `src` without moving it. This leaves the
/// memory in `src` unchanged.
///
/// Unlike `read`, the pointer may be unaligned.
///
/// # Safety
///
/// Beyond accepting a raw pointer, this is unsafe because it semantically
/// moves the value out of `src` without preventing further usage of `src`.
/// If `T` is not `Copy`, then care must be taken to ensure that the value at
/// `src` is not used before the data is overwritten again (e.g. with `write`,
/// `zero_memory`, or `copy_memory`). Note that `*src = foo` counts as a use
/// because it will attempt to drop the value previously at `*src`.
///
/// # Examples
///
/// Basic usage:
///
/// ```
/// let x = 12;
/// let y = &x as *const i32;
///
/// unsafe {
///     assert_eq!(std::ptr::read_unaligned(y), 12);
/// }
/// ```
#[inline(always)]
#[stable(feature = "ptr_unaligned", since = "1.17.0")]
pub unsafe fn read_unaligned<T>(src: *const T) -> T {
    let mut tmp: T = mem::uninitialized();
    copy_nonoverlapping(src as *const u8,
                        &mut tmp as *mut T as *mut u8,
                        mem::size_of::<T>());
    tmp
}

/// Overwrites a memory location with the given value without reading or
/// dropping the old value.
///
/// # Safety
///
/// This operation is marked unsafe because it accepts a raw pointer.
///
/// It does not drop the contents of `dst`. This is safe, but it could leak
/// allocations or resources, so care must be taken not to overwrite an object
/// that should be dropped.
///
/// Additionally, it does not drop `src`. Semantically, `src` is moved into the
/// location pointed to by `dst`.
///
/// This is appropriate for initializing uninitialized memory, or overwriting
/// memory that has previously been `read` from.
///
/// The pointer must be aligned; use `write_unaligned` if that is not the case.
///
/// # Examples
///
/// Basic usage:
///
/// ```
/// let mut x = 0;
/// let y = &mut x as *mut i32;
/// let z = 12;
///
/// unsafe {
///     std::ptr::write(y, z);
///     assert_eq!(std::ptr::read(y), 12);
/// }
/// ```
#[inline]
#[stable(feature = "rust1", since = "1.0.0")]
pub unsafe fn write<T>(dst: *mut T, src: T) {
    intrinsics::move_val_init(&mut *dst, src)
}

/// Overwrites a memory location with the given value without reading or
/// dropping the old value.
///
/// Unlike `write`, the pointer may be unaligned.
///
/// # Safety
///
/// This operation is marked unsafe because it accepts a raw pointer.
///
/// It does not drop the contents of `dst`. This is safe, but it could leak
/// allocations or resources, so care must be taken not to overwrite an object
/// that should be dropped.
///
/// Additionally, it does not drop `src`. Semantically, `src` is moved into the
/// location pointed to by `dst`.
///
/// This is appropriate for initializing uninitialized memory, or overwriting
/// memory that has previously been `read` from.
///
/// # Examples
///
/// Basic usage:
///
/// ```
/// let mut x = 0;
/// let y = &mut x as *mut i32;
/// let z = 12;
///
/// unsafe {
///     std::ptr::write_unaligned(y, z);
///     assert_eq!(std::ptr::read_unaligned(y), 12);
/// }
/// ```
#[inline]
#[stable(feature = "ptr_unaligned", since = "1.17.0")]
pub unsafe fn write_unaligned<T>(dst: *mut T, src: T) {
    copy_nonoverlapping(&src as *const T as *const u8,
                        dst as *mut u8,
                        mem::size_of::<T>());
    mem::forget(src);
}

/// Performs a volatile read of the value from `src` without moving it. This
/// leaves the memory in `src` unchanged.
///
/// Volatile operations are intended to act on I/O memory, and are guaranteed
/// to not be elided or reordered by the compiler across other volatile
/// operations.
///
/// # Notes
///
/// Rust does not currently have a rigorously and formally defined memory model,
/// so the precise semantics of what "volatile" means here is subject to change
/// over time. That being said, the semantics will almost always end up pretty
/// similar to [C11's definition of volatile][c11].
///
/// [c11]: http://www.open-std.org/jtc1/sc22/wg14/www/docs/n1570.pdf
///
/// # Safety
///
/// Beyond accepting a raw pointer, this is unsafe because it semantically
/// moves the value out of `src` without preventing further usage of `src`.
/// If `T` is not `Copy`, then care must be taken to ensure that the value at
/// `src` is not used before the data is overwritten again (e.g. with `write`,
/// `zero_memory`, or `copy_memory`). Note that `*src = foo` counts as a use
/// because it will attempt to drop the value previously at `*src`.
///
/// # Examples
///
/// Basic usage:
///
/// ```
/// let x = 12;
/// let y = &x as *const i32;
///
/// unsafe {
///     assert_eq!(std::ptr::read_volatile(y), 12);
/// }
/// ```
#[inline]
#[stable(feature = "volatile", since = "1.9.0")]
pub unsafe fn read_volatile<T>(src: *const T) -> T {
    intrinsics::volatile_load(src)
}

/// Performs a volatile write of a memory location with the given value without
/// reading or dropping the old value.
///
/// Volatile operations are intended to act on I/O memory, and are guaranteed
/// to not be elided or reordered by the compiler across other volatile
/// operations.
///
/// # Notes
///
/// Rust does not currently have a rigorously and formally defined memory model,
/// so the precise semantics of what "volatile" means here is subject to change
/// over time. That being said, the semantics will almost always end up pretty
/// similar to [C11's definition of volatile][c11].
///
/// [c11]: http://www.open-std.org/jtc1/sc22/wg14/www/docs/n1570.pdf
///
/// # Safety
///
/// This operation is marked unsafe because it accepts a raw pointer.
///
/// It does not drop the contents of `dst`. This is safe, but it could leak
/// allocations or resources, so care must be taken not to overwrite an object
/// that should be dropped.
///
/// This is appropriate for initializing uninitialized memory, or overwriting
/// memory that has previously been `read` from.
///
/// # Examples
///
/// Basic usage:
///
/// ```
/// let mut x = 0;
/// let y = &mut x as *mut i32;
/// let z = 12;
///
/// unsafe {
///     std::ptr::write_volatile(y, z);
///     assert_eq!(std::ptr::read_volatile(y), 12);
/// }
/// ```
#[inline]
#[stable(feature = "volatile", since = "1.9.0")]
pub unsafe fn write_volatile<T>(dst: *mut T, src: T) {
    intrinsics::volatile_store(dst, src);
}

#[lang = "const_ptr"]
impl<T: ?Sized> *const T {
    /// Returns `true` if the pointer is null.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// let s: &str = "Follow the rabbit";
    /// let ptr: *const u8 = s.as_ptr();
    /// assert!(!ptr.is_null());
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn is_null(self) -> bool where T: Sized {
        self == null()
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
    /// Additionally, the lifetime `'a` returned is arbitrarily chosen and does
    /// not necessarily reflect the actual lifetime of the data.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// let ptr: *const u8 = &10u8 as *const u8;
    ///
    /// unsafe {
    ///     if let Some(val_back) = ptr.as_ref() {
    ///         println!("We got back the value: {}!", val_back);
    ///     }
    /// }
    /// ```
    #[stable(feature = "ptr_as_ref", since = "1.9.0")]
    #[inline]
    pub unsafe fn as_ref<'a>(self) -> Option<&'a T> where T: Sized {
        if self.is_null() {
            None
        } else {
            Some(&*self)
        }
    }

    /// Calculates the offset from a pointer. `count` is in units of T; e.g. a
    /// `count` of 3 represents a pointer offset of `3 * size_of::<T>()` bytes.
    ///
    /// # Safety
    ///
    /// Both the starting and resulting pointer must be either in bounds or one
    /// byte past the end of an allocated object. If either pointer is out of
    /// bounds or arithmetic overflow occurs then
    /// any further use of the returned value will result in undefined behavior.
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
    ///     println!("{}", *ptr.offset(1) as char);
    ///     println!("{}", *ptr.offset(2) as char);
    /// }
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub unsafe fn offset(self, count: isize) -> *const T where T: Sized {
        intrinsics::offset(self, count)
    }

    /// Calculates the offset from a pointer using wrapping arithmetic.
    /// `count` is in units of T; e.g. a `count` of 3 represents a pointer
    /// offset of `3 * size_of::<T>()` bytes.
    ///
    /// # Safety
    ///
    /// The resulting pointer does not need to be in bounds, but it is
    /// potentially hazardous to dereference (which requires `unsafe`).
    ///
    /// Always use `.offset(count)` instead when possible, because `offset`
    /// allows the compiler to optimize better.
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
    #[inline]
    pub fn wrapping_offset(self, count: isize) -> *const T where T: Sized {
        unsafe {
            intrinsics::arith_offset(self, count)
        }
    }

    /// Calculates the distance between two pointers. The returned value is in
    /// units of T: the distance in bytes is divided by `mem::size_of::<T>()`.
    ///
    /// If the address different between the two pointers ia not a multiple of
    /// `mem::size_of::<T>()` then the result of the division is rounded towards
    /// zero.
    ///
    /// This function returns `None` if `T` is a zero-sized typed.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// #![feature(offset_to)]
    ///
    /// fn main() {
    ///     let a = [0; 5];
    ///     let ptr1: *const i32 = &a[1];
    ///     let ptr2: *const i32 = &a[3];
    ///     assert_eq!(ptr1.offset_to(ptr2), Some(2));
    ///     assert_eq!(ptr2.offset_to(ptr1), Some(-2));
    ///     assert_eq!(unsafe { ptr1.offset(2) }, ptr2);
    ///     assert_eq!(unsafe { ptr2.offset(-2) }, ptr1);
    /// }
    /// ```
    #[unstable(feature = "offset_to", issue = "41079")]
    #[inline]
    pub fn offset_to(self, other: *const T) -> Option<isize> where T: Sized {
        let size = mem::size_of::<T>();
        if size == 0 {
            None
        } else {
            let diff = (other as isize).wrapping_sub(self as isize);
            Some(diff / size as isize)
        }
    }
}

#[lang = "mut_ptr"]
impl<T: ?Sized> *mut T {
    /// Returns `true` if the pointer is null.
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
    pub fn is_null(self) -> bool where T: Sized {
        self == null_mut()
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
    /// Additionally, the lifetime `'a` returned is arbitrarily chosen and does
    /// not necessarily reflect the actual lifetime of the data.
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
    #[stable(feature = "ptr_as_ref", since = "1.9.0")]
    #[inline]
    pub unsafe fn as_ref<'a>(self) -> Option<&'a T> where T: Sized {
        if self.is_null() {
            None
        } else {
            Some(&*self)
        }
    }

    /// Calculates the offset from a pointer. `count` is in units of T; e.g. a
    /// `count` of 3 represents a pointer offset of `3 * size_of::<T>()` bytes.
    ///
    /// # Safety
    ///
    /// The offset must be in-bounds of the object, or one-byte-past-the-end.
    /// Otherwise `offset` invokes Undefined Behavior, regardless of whether
    /// the pointer is used.
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
    #[inline]
    pub unsafe fn offset(self, count: isize) -> *mut T where T: Sized {
        intrinsics::offset(self, count) as *mut T
    }

    /// Calculates the offset from a pointer using wrapping arithmetic.
    /// `count` is in units of T; e.g. a `count` of 3 represents a pointer
    /// offset of `3 * size_of::<T>()` bytes.
    ///
    /// # Safety
    ///
    /// The resulting pointer does not need to be in bounds, but it is
    /// potentially hazardous to dereference (which requires `unsafe`).
    ///
    /// Always use `.offset(count)` instead when possible, because `offset`
    /// allows the compiler to optimize better.
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
    #[inline]
    pub fn wrapping_offset(self, count: isize) -> *mut T where T: Sized {
        unsafe {
            intrinsics::arith_offset(self, count) as *mut T
        }
    }

    /// Returns `None` if the pointer is null, or else returns a mutable
    /// reference to the value wrapped in `Some`.
    ///
    /// # Safety
    ///
    /// As with `as_ref`, this is unsafe because it cannot verify the validity
    /// of the returned pointer, nor can it ensure that the lifetime `'a`
    /// returned is indeed a valid lifetime for the contained data.
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
    #[stable(feature = "ptr_as_ref", since = "1.9.0")]
    #[inline]
    pub unsafe fn as_mut<'a>(self) -> Option<&'a mut T> where T: Sized {
        if self.is_null() {
            None
        } else {
            Some(&mut *self)
        }
    }

    /// Calculates the distance between two pointers. The returned value is in
    /// units of T: the distance in bytes is divided by `mem::size_of::<T>()`.
    ///
    /// If the address different between the two pointers ia not a multiple of
    /// `mem::size_of::<T>()` then the result of the division is rounded towards
    /// zero.
    ///
    /// This function returns `None` if `T` is a zero-sized typed.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// #![feature(offset_to)]
    ///
    /// fn main() {
    ///     let mut a = [0; 5];
    ///     let ptr1: *mut i32 = &mut a[1];
    ///     let ptr2: *mut i32 = &mut a[3];
    ///     assert_eq!(ptr1.offset_to(ptr2), Some(2));
    ///     assert_eq!(ptr2.offset_to(ptr1), Some(-2));
    ///     assert_eq!(unsafe { ptr1.offset(2) }, ptr2);
    ///     assert_eq!(unsafe { ptr2.offset(-2) }, ptr1);
    /// }
    /// ```
    #[unstable(feature = "offset_to", issue = "41079")]
    #[inline]
    pub fn offset_to(self, other: *const T) -> Option<isize> where T: Sized {
        let size = mem::size_of::<T>();
        if size == 0 {
            None
        } else {
            let diff = (other as isize).wrapping_sub(self as isize);
            Some(diff / size as isize)
        }
    }
}

// Equality for pointers
#[stable(feature = "rust1", since = "1.0.0")]
impl<T: ?Sized> PartialEq for *const T {
    #[inline]
    fn eq(&self, other: &*const T) -> bool { *self == *other }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: ?Sized> Eq for *const T {}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: ?Sized> PartialEq for *mut T {
    #[inline]
    fn eq(&self, other: &*mut T) -> bool { *self == *other }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: ?Sized> Eq for *mut T {}

/// Compare raw pointers for equality.
///
/// This is the same as using the `==` operator, but less generic:
/// the arguments have to be `*const T` raw pointers,
/// not anything that implements `PartialEq`.
///
/// This can be used to compare `&T` references (which coerce to `*const T` implicitly)
/// by their address rather than comparing the values they point to
/// (which is what the `PartialEq for &T` implementation does).
///
/// # Examples
///
/// ```
/// use std::ptr;
///
/// let five = 5;
/// let other_five = 5;
/// let five_ref = &five;
/// let same_five_ref = &five;
/// let other_five_ref = &other_five;
///
/// assert!(five_ref == same_five_ref);
/// assert!(five_ref == other_five_ref);
///
/// assert!(ptr::eq(five_ref, same_five_ref));
/// assert!(!ptr::eq(five_ref, other_five_ref));
/// ```
#[stable(feature = "ptr_eq", since = "1.17.0")]
#[inline]
pub fn eq<T: ?Sized>(a: *const T, b: *const T) -> bool {
    a == b
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: ?Sized> Clone for *const T {
    #[inline]
    fn clone(&self) -> *const T {
        *self
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: ?Sized> Clone for *mut T {
    #[inline]
    fn clone(&self) -> *mut T {
        *self
    }
}

// Impls for function pointers
macro_rules! fnptr_impls_safety_abi {
    ($FnTy: ty, $($Arg: ident),*) => {
        #[stable(feature = "rust1", since = "1.0.0")]
        impl<Ret, $($Arg),*> Clone for $FnTy {
            #[inline]
            fn clone(&self) -> Self {
                *self
            }
        }

        #[stable(feature = "fnptr_impls", since = "1.4.0")]
        impl<Ret, $($Arg),*> PartialEq for $FnTy {
            #[inline]
            fn eq(&self, other: &Self) -> bool {
                *self as usize == *other as usize
            }
        }

        #[stable(feature = "fnptr_impls", since = "1.4.0")]
        impl<Ret, $($Arg),*> Eq for $FnTy {}

        #[stable(feature = "fnptr_impls", since = "1.4.0")]
        impl<Ret, $($Arg),*> PartialOrd for $FnTy {
            #[inline]
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                (*self as usize).partial_cmp(&(*other as usize))
            }
        }

        #[stable(feature = "fnptr_impls", since = "1.4.0")]
        impl<Ret, $($Arg),*> Ord for $FnTy {
            #[inline]
            fn cmp(&self, other: &Self) -> Ordering {
                (*self as usize).cmp(&(*other as usize))
            }
        }

        #[stable(feature = "fnptr_impls", since = "1.4.0")]
        impl<Ret, $($Arg),*> hash::Hash for $FnTy {
            fn hash<HH: hash::Hasher>(&self, state: &mut HH) {
                state.write_usize(*self as usize)
            }
        }

        #[stable(feature = "fnptr_impls", since = "1.4.0")]
        impl<Ret, $($Arg),*> fmt::Pointer for $FnTy {
            fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                fmt::Pointer::fmt(&(*self as *const ()), f)
            }
        }

        #[stable(feature = "fnptr_impls", since = "1.4.0")]
        impl<Ret, $($Arg),*> fmt::Debug for $FnTy {
            fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                fmt::Pointer::fmt(&(*self as *const ()), f)
            }
        }
    }
}

macro_rules! fnptr_impls_args {
    ($($Arg: ident),+) => {
        fnptr_impls_safety_abi! { extern "Rust" fn($($Arg),*) -> Ret, $($Arg),* }
        fnptr_impls_safety_abi! { extern "C" fn($($Arg),*) -> Ret, $($Arg),* }
        fnptr_impls_safety_abi! { extern "C" fn($($Arg),* , ...) -> Ret, $($Arg),* }
        fnptr_impls_safety_abi! { unsafe extern "Rust" fn($($Arg),*) -> Ret, $($Arg),* }
        fnptr_impls_safety_abi! { unsafe extern "C" fn($($Arg),*) -> Ret, $($Arg),* }
        fnptr_impls_safety_abi! { unsafe extern "C" fn($($Arg),* , ...) -> Ret, $($Arg),* }
    };
    () => {
        // No variadic functions with 0 parameters
        fnptr_impls_safety_abi! { extern "Rust" fn() -> Ret, }
        fnptr_impls_safety_abi! { extern "C" fn() -> Ret, }
        fnptr_impls_safety_abi! { unsafe extern "Rust" fn() -> Ret, }
        fnptr_impls_safety_abi! { unsafe extern "C" fn() -> Ret, }
    };
}

fnptr_impls_args! { }
fnptr_impls_args! { A }
fnptr_impls_args! { A, B }
fnptr_impls_args! { A, B, C }
fnptr_impls_args! { A, B, C, D }
fnptr_impls_args! { A, B, C, D, E }
fnptr_impls_args! { A, B, C, D, E, F }
fnptr_impls_args! { A, B, C, D, E, F, G }
fnptr_impls_args! { A, B, C, D, E, F, G, H }
fnptr_impls_args! { A, B, C, D, E, F, G, H, I }
fnptr_impls_args! { A, B, C, D, E, F, G, H, I, J }
fnptr_impls_args! { A, B, C, D, E, F, G, H, I, J, K }
fnptr_impls_args! { A, B, C, D, E, F, G, H, I, J, K, L }

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
    fn lt(&self, other: &*const T) -> bool { *self < *other }

    #[inline]
    fn le(&self, other: &*const T) -> bool { *self <= *other }

    #[inline]
    fn gt(&self, other: &*const T) -> bool { *self > *other }

    #[inline]
    fn ge(&self, other: &*const T) -> bool { *self >= *other }
}

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
    fn lt(&self, other: &*mut T) -> bool { *self < *other }

    #[inline]
    fn le(&self, other: &*mut T) -> bool { *self <= *other }

    #[inline]
    fn gt(&self, other: &*mut T) -> bool { *self > *other }

    #[inline]
    fn ge(&self, other: &*mut T) -> bool { *self >= *other }
}

/// A wrapper around a raw non-null `*mut T` that indicates that the possessor
/// of this wrapper owns the referent. Useful for building abstractions like
/// `Box<T>`, `Vec<T>`, `String`, and `HashMap<K, V>`.
///
/// Unlike `*mut T`, `Unique<T>` behaves "as if" it were an instance of `T`.
/// It implements `Send`/`Sync` if `T` is `Send`/`Sync`. It also implies
/// the kind of strong aliasing guarantees an instance of `T` can expect:
/// the referent of the pointer should not be modified without a unique path to
/// its owning Unique.
///
/// If you're uncertain of whether it's correct to use `Unique` for your purposes,
/// consider using `Shared`, which has weaker semantics.
///
/// Unlike `*mut T`, the pointer must always be non-null, even if the pointer
/// is never dereferenced. This is so that enums may use this forbidden value
/// as a discriminant -- `Option<Unique<T>>` has the same size as `Unique<T>`.
/// However the pointer may still dangle if it isn't dereferenced.
///
/// Unlike `*mut T`, `Unique<T>` is covariant over `T`. This should always be correct
/// for any type which upholds Unique's aliasing requirements.
#[allow(missing_debug_implementations)]
#[unstable(feature = "unique", reason = "needs an RFC to flesh out design",
           issue = "27730")]
pub struct Unique<T: ?Sized> {
    pointer: NonZero<*const T>,
    // NOTE: this marker has no consequences for variance, but is necessary
    // for dropck to understand that we logically own a `T`.
    //
    // For details, see:
    // https://github.com/rust-lang/rfcs/blob/master/text/0769-sound-generic-drop.md#phantom-data
    _marker: PhantomData<T>,
}

/// `Unique` pointers are `Send` if `T` is `Send` because the data they
/// reference is unaliased. Note that this aliasing invariant is
/// unenforced by the type system; the abstraction using the
/// `Unique` must enforce it.
#[unstable(feature = "unique", issue = "27730")]
unsafe impl<T: Send + ?Sized> Send for Unique<T> { }

/// `Unique` pointers are `Sync` if `T` is `Sync` because the data they
/// reference is unaliased. Note that this aliasing invariant is
/// unenforced by the type system; the abstraction using the
/// `Unique` must enforce it.
#[unstable(feature = "unique", issue = "27730")]
unsafe impl<T: Sync + ?Sized> Sync for Unique<T> { }

#[unstable(feature = "unique", issue = "27730")]
impl<T: Sized> Unique<T> {
    /// Creates a new `Unique` that is dangling, but well-aligned.
    ///
    /// This is useful for initializing types which lazily allocate, like
    /// `Vec::new` does.
    pub fn empty() -> Self {
        unsafe {
            let ptr = mem::align_of::<T>() as *mut T;
            Unique::new(ptr)
        }
    }
}

#[unstable(feature = "unique", issue = "27730")]
impl<T: ?Sized> Unique<T> {
    /// Creates a new `Unique`.
    ///
    /// # Safety
    ///
    /// `ptr` must be non-null.
    pub const unsafe fn new(ptr: *mut T) -> Unique<T> {
        Unique { pointer: NonZero::new(ptr), _marker: PhantomData }
    }

    /// Acquires the underlying `*mut` pointer.
    pub fn as_ptr(self) -> *mut T {
        self.pointer.get() as *mut T
    }

    /// Dereferences the content.
    ///
    /// The resulting lifetime is bound to self so this behaves "as if"
    /// it were actually an instance of T that is getting borrowed. If a longer
    /// (unbound) lifetime is needed, use `&*my_ptr.ptr()`.
    pub unsafe fn as_ref(&self) -> &T {
        &*self.as_ptr()
    }

    /// Mutably dereferences the content.
    ///
    /// The resulting lifetime is bound to self so this behaves "as if"
    /// it were actually an instance of T that is getting borrowed. If a longer
    /// (unbound) lifetime is needed, use `&mut *my_ptr.ptr()`.
    pub unsafe fn as_mut(&mut self) -> &mut T {
        &mut *self.as_ptr()
    }
}

#[unstable(feature = "shared", issue = "27730")]
impl<T: ?Sized> Clone for Unique<T> {
    fn clone(&self) -> Self {
        *self
    }
}

#[unstable(feature = "shared", issue = "27730")]
impl<T: ?Sized> Copy for Unique<T> { }

#[unstable(feature = "unique", issue = "27730")]
impl<T: ?Sized, U: ?Sized> CoerceUnsized<Unique<U>> for Unique<T> where T: Unsize<U> { }

#[unstable(feature = "unique", issue = "27730")]
impl<T: ?Sized> fmt::Pointer for Unique<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Pointer::fmt(&self.as_ptr(), f)
    }
}

/// A wrapper around a raw `*mut T` that indicates that the possessor
/// of this wrapper has shared ownership of the referent. Useful for
/// building abstractions like `Rc<T>`, `Arc<T>`, or doubly-linked lists, which
/// internally use aliased raw pointers to manage the memory that they own.
///
/// This is similar to `Unique`, except that it doesn't make any aliasing
/// guarantees, and doesn't derive Send and Sync. Note that unlike `&T`,
/// Shared has no special mutability requirements. Shared may mutate data
/// aliased by other Shared pointers. More precise rules require Rust to
/// develop an actual aliasing model.
///
/// Unlike `*mut T`, the pointer must always be non-null, even if the pointer
/// is never dereferenced. This is so that enums may use this forbidden value
/// as a discriminant -- `Option<Shared<T>>` has the same size as `Shared<T>`.
/// However the pointer may still dangle if it isn't dereferenced.
///
/// Unlike `*mut T`, `Shared<T>` is covariant over `T`. If this is incorrect
/// for your use case, you should include some PhantomData in your type to
/// provide invariance, such as `PhantomData<Cell<T>>` or `PhantomData<&'a mut T>`.
/// Usually this won't be necessary; covariance is correct for Rc, Arc, and LinkedList
/// because they provide a public API that follows the normal shared XOR mutable
/// rules of Rust.
#[allow(missing_debug_implementations)]
#[unstable(feature = "shared", reason = "needs an RFC to flesh out design",
           issue = "27730")]
pub struct Shared<T: ?Sized> {
    pointer: NonZero<*const T>,
    // NOTE: this marker has no consequences for variance, but is necessary
    // for dropck to understand that we logically own a `T`.
    //
    // For details, see:
    // https://github.com/rust-lang/rfcs/blob/master/text/0769-sound-generic-drop.md#phantom-data
    _marker: PhantomData<T>,
}

/// `Shared` pointers are not `Send` because the data they reference may be aliased.
// NB: This impl is unnecessary, but should provide better error messages.
#[unstable(feature = "shared", issue = "27730")]
impl<T: ?Sized> !Send for Shared<T> { }

/// `Shared` pointers are not `Sync` because the data they reference may be aliased.
// NB: This impl is unnecessary, but should provide better error messages.
#[unstable(feature = "shared", issue = "27730")]
impl<T: ?Sized> !Sync for Shared<T> { }

#[unstable(feature = "shared", issue = "27730")]
impl<T: Sized> Shared<T> {
    /// Creates a new `Shared` that is dangling, but well-aligned.
    ///
    /// This is useful for initializing types which lazily allocate, like
    /// `Vec::new` does.
    pub fn empty() -> Self {
        unsafe {
            let ptr = mem::align_of::<T>() as *mut T;
            Shared::new(ptr)
        }
    }
}

#[unstable(feature = "shared", issue = "27730")]
impl<T: ?Sized> Shared<T> {
    /// Creates a new `Shared`.
    ///
    /// # Safety
    ///
    /// `ptr` must be non-null.
    pub unsafe fn new(ptr: *mut T) -> Self {
        Shared { pointer: NonZero::new(ptr), _marker: PhantomData }
    }

    /// Acquires the underlying `*mut` pointer.
    pub fn as_ptr(self) -> *mut T {
        self.pointer.get() as *mut T
    }

    /// Dereferences the content.
    ///
    /// The resulting lifetime is bound to self so this behaves "as if"
    /// it were actually an instance of T that is getting borrowed. If a longer
    /// (unbound) lifetime is needed, use `&*my_ptr.ptr()`.
    pub unsafe fn as_ref(&self) -> &T {
        &*self.as_ptr()
    }

    /// Mutably dereferences the content.
    ///
    /// The resulting lifetime is bound to self so this behaves "as if"
    /// it were actually an instance of T that is getting borrowed. If a longer
    /// (unbound) lifetime is needed, use `&mut *my_ptr.ptr_mut()`.
    pub unsafe fn as_mut(&mut self) -> &mut T {
        &mut *self.as_ptr()
    }

    /// Acquires the underlying pointer as a `*mut` pointer.
    #[rustc_deprecated(since = "1.19", reason = "renamed to `as_ptr` for ergonomics/consistency")]
    #[unstable(feature = "shared", issue = "27730")]
    pub unsafe fn as_mut_ptr(&self) -> *mut T {
        self.as_ptr()
    }
}

#[unstable(feature = "shared", issue = "27730")]
impl<T: ?Sized> Clone for Shared<T> {
    fn clone(&self) -> Self {
        *self
    }
}

#[unstable(feature = "shared", issue = "27730")]
impl<T: ?Sized> Copy for Shared<T> { }

#[unstable(feature = "shared", issue = "27730")]
impl<T: ?Sized, U: ?Sized> CoerceUnsized<Shared<U>> for Shared<T> where T: Unsize<U> { }

#[unstable(feature = "shared", issue = "27730")]
impl<T: ?Sized> fmt::Pointer for Shared<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Pointer::fmt(&self.as_ptr(), f)
    }
}
