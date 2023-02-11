//! Basic functions for dealing with memory.
//!
//! This module contains functions for querying the size and alignment of
//! types, initializing and manipulating memory.

#![stable(feature = "rust1", since = "1.0.0")]

use crate::clone;
use crate::cmp;
use crate::fmt;
use crate::hash;
use crate::intrinsics;
use crate::marker::{Copy, DiscriminantKind, Sized};
use crate::ptr;

mod manually_drop;
#[stable(feature = "manually_drop", since = "1.20.0")]
pub use manually_drop::ManuallyDrop;

mod maybe_uninit;
#[stable(feature = "maybe_uninit", since = "1.36.0")]
pub use maybe_uninit::MaybeUninit;

mod transmutability;
#[unstable(feature = "transmutability", issue = "99571")]
pub use transmutability::{Assume, BikeshedIntrinsicFrom};

#[stable(feature = "rust1", since = "1.0.0")]
#[doc(inline)]
pub use crate::intrinsics::transmute;

/// Takes ownership and "forgets" about the value **without running its destructor**.
///
/// Any resources the value manages, such as heap memory or a file handle, will linger
/// forever in an unreachable state. However, it does not guarantee that pointers
/// to this memory will remain valid.
///
/// * If you want to leak memory, see [`Box::leak`].
/// * If you want to obtain a raw pointer to the memory, see [`Box::into_raw`].
/// * If you want to dispose of a value properly, running its destructor, see
/// [`mem::drop`].
///
/// # Safety
///
/// `forget` is not marked as `unsafe`, because Rust's safety guarantees
/// do not include a guarantee that destructors will always run. For example,
/// a program can create a reference cycle using [`Rc`][rc], or call
/// [`process::exit`][exit] to exit without running destructors. Thus, allowing
/// `mem::forget` from safe code does not fundamentally change Rust's safety
/// guarantees.
///
/// That said, leaking resources such as memory or I/O objects is usually undesirable.
/// The need comes up in some specialized use cases for FFI or unsafe code, but even
/// then, [`ManuallyDrop`] is typically preferred.
///
/// Because forgetting a value is allowed, any `unsafe` code you write must
/// allow for this possibility. You cannot return a value and expect that the
/// caller will necessarily run the value's destructor.
///
/// [rc]: ../../std/rc/struct.Rc.html
/// [exit]: ../../std/process/fn.exit.html
///
/// # Examples
///
/// The canonical safe use of `mem::forget` is to circumvent a value's destructor
/// implemented by the `Drop` trait. For example, this will leak a `File`, i.e. reclaim
/// the space taken by the variable but never close the underlying system resource:
///
/// ```no_run
/// use std::mem;
/// use std::fs::File;
///
/// let file = File::open("foo.txt").unwrap();
/// mem::forget(file);
/// ```
///
/// This is useful when the ownership of the underlying resource was previously
/// transferred to code outside of Rust, for example by transmitting the raw
/// file descriptor to C code.
///
/// # Relationship with `ManuallyDrop`
///
/// While `mem::forget` can also be used to transfer *memory* ownership, doing so is error-prone.
/// [`ManuallyDrop`] should be used instead. Consider, for example, this code:
///
/// ```
/// use std::mem;
///
/// let mut v = vec![65, 122];
/// // Build a `String` using the contents of `v`
/// let s = unsafe { String::from_raw_parts(v.as_mut_ptr(), v.len(), v.capacity()) };
/// // leak `v` because its memory is now managed by `s`
/// mem::forget(v);  // ERROR - v is invalid and must not be passed to a function
/// assert_eq!(s, "Az");
/// // `s` is implicitly dropped and its memory deallocated.
/// ```
///
/// There are two issues with the above example:
///
/// * If more code were added between the construction of `String` and the invocation of
///   `mem::forget()`, a panic within it would cause a double free because the same memory
///   is handled by both `v` and `s`.
/// * After calling `v.as_mut_ptr()` and transmitting the ownership of the data to `s`,
///   the `v` value is invalid. Even when a value is just moved to `mem::forget` (which won't
///   inspect it), some types have strict requirements on their values that
///   make them invalid when dangling or no longer owned. Using invalid values in any
///   way, including passing them to or returning them from functions, constitutes
///   undefined behavior and may break the assumptions made by the compiler.
///
/// Switching to `ManuallyDrop` avoids both issues:
///
/// ```
/// use std::mem::ManuallyDrop;
///
/// let v = vec![65, 122];
/// // Before we disassemble `v` into its raw parts, make sure it
/// // does not get dropped!
/// let mut v = ManuallyDrop::new(v);
/// // Now disassemble `v`. These operations cannot panic, so there cannot be a leak.
/// let (ptr, len, cap) = (v.as_mut_ptr(), v.len(), v.capacity());
/// // Finally, build a `String`.
/// let s = unsafe { String::from_raw_parts(ptr, len, cap) };
/// assert_eq!(s, "Az");
/// // `s` is implicitly dropped and its memory deallocated.
/// ```
///
/// `ManuallyDrop` robustly prevents double-free because we disable `v`'s destructor
/// before doing anything else. `mem::forget()` doesn't allow this because it consumes its
/// argument, forcing us to call it only after extracting anything we need from `v`. Even
/// if a panic were introduced between construction of `ManuallyDrop` and building the
/// string (which cannot happen in the code as shown), it would result in a leak and not a
/// double free. In other words, `ManuallyDrop` errs on the side of leaking instead of
/// erring on the side of (double-)dropping.
///
/// Also, `ManuallyDrop` prevents us from having to "touch" `v` after transferring the
/// ownership to `s` — the final step of interacting with `v` to dispose of it without
/// running its destructor is entirely avoided.
///
/// [`Box`]: ../../std/boxed/struct.Box.html
/// [`Box::leak`]: ../../std/boxed/struct.Box.html#method.leak
/// [`Box::into_raw`]: ../../std/boxed/struct.Box.html#method.into_raw
/// [`mem::drop`]: drop
/// [ub]: ../../reference/behavior-considered-undefined.html
#[inline]
#[rustc_const_stable(feature = "const_forget", since = "1.46.0")]
#[stable(feature = "rust1", since = "1.0.0")]
#[cfg_attr(not(test), rustc_diagnostic_item = "mem_forget")]
pub const fn forget<T>(t: T) {
    let _ = ManuallyDrop::new(t);
}

/// Like [`forget`], but also accepts unsized values.
///
/// This function is just a shim intended to be removed when the `unsized_locals` feature gets
/// stabilized.
#[inline]
#[unstable(feature = "forget_unsized", issue = "none")]
pub fn forget_unsized<T: ?Sized>(t: T) {
    intrinsics::forget(t)
}

/// Returns the size of a type in bytes.
///
/// More specifically, this is the offset in bytes between successive elements
/// in an array with that item type including alignment padding. Thus, for any
/// type `T` and length `n`, `[T; n]` has a size of `n * size_of::<T>()`.
///
/// In general, the size of a type is not stable across compilations, but
/// specific types such as primitives are.
///
/// The following table gives the size for primitives.
///
/// Type | size_of::\<Type>()
/// ---- | ---------------
/// () | 0
/// bool | 1
/// u8 | 1
/// u16 | 2
/// u32 | 4
/// u64 | 8
/// u128 | 16
/// i8 | 1
/// i16 | 2
/// i32 | 4
/// i64 | 8
/// i128 | 16
/// f32 | 4
/// f64 | 8
/// char | 4
///
/// Furthermore, `usize` and `isize` have the same size.
///
/// The types `*const T`, `&T`, `Box<T>`, `Option<&T>`, and `Option<Box<T>>` all have
/// the same size. If `T` is Sized, all of those types have the same size as `usize`.
///
/// The mutability of a pointer does not change its size. As such, `&T` and `&mut T`
/// have the same size. Likewise for `*const T` and `*mut T`.
///
/// # Size of `#[repr(C)]` items
///
/// The `C` representation for items has a defined layout. With this layout,
/// the size of items is also stable as long as all fields have a stable size.
///
/// ## Size of Structs
///
/// For `structs`, the size is determined by the following algorithm.
///
/// For each field in the struct ordered by declaration order:
///
/// 1. Add the size of the field.
/// 2. Round up the current size to the nearest multiple of the next field's [alignment].
///
/// Finally, round the size of the struct to the nearest multiple of its [alignment].
/// The alignment of the struct is usually the largest alignment of all its
/// fields; this can be changed with the use of `repr(align(N))`.
///
/// Unlike `C`, zero sized structs are not rounded up to one byte in size.
///
/// ## Size of Enums
///
/// Enums that carry no data other than the discriminant have the same size as C enums
/// on the platform they are compiled for.
///
/// ## Size of Unions
///
/// The size of a union is the size of its largest field.
///
/// Unlike `C`, zero sized unions are not rounded up to one byte in size.
///
/// # Examples
///
/// ```
/// use std::mem;
///
/// // Some primitives
/// assert_eq!(4, mem::size_of::<i32>());
/// assert_eq!(8, mem::size_of::<f64>());
/// assert_eq!(0, mem::size_of::<()>());
///
/// // Some arrays
/// assert_eq!(8, mem::size_of::<[i32; 2]>());
/// assert_eq!(12, mem::size_of::<[i32; 3]>());
/// assert_eq!(0, mem::size_of::<[i32; 0]>());
///
///
/// // Pointer size equality
/// assert_eq!(mem::size_of::<&i32>(), mem::size_of::<*const i32>());
/// assert_eq!(mem::size_of::<&i32>(), mem::size_of::<Box<i32>>());
/// assert_eq!(mem::size_of::<&i32>(), mem::size_of::<Option<&i32>>());
/// assert_eq!(mem::size_of::<Box<i32>>(), mem::size_of::<Option<Box<i32>>>());
/// ```
///
/// Using `#[repr(C)]`.
///
/// ```
/// use std::mem;
///
/// #[repr(C)]
/// struct FieldStruct {
///     first: u8,
///     second: u16,
///     third: u8
/// }
///
/// // The size of the first field is 1, so add 1 to the size. Size is 1.
/// // The alignment of the second field is 2, so add 1 to the size for padding. Size is 2.
/// // The size of the second field is 2, so add 2 to the size. Size is 4.
/// // The alignment of the third field is 1, so add 0 to the size for padding. Size is 4.
/// // The size of the third field is 1, so add 1 to the size. Size is 5.
/// // Finally, the alignment of the struct is 2 (because the largest alignment amongst its
/// // fields is 2), so add 1 to the size for padding. Size is 6.
/// assert_eq!(6, mem::size_of::<FieldStruct>());
///
/// #[repr(C)]
/// struct TupleStruct(u8, u16, u8);
///
/// // Tuple structs follow the same rules.
/// assert_eq!(6, mem::size_of::<TupleStruct>());
///
/// // Note that reordering the fields can lower the size. We can remove both padding bytes
/// // by putting `third` before `second`.
/// #[repr(C)]
/// struct FieldStructOptimized {
///     first: u8,
///     third: u8,
///     second: u16
/// }
///
/// assert_eq!(4, mem::size_of::<FieldStructOptimized>());
///
/// // Union size is the size of the largest field.
/// #[repr(C)]
/// union ExampleUnion {
///     smaller: u8,
///     larger: u16
/// }
///
/// assert_eq!(2, mem::size_of::<ExampleUnion>());
/// ```
///
/// [alignment]: align_of
#[inline(always)]
#[must_use]
#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_promotable]
#[rustc_const_stable(feature = "const_mem_size_of", since = "1.24.0")]
#[cfg_attr(not(test), rustc_diagnostic_item = "mem_size_of")]
pub const fn size_of<T>() -> usize {
    intrinsics::size_of::<T>()
}

/// Returns the size of the pointed-to value in bytes.
///
/// This is usually the same as `size_of::<T>()`. However, when `T` *has* no
/// statically-known size, e.g., a slice [`[T]`][slice] or a [trait object],
/// then `size_of_val` can be used to get the dynamically-known size.
///
/// [trait object]: ../../book/ch17-02-trait-objects.html
///
/// # Examples
///
/// ```
/// use std::mem;
///
/// assert_eq!(4, mem::size_of_val(&5i32));
///
/// let x: [u8; 13] = [0; 13];
/// let y: &[u8] = &x;
/// assert_eq!(13, mem::size_of_val(y));
/// ```
#[inline]
#[must_use]
#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_const_unstable(feature = "const_size_of_val", issue = "46571")]
#[cfg_attr(not(test), rustc_diagnostic_item = "mem_size_of_val")]
pub const fn size_of_val<T: ?Sized>(val: &T) -> usize {
    // SAFETY: `val` is a reference, so it's a valid raw pointer
    unsafe { intrinsics::size_of_val(val) }
}

/// Returns the size of the pointed-to value in bytes.
///
/// This is usually the same as `size_of::<T>()`. However, when `T` *has* no
/// statically-known size, e.g., a slice [`[T]`][slice] or a [trait object],
/// then `size_of_val_raw` can be used to get the dynamically-known size.
///
/// # Safety
///
/// This function is only safe to call if the following conditions hold:
///
/// - If `T` is `Sized`, this function is always safe to call.
/// - If the unsized tail of `T` is:
///     - a [slice], then the length of the slice tail must be an initialized
///       integer, and the size of the *entire value*
///       (dynamic tail length + statically sized prefix) must fit in `isize`.
///     - a [trait object], then the vtable part of the pointer must point
///       to a valid vtable acquired by an unsizing coercion, and the size
///       of the *entire value* (dynamic tail length + statically sized prefix)
///       must fit in `isize`.
///     - an (unstable) [extern type], then this function is always safe to
///       call, but may panic or otherwise return the wrong value, as the
///       extern type's layout is not known. This is the same behavior as
///       [`size_of_val`] on a reference to a type with an extern type tail.
///     - otherwise, it is conservatively not allowed to call this function.
///
/// [trait object]: ../../book/ch17-02-trait-objects.html
/// [extern type]: ../../unstable-book/language-features/extern-types.html
///
/// # Examples
///
/// ```
/// #![feature(layout_for_ptr)]
/// use std::mem;
///
/// assert_eq!(4, mem::size_of_val(&5i32));
///
/// let x: [u8; 13] = [0; 13];
/// let y: &[u8] = &x;
/// assert_eq!(13, unsafe { mem::size_of_val_raw(y) });
/// ```
#[inline]
#[must_use]
#[unstable(feature = "layout_for_ptr", issue = "69835")]
#[rustc_const_unstable(feature = "const_size_of_val_raw", issue = "46571")]
pub const unsafe fn size_of_val_raw<T: ?Sized>(val: *const T) -> usize {
    // SAFETY: the caller must provide a valid raw pointer
    unsafe { intrinsics::size_of_val(val) }
}

/// Returns the [ABI]-required minimum alignment of a type in bytes.
///
/// Every reference to a value of the type `T` must be a multiple of this number.
///
/// This is the alignment used for struct fields. It may be smaller than the preferred alignment.
///
/// [ABI]: https://en.wikipedia.org/wiki/Application_binary_interface
///
/// # Examples
///
/// ```
/// # #![allow(deprecated)]
/// use std::mem;
///
/// assert_eq!(4, mem::min_align_of::<i32>());
/// ```
#[inline]
#[must_use]
#[stable(feature = "rust1", since = "1.0.0")]
#[deprecated(note = "use `align_of` instead", since = "1.2.0")]
pub fn min_align_of<T>() -> usize {
    intrinsics::min_align_of::<T>()
}

/// Returns the [ABI]-required minimum alignment of the type of the value that `val` points to in
/// bytes.
///
/// Every reference to a value of the type `T` must be a multiple of this number.
///
/// [ABI]: https://en.wikipedia.org/wiki/Application_binary_interface
///
/// # Examples
///
/// ```
/// # #![allow(deprecated)]
/// use std::mem;
///
/// assert_eq!(4, mem::min_align_of_val(&5i32));
/// ```
#[inline]
#[must_use]
#[stable(feature = "rust1", since = "1.0.0")]
#[deprecated(note = "use `align_of_val` instead", since = "1.2.0")]
pub fn min_align_of_val<T: ?Sized>(val: &T) -> usize {
    // SAFETY: val is a reference, so it's a valid raw pointer
    unsafe { intrinsics::min_align_of_val(val) }
}

/// Returns the [ABI]-required minimum alignment of a type in bytes.
///
/// Every reference to a value of the type `T` must be a multiple of this number.
///
/// This is the alignment used for struct fields. It may be smaller than the preferred alignment.
///
/// [ABI]: https://en.wikipedia.org/wiki/Application_binary_interface
///
/// # Examples
///
/// ```
/// use std::mem;
///
/// assert_eq!(4, mem::align_of::<i32>());
/// ```
#[inline(always)]
#[must_use]
#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_promotable]
#[rustc_const_stable(feature = "const_align_of", since = "1.24.0")]
pub const fn align_of<T>() -> usize {
    intrinsics::min_align_of::<T>()
}

/// Returns the [ABI]-required minimum alignment of the type of the value that `val` points to in
/// bytes.
///
/// Every reference to a value of the type `T` must be a multiple of this number.
///
/// [ABI]: https://en.wikipedia.org/wiki/Application_binary_interface
///
/// # Examples
///
/// ```
/// use std::mem;
///
/// assert_eq!(4, mem::align_of_val(&5i32));
/// ```
#[inline]
#[must_use]
#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_const_unstable(feature = "const_align_of_val", issue = "46571")]
#[allow(deprecated)]
pub const fn align_of_val<T: ?Sized>(val: &T) -> usize {
    // SAFETY: val is a reference, so it's a valid raw pointer
    unsafe { intrinsics::min_align_of_val(val) }
}

/// Returns the [ABI]-required minimum alignment of the type of the value that `val` points to in
/// bytes.
///
/// Every reference to a value of the type `T` must be a multiple of this number.
///
/// [ABI]: https://en.wikipedia.org/wiki/Application_binary_interface
///
/// # Safety
///
/// This function is only safe to call if the following conditions hold:
///
/// - If `T` is `Sized`, this function is always safe to call.
/// - If the unsized tail of `T` is:
///     - a [slice], then the length of the slice tail must be an initialized
///       integer, and the size of the *entire value*
///       (dynamic tail length + statically sized prefix) must fit in `isize`.
///     - a [trait object], then the vtable part of the pointer must point
///       to a valid vtable acquired by an unsizing coercion, and the size
///       of the *entire value* (dynamic tail length + statically sized prefix)
///       must fit in `isize`.
///     - an (unstable) [extern type], then this function is always safe to
///       call, but may panic or otherwise return the wrong value, as the
///       extern type's layout is not known. This is the same behavior as
///       [`align_of_val`] on a reference to a type with an extern type tail.
///     - otherwise, it is conservatively not allowed to call this function.
///
/// [trait object]: ../../book/ch17-02-trait-objects.html
/// [extern type]: ../../unstable-book/language-features/extern-types.html
///
/// # Examples
///
/// ```
/// #![feature(layout_for_ptr)]
/// use std::mem;
///
/// assert_eq!(4, unsafe { mem::align_of_val_raw(&5i32) });
/// ```
#[inline]
#[must_use]
#[unstable(feature = "layout_for_ptr", issue = "69835")]
#[rustc_const_unstable(feature = "const_align_of_val_raw", issue = "46571")]
pub const unsafe fn align_of_val_raw<T: ?Sized>(val: *const T) -> usize {
    // SAFETY: the caller must provide a valid raw pointer
    unsafe { intrinsics::min_align_of_val(val) }
}

/// Returns `true` if dropping values of type `T` matters.
///
/// This is purely an optimization hint, and may be implemented conservatively:
/// it may return `true` for types that don't actually need to be dropped.
/// As such always returning `true` would be a valid implementation of
/// this function. However if this function actually returns `false`, then you
/// can be certain dropping `T` has no side effect.
///
/// Low level implementations of things like collections, which need to manually
/// drop their data, should use this function to avoid unnecessarily
/// trying to drop all their contents when they are destroyed. This might not
/// make a difference in release builds (where a loop that has no side-effects
/// is easily detected and eliminated), but is often a big win for debug builds.
///
/// Note that [`drop_in_place`] already performs this check, so if your workload
/// can be reduced to some small number of [`drop_in_place`] calls, using this is
/// unnecessary. In particular note that you can [`drop_in_place`] a slice, and that
/// will do a single needs_drop check for all the values.
///
/// Types like Vec therefore just `drop_in_place(&mut self[..])` without using
/// `needs_drop` explicitly. Types like [`HashMap`], on the other hand, have to drop
/// values one at a time and should use this API.
///
/// [`drop_in_place`]: crate::ptr::drop_in_place
/// [`HashMap`]: ../../std/collections/struct.HashMap.html
///
/// # Examples
///
/// Here's an example of how a collection might make use of `needs_drop`:
///
/// ```
/// use std::{mem, ptr};
///
/// pub struct MyCollection<T> {
/// #   data: [T; 1],
///     /* ... */
/// }
/// # impl<T> MyCollection<T> {
/// #   fn iter_mut(&mut self) -> &mut [T] { &mut self.data }
/// #   fn free_buffer(&mut self) {}
/// # }
///
/// impl<T> Drop for MyCollection<T> {
///     fn drop(&mut self) {
///         unsafe {
///             // drop the data
///             if mem::needs_drop::<T>() {
///                 for x in self.iter_mut() {
///                     ptr::drop_in_place(x);
///                 }
///             }
///             self.free_buffer();
///         }
///     }
/// }
/// ```
#[inline]
#[must_use]
#[stable(feature = "needs_drop", since = "1.21.0")]
#[rustc_const_stable(feature = "const_mem_needs_drop", since = "1.36.0")]
#[rustc_diagnostic_item = "needs_drop"]
pub const fn needs_drop<T: ?Sized>() -> bool {
    intrinsics::needs_drop::<T>()
}

/// Returns the value of type `T` represented by the all-zero byte-pattern.
///
/// This means that, for example, the padding byte in `(u8, u16)` is not
/// necessarily zeroed.
///
/// There is no guarantee that an all-zero byte-pattern represents a valid value
/// of some type `T`. For example, the all-zero byte-pattern is not a valid value
/// for reference types (`&T`, `&mut T`) and functions pointers. Using `zeroed`
/// on such types causes immediate [undefined behavior][ub] because [the Rust
/// compiler assumes][inv] that there always is a valid value in a variable it
/// considers initialized.
///
/// This has the same effect as [`MaybeUninit::zeroed().assume_init()`][zeroed].
/// It is useful for FFI sometimes, but should generally be avoided.
///
/// [zeroed]: MaybeUninit::zeroed
/// [ub]: ../../reference/behavior-considered-undefined.html
/// [inv]: MaybeUninit#initialization-invariant
///
/// # Examples
///
/// Correct usage of this function: initializing an integer with zero.
///
/// ```
/// use std::mem;
///
/// let x: i32 = unsafe { mem::zeroed() };
/// assert_eq!(0, x);
/// ```
///
/// *Incorrect* usage of this function: initializing a reference with zero.
///
/// ```rust,no_run
/// # #![allow(invalid_value)]
/// use std::mem;
///
/// let _x: &i32 = unsafe { mem::zeroed() }; // Undefined behavior!
/// let _y: fn() = unsafe { mem::zeroed() }; // And again!
/// ```
#[inline(always)]
#[must_use]
#[stable(feature = "rust1", since = "1.0.0")]
#[allow(deprecated_in_future)]
#[allow(deprecated)]
#[rustc_diagnostic_item = "mem_zeroed"]
#[track_caller]
pub unsafe fn zeroed<T>() -> T {
    // SAFETY: the caller must guarantee that an all-zero value is valid for `T`.
    unsafe {
        intrinsics::assert_zero_valid::<T>();
        MaybeUninit::zeroed().assume_init()
    }
}

/// Bypasses Rust's normal memory-initialization checks by pretending to
/// produce a value of type `T`, while doing nothing at all.
///
/// **This function is deprecated.** Use [`MaybeUninit<T>`] instead.
/// It also might be slower than using `MaybeUninit<T>` due to mitigations that were put in place to
/// limit the potential harm caused by incorrect use of this function in legacy code.
///
/// The reason for deprecation is that the function basically cannot be used
/// correctly: it has the same effect as [`MaybeUninit::uninit().assume_init()`][uninit].
/// As the [`assume_init` documentation][assume_init] explains,
/// [the Rust compiler assumes][inv] that values are properly initialized.
///
/// Truly uninitialized memory like what gets returned here
/// is special in that the compiler knows that it does not have a fixed value.
/// This makes it undefined behavior to have uninitialized data in a variable even
/// if that variable has an integer type.
///
/// Therefore, it is immediate undefined behavior to call this function on nearly all types,
/// including integer types and arrays of integer types, and even if the result is unused.
///
/// [uninit]: MaybeUninit::uninit
/// [assume_init]: MaybeUninit::assume_init
/// [inv]: MaybeUninit#initialization-invariant
#[inline(always)]
#[must_use]
#[deprecated(since = "1.39.0", note = "use `mem::MaybeUninit` instead")]
#[stable(feature = "rust1", since = "1.0.0")]
#[allow(deprecated_in_future)]
#[allow(deprecated)]
#[rustc_diagnostic_item = "mem_uninitialized"]
#[track_caller]
pub unsafe fn uninitialized<T>() -> T {
    // SAFETY: the caller must guarantee that an uninitialized value is valid for `T`.
    unsafe {
        intrinsics::assert_mem_uninitialized_valid::<T>();
        let mut val = MaybeUninit::<T>::uninit();

        // Fill memory with 0x01, as an imperfect mitigation for old code that uses this function on
        // bool, nonnull, and noundef types. But don't do this if we actively want to detect UB.
        if !cfg!(any(miri, sanitize = "memory")) {
            val.as_mut_ptr().write_bytes(0x01, 1);
        }

        val.assume_init()
    }
}

/// Swaps the values at two mutable locations, without deinitializing either one.
///
/// * If you want to swap with a default or dummy value, see [`take`].
/// * If you want to swap with a passed value, returning the old value, see [`replace`].
///
/// # Examples
///
/// ```
/// use std::mem;
///
/// let mut x = 5;
/// let mut y = 42;
///
/// mem::swap(&mut x, &mut y);
///
/// assert_eq!(42, x);
/// assert_eq!(5, y);
/// ```
#[inline]
#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_const_unstable(feature = "const_swap", issue = "83163")]
pub const fn swap<T>(x: &mut T, y: &mut T) {
    // NOTE(eddyb) SPIR-V's Logical addressing model doesn't allow for arbitrary
    // reinterpretation of values as (chunkable) byte arrays, and the loop in the
    // block optimization in `swap_slice` is hard to rewrite back
    // into the (unoptimized) direct swapping implementation, so we disable it.
    // FIXME(eddyb) the block optimization also prevents MIR optimizations from
    // understanding `mem::replace`, `Option::take`, etc. - a better overall
    // solution might be to make `ptr::swap_nonoverlapping` into an intrinsic, which
    // a backend can choose to implement using the block optimization, or not.
    #[cfg(not(any(target_arch = "spirv")))]
    {
        // For types that are larger multiples of their alignment, the simple way
        // tends to copy the whole thing to stack rather than doing it one part
        // at a time, so instead treat them as one-element slices and piggy-back
        // the slice optimizations that will split up the swaps.
        if size_of::<T>() / align_of::<T>() > 4 {
            // SAFETY: exclusive references always point to one non-overlapping
            // element and are non-null and properly aligned.
            return unsafe { ptr::swap_nonoverlapping(x, y, 1) };
        }
    }

    // If a scalar consists of just a small number of alignment units, let
    // the codegen just swap those pieces directly, as it's likely just a
    // few instructions and anything else is probably overcomplicated.
    //
    // Most importantly, this covers primitives and simd types that tend to
    // have size=align where doing anything else can be a pessimization.
    // (This will also be used for ZSTs, though any solution works for them.)
    swap_simple(x, y);
}

/// Same as [`swap`] semantically, but always uses the simple implementation.
///
/// Used elsewhere in `mem` and `ptr` at the bottom layer of calls.
#[rustc_const_unstable(feature = "const_swap", issue = "83163")]
#[inline]
pub(crate) const fn swap_simple<T>(x: &mut T, y: &mut T) {
    // We arrange for this to typically be called with small types,
    // so this reads-and-writes approach is actually better than using
    // copy_nonoverlapping as it easily puts things in LLVM registers
    // directly and doesn't end up inlining allocas.
    // And LLVM actually optimizes it to 3×memcpy if called with
    // a type larger than it's willing to keep in a register.
    // Having typed reads and writes in MIR here is also good as
    // it lets MIRI and CTFE understand them better, including things
    // like enforcing type validity for them.
    // Importantly, read+copy_nonoverlapping+write introduces confusing
    // asymmetry to the behaviour where one value went through read+write
    // whereas the other was copied over by the intrinsic (see #94371).

    // SAFETY: exclusive references are always valid to read/write,
    // including being aligned, and nothing here panics so it's drop-safe.
    unsafe {
        let a = ptr::read(x);
        let b = ptr::read(y);
        ptr::write(x, b);
        ptr::write(y, a);
    }
}

/// Replaces `dest` with the default value of `T`, returning the previous `dest` value.
///
/// * If you want to replace the values of two variables, see [`swap`].
/// * If you want to replace with a passed value instead of the default value, see [`replace`].
///
/// # Examples
///
/// A simple example:
///
/// ```
/// use std::mem;
///
/// let mut v: Vec<i32> = vec![1, 2];
///
/// let old_v = mem::take(&mut v);
/// assert_eq!(vec![1, 2], old_v);
/// assert!(v.is_empty());
/// ```
///
/// `take` allows taking ownership of a struct field by replacing it with an "empty" value.
/// Without `take` you can run into issues like these:
///
/// ```compile_fail,E0507
/// struct Buffer<T> { buf: Vec<T> }
///
/// impl<T> Buffer<T> {
///     fn get_and_reset(&mut self) -> Vec<T> {
///         // error: cannot move out of dereference of `&mut`-pointer
///         let buf = self.buf;
///         self.buf = Vec::new();
///         buf
///     }
/// }
/// ```
///
/// Note that `T` does not necessarily implement [`Clone`], so it can't even clone and reset
/// `self.buf`. But `take` can be used to disassociate the original value of `self.buf` from
/// `self`, allowing it to be returned:
///
/// ```
/// use std::mem;
///
/// # struct Buffer<T> { buf: Vec<T> }
/// impl<T> Buffer<T> {
///     fn get_and_reset(&mut self) -> Vec<T> {
///         mem::take(&mut self.buf)
///     }
/// }
///
/// let mut buffer = Buffer { buf: vec![0, 1] };
/// assert_eq!(buffer.buf.len(), 2);
///
/// assert_eq!(buffer.get_and_reset(), vec![0, 1]);
/// assert_eq!(buffer.buf.len(), 0);
/// ```
#[inline]
#[stable(feature = "mem_take", since = "1.40.0")]
pub fn take<T: Default>(dest: &mut T) -> T {
    replace(dest, T::default())
}

/// Moves `src` into the referenced `dest`, returning the previous `dest` value.
///
/// Neither value is dropped.
///
/// * If you want to replace the values of two variables, see [`swap`].
/// * If you want to replace with a default value, see [`take`].
///
/// # Examples
///
/// A simple example:
///
/// ```
/// use std::mem;
///
/// let mut v: Vec<i32> = vec![1, 2];
///
/// let old_v = mem::replace(&mut v, vec![3, 4, 5]);
/// assert_eq!(vec![1, 2], old_v);
/// assert_eq!(vec![3, 4, 5], v);
/// ```
///
/// `replace` allows consumption of a struct field by replacing it with another value.
/// Without `replace` you can run into issues like these:
///
/// ```compile_fail,E0507
/// struct Buffer<T> { buf: Vec<T> }
///
/// impl<T> Buffer<T> {
///     fn replace_index(&mut self, i: usize, v: T) -> T {
///         // error: cannot move out of dereference of `&mut`-pointer
///         let t = self.buf[i];
///         self.buf[i] = v;
///         t
///     }
/// }
/// ```
///
/// Note that `T` does not necessarily implement [`Clone`], so we can't even clone `self.buf[i]` to
/// avoid the move. But `replace` can be used to disassociate the original value at that index from
/// `self`, allowing it to be returned:
///
/// ```
/// # #![allow(dead_code)]
/// use std::mem;
///
/// # struct Buffer<T> { buf: Vec<T> }
/// impl<T> Buffer<T> {
///     fn replace_index(&mut self, i: usize, v: T) -> T {
///         mem::replace(&mut self.buf[i], v)
///     }
/// }
///
/// let mut buffer = Buffer { buf: vec![0, 1] };
/// assert_eq!(buffer.buf[0], 0);
///
/// assert_eq!(buffer.replace_index(0, 2), 0);
/// assert_eq!(buffer.buf[0], 2);
/// ```
#[inline]
#[stable(feature = "rust1", since = "1.0.0")]
#[must_use = "if you don't need the old value, you can just assign the new value directly"]
#[rustc_const_unstable(feature = "const_replace", issue = "83164")]
#[cfg_attr(not(test), rustc_diagnostic_item = "mem_replace")]
pub const fn replace<T>(dest: &mut T, src: T) -> T {
    // SAFETY: We read from `dest` but directly write `src` into it afterwards,
    // such that the old value is not duplicated. Nothing is dropped and
    // nothing here can panic.
    unsafe {
        let result = ptr::read(dest);
        ptr::write(dest, src);
        result
    }
}

/// Disposes of a value.
///
/// This does so by calling the argument's implementation of [`Drop`][drop].
///
/// This effectively does nothing for types which implement `Copy`, e.g.
/// integers. Such values are copied and _then_ moved into the function, so the
/// value persists after this function call.
///
/// This function is not magic; it is literally defined as
///
/// ```
/// pub fn drop<T>(_x: T) { }
/// ```
///
/// Because `_x` is moved into the function, it is automatically dropped before
/// the function returns.
///
/// [drop]: Drop
///
/// # Examples
///
/// Basic usage:
///
/// ```
/// let v = vec![1, 2, 3];
///
/// drop(v); // explicitly drop the vector
/// ```
///
/// Since [`RefCell`] enforces the borrow rules at runtime, `drop` can
/// release a [`RefCell`] borrow:
///
/// ```
/// use std::cell::RefCell;
///
/// let x = RefCell::new(1);
///
/// let mut mutable_borrow = x.borrow_mut();
/// *mutable_borrow = 1;
///
/// drop(mutable_borrow); // relinquish the mutable borrow on this slot
///
/// let borrow = x.borrow();
/// println!("{}", *borrow);
/// ```
///
/// Integers and other types implementing [`Copy`] are unaffected by `drop`.
///
/// ```
/// #[derive(Copy, Clone)]
/// struct Foo(u8);
///
/// let x = 1;
/// let y = Foo(2);
/// drop(x); // a copy of `x` is moved and dropped
/// drop(y); // a copy of `y` is moved and dropped
///
/// println!("x: {}, y: {}", x, y.0); // still available
/// ```
///
/// [`RefCell`]: crate::cell::RefCell
#[inline]
#[stable(feature = "rust1", since = "1.0.0")]
#[cfg_attr(not(test), rustc_diagnostic_item = "mem_drop")]
pub fn drop<T>(_x: T) {}

/// Bitwise-copies a value.
///
/// This function is not magic; it is literally defined as
/// ```
/// pub fn copy<T: Copy>(x: &T) -> T { *x }
/// ```
///
/// It is useful when you want to pass a function pointer to a combinator, rather than defining a new closure.
///
/// Example:
/// ```
/// #![feature(mem_copy_fn)]
/// use core::mem::copy;
/// let result_from_ffi_function: Result<(), &i32> = Err(&1);
/// let result_copied: Result<(), i32> = result_from_ffi_function.map_err(copy);
/// ```
#[inline]
#[unstable(feature = "mem_copy_fn", issue = "98262")]
pub const fn copy<T: Copy>(x: &T) -> T {
    *x
}

/// Interprets `src` as having type `&Dst`, and then reads `src` without moving
/// the contained value.
///
/// This function will unsafely assume the pointer `src` is valid for [`size_of::<Dst>`][size_of]
/// bytes by transmuting `&Src` to `&Dst` and then reading the `&Dst` (except that this is done
/// in a way that is correct even when `&Dst` has stricter alignment requirements than `&Src`).
/// It will also unsafely create a copy of the contained value instead of moving out of `src`.
///
/// It is not a compile-time error if `Src` and `Dst` have different sizes, but it
/// is highly encouraged to only invoke this function where `Src` and `Dst` have the
/// same size. This function triggers [undefined behavior][ub] if `Dst` is larger than
/// `Src`.
///
/// [ub]: ../../reference/behavior-considered-undefined.html
///
/// # Examples
///
/// ```
/// use std::mem;
///
/// #[repr(packed)]
/// struct Foo {
///     bar: u8,
/// }
///
/// let foo_array = [10u8];
///
/// unsafe {
///     // Copy the data from 'foo_array' and treat it as a 'Foo'
///     let mut foo_struct: Foo = mem::transmute_copy(&foo_array);
///     assert_eq!(foo_struct.bar, 10);
///
///     // Modify the copied data
///     foo_struct.bar = 20;
///     assert_eq!(foo_struct.bar, 20);
/// }
///
/// // The contents of 'foo_array' should not have changed
/// assert_eq!(foo_array, [10]);
/// ```
#[inline]
#[must_use]
#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_const_unstable(feature = "const_transmute_copy", issue = "83165")]
pub const unsafe fn transmute_copy<Src, Dst>(src: &Src) -> Dst {
    assert!(
        size_of::<Src>() >= size_of::<Dst>(),
        "cannot transmute_copy if Dst is larger than Src"
    );

    // If Dst has a higher alignment requirement, src might not be suitably aligned.
    if align_of::<Dst>() > align_of::<Src>() {
        // SAFETY: `src` is a reference which is guaranteed to be valid for reads.
        // The caller must guarantee that the actual transmutation is safe.
        unsafe { ptr::read_unaligned(src as *const Src as *const Dst) }
    } else {
        // SAFETY: `src` is a reference which is guaranteed to be valid for reads.
        // We just checked that `src as *const Dst` was properly aligned.
        // The caller must guarantee that the actual transmutation is safe.
        unsafe { ptr::read(src as *const Src as *const Dst) }
    }
}

/// Opaque type representing the discriminant of an enum.
///
/// See the [`discriminant`] function in this module for more information.
#[stable(feature = "discriminant_value", since = "1.21.0")]
pub struct Discriminant<T>(<T as DiscriminantKind>::Discriminant);

// N.B. These trait implementations cannot be derived because we don't want any bounds on T.

#[stable(feature = "discriminant_value", since = "1.21.0")]
impl<T> Copy for Discriminant<T> {}

#[stable(feature = "discriminant_value", since = "1.21.0")]
impl<T> clone::Clone for Discriminant<T> {
    fn clone(&self) -> Self {
        *self
    }
}

#[stable(feature = "discriminant_value", since = "1.21.0")]
impl<T> cmp::PartialEq for Discriminant<T> {
    fn eq(&self, rhs: &Self) -> bool {
        self.0 == rhs.0
    }
}

#[stable(feature = "discriminant_value", since = "1.21.0")]
impl<T> cmp::Eq for Discriminant<T> {}

#[stable(feature = "discriminant_value", since = "1.21.0")]
impl<T> hash::Hash for Discriminant<T> {
    fn hash<H: hash::Hasher>(&self, state: &mut H) {
        self.0.hash(state);
    }
}

#[stable(feature = "discriminant_value", since = "1.21.0")]
impl<T> fmt::Debug for Discriminant<T> {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt.debug_tuple("Discriminant").field(&self.0).finish()
    }
}

/// Returns a value uniquely identifying the enum variant in `v`.
///
/// If `T` is not an enum, calling this function will not result in undefined behavior, but the
/// return value is unspecified.
///
/// # Stability
///
/// The discriminant of an enum variant may change if the enum definition changes. A discriminant
/// of some variant will not change between compilations with the same compiler. See the [Reference]
/// for more information.
///
/// [Reference]: ../../reference/items/enumerations.html#custom-discriminant-values-for-fieldless-enumerations
///
/// # Examples
///
/// This can be used to compare enums that carry data, while disregarding
/// the actual data:
///
/// ```
/// use std::mem;
///
/// enum Foo { A(&'static str), B(i32), C(i32) }
///
/// assert_eq!(mem::discriminant(&Foo::A("bar")), mem::discriminant(&Foo::A("baz")));
/// assert_eq!(mem::discriminant(&Foo::B(1)), mem::discriminant(&Foo::B(2)));
/// assert_ne!(mem::discriminant(&Foo::B(3)), mem::discriminant(&Foo::C(3)));
/// ```
///
/// ## Accessing the numeric value of the discriminant
///
/// Note that it is *undefined behavior* to [`transmute`] from [`Discriminant`] to a primitive!
///
/// If an enum has only unit variants, then the numeric value of the discriminant can be accessed
/// with an [`as`] cast:
///
/// ```
/// enum Enum {
///     Foo,
///     Bar,
///     Baz,
/// }
///
/// assert_eq!(0, Enum::Foo as isize);
/// assert_eq!(1, Enum::Bar as isize);
/// assert_eq!(2, Enum::Baz as isize);
/// ```
///
/// If an enum has opted-in to having a [primitive representation] for its discriminant,
/// then it's possible to use pointers to read the memory location storing the discriminant.
/// That **cannot** be done for enums using the [default representation], however, as it's
/// undefined what layout the discriminant has and where it's stored — it might not even be
/// stored at all!
///
/// [`as`]: ../../std/keyword.as.html
/// [primitive representation]: ../../reference/type-layout.html#primitive-representations
/// [default representation]: ../../reference/type-layout.html#the-default-representation
/// ```
/// #[repr(u8)]
/// enum Enum {
///     Unit,
///     Tuple(bool),
///     Struct { a: bool },
/// }
///
/// impl Enum {
///     fn discriminant(&self) -> u8 {
///         // SAFETY: Because `Self` is marked `repr(u8)`, its layout is a `repr(C)` `union`
///         // between `repr(C)` structs, each of which has the `u8` discriminant as its first
///         // field, so we can read the discriminant without offsetting the pointer.
///         unsafe { *<*const _>::from(self).cast::<u8>() }
///     }
/// }
///
/// let unit_like = Enum::Unit;
/// let tuple_like = Enum::Tuple(true);
/// let struct_like = Enum::Struct { a: false };
/// assert_eq!(0, unit_like.discriminant());
/// assert_eq!(1, tuple_like.discriminant());
/// assert_eq!(2, struct_like.discriminant());
///
/// // ⚠️ This is undefined behavior. Don't do this. ⚠️
/// // assert_eq!(0, unsafe { std::mem::transmute::<_, u8>(std::mem::discriminant(&unit_like)) });
/// ```
#[stable(feature = "discriminant_value", since = "1.21.0")]
#[rustc_const_unstable(feature = "const_discriminant", issue = "69821")]
#[cfg_attr(not(test), rustc_diagnostic_item = "mem_discriminant")]
#[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
pub const fn discriminant<T>(v: &T) -> Discriminant<T> {
    Discriminant(intrinsics::discriminant_value(v))
}

/// Returns the number of variants in the enum type `T`.
///
/// If `T` is not an enum, calling this function will not result in undefined behavior, but the
/// return value is unspecified. Equally, if `T` is an enum with more variants than `usize::MAX`
/// the return value is unspecified. Uninhabited variants will be counted.
///
/// Note that an enum may be expanded with additional variants in the future
/// as a non-breaking change, for example if it is marked `#[non_exhaustive]`,
/// which will change the result of this function.
///
/// # Examples
///
/// ```
/// # #![feature(never_type)]
/// # #![feature(variant_count)]
///
/// use std::mem;
///
/// enum Void {}
/// enum Foo { A(&'static str), B(i32), C(i32) }
///
/// assert_eq!(mem::variant_count::<Void>(), 0);
/// assert_eq!(mem::variant_count::<Foo>(), 3);
///
/// assert_eq!(mem::variant_count::<Option<!>>(), 2);
/// assert_eq!(mem::variant_count::<Result<!, !>>(), 2);
/// ```
#[inline(always)]
#[must_use]
#[unstable(feature = "variant_count", issue = "73662")]
#[rustc_const_unstable(feature = "variant_count", issue = "73662")]
#[rustc_diagnostic_item = "mem_variant_count"]
pub const fn variant_count<T>() -> usize {
    intrinsics::variant_count::<T>()
}

/// Provides associated constants for various useful properties of types,
/// to give them a canonical form in our code and make them easier to read.
///
/// This is here only to simplify all the ZST checks we need in the library.
/// It's not on a stabilization track right now.
#[doc(hidden)]
#[unstable(feature = "sized_type_properties", issue = "none")]
pub trait SizedTypeProperties: Sized {
    /// `true` if this type requires no storage.
    /// `false` if its [size](size_of) is greater than zero.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(sized_type_properties)]
    /// use core::mem::SizedTypeProperties;
    ///
    /// fn do_something_with<T>() {
    ///     if T::IS_ZST {
    ///         // ... special approach ...
    ///     } else {
    ///         // ... the normal thing ...
    ///     }
    /// }
    ///
    /// struct MyUnit;
    /// assert!(MyUnit::IS_ZST);
    ///
    /// // For negative checks, consider using UFCS to emphasize the negation
    /// assert!(!<i32>::IS_ZST);
    /// // As it can sometimes hide in the type otherwise
    /// assert!(!String::IS_ZST);
    /// ```
    #[doc(hidden)]
    #[unstable(feature = "sized_type_properties", issue = "none")]
    const IS_ZST: bool = size_of::<Self>() == 0;
}
#[doc(hidden)]
#[unstable(feature = "sized_type_properties", issue = "none")]
impl<T> SizedTypeProperties for T {}
