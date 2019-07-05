//! Manually manage memory through raw pointers.
//!
//! *[See also the pointer primitive types](../../std/primitive.pointer.html).*
//!
//! # Safety
//!
//! Many functions in this module take raw pointers as arguments and read from
//! or write to them. For this to be safe, these pointers must be *valid*.
//! Whether a pointer is valid depends on the operation it is used for
//! (read or write), and the extent of the memory that is accessed (i.e.,
//! how many bytes are read/written). Most functions use `*mut T` and `*const T`
//! to access only a single value, in which case the documentation omits the size
//! and implicitly assumes it to be `size_of::<T>()` bytes.
//!
//! The precise rules for validity are not determined yet. The guarantees that are
//! provided at this point are very minimal:
//!
//! * A [null] pointer is *never* valid, not even for accesses of [size zero][zst].
//! * All pointers (except for the null pointer) are valid for all operations of
//!   [size zero][zst].
//! * All accesses performed by functions in this module are *non-atomic* in the sense
//!   of [atomic operations] used to synchronize between threads. This means it is
//!   undefined behavior to perform two concurrent accesses to the same location from different
//!   threads unless both accesses only read from memory. Notice that this explicitly
//!   includes [`read_volatile`] and [`write_volatile`]: Volatile accesses cannot
//!   be used for inter-thread synchronization.
//! * The result of casting a reference to a pointer is valid for as long as the
//!   underlying object is live and no reference (just raw pointers) is used to
//!   access the same memory.
//!
//! These axioms, along with careful use of [`offset`] for pointer arithmetic,
//! are enough to correctly implement many useful things in unsafe code. Stronger guarantees
//! will be provided eventually, as the [aliasing] rules are being determined. For more
//! information, see the [book] as well as the section in the reference devoted
//! to [undefined behavior][ub].
//!
//! ## Alignment
//!
//! Valid raw pointers as defined above are not necessarily properly aligned (where
//! "proper" alignment is defined by the pointee type, i.e., `*const T` must be
//! aligned to `mem::align_of::<T>()`). However, most functions require their
//! arguments to be properly aligned, and will explicitly state
//! this requirement in their documentation. Notable exceptions to this are
//! [`read_unaligned`] and [`write_unaligned`].
//!
//! When a function requires proper alignment, it does so even if the access
//! has size 0, i.e., even if memory is not actually touched. Consider using
//! [`NonNull::dangling`] in such cases.
//!
//! [aliasing]: ../../nomicon/aliasing.html
//! [book]: ../../book/ch19-01-unsafe-rust.html#dereferencing-a-raw-pointer
//! [ub]: ../../reference/behavior-considered-undefined.html
//! [null]: ./fn.null.html
//! [zst]: ../../nomicon/exotic-sizes.html#zero-sized-types-zsts
//! [atomic operations]: ../../std/sync/atomic/index.html
//! [`copy`]: ../../std/ptr/fn.copy.html
//! [`offset`]: ../../std/primitive.pointer.html#method.offset
//! [`read_unaligned`]: ./fn.read_unaligned.html
//! [`write_unaligned`]: ./fn.write_unaligned.html
//! [`read_volatile`]: ./fn.read_volatile.html
//! [`write_volatile`]: ./fn.write_volatile.html
//! [`NonNull::dangling`]: ./struct.NonNull.html#method.dangling

#![stable(feature = "rust1", since = "1.0.0")]

use crate::intrinsics;
use crate::fmt;
use crate::hash;
use crate::mem::{self, MaybeUninit};
use crate::cmp::Ordering::{self, Less, Equal, Greater};

#[stable(feature = "rust1", since = "1.0.0")]
pub use crate::intrinsics::copy_nonoverlapping;

#[stable(feature = "rust1", since = "1.0.0")]
pub use crate::intrinsics::copy;

#[stable(feature = "rust1", since = "1.0.0")]
pub use crate::intrinsics::write_bytes;

mod non_null;
#[stable(feature = "nonnull", since = "1.25.0")]
pub use non_null::NonNull;

mod unique;
#[unstable(feature = "ptr_internals", issue = "0")]
pub use unique::Unique;

/// Executes the destructor (if any) of the pointed-to value.
///
/// This is semantically equivalent to calling [`ptr::read`] and discarding
/// the result, but has the following advantages:
///
/// * It is *required* to use `drop_in_place` to drop unsized types like
///   trait objects, because they can't be read out onto the stack and
///   dropped normally.
///
/// * It is friendlier to the optimizer to do this over [`ptr::read`] when
///   dropping manually allocated memory (e.g., when writing Box/Rc/Vec),
///   as the compiler doesn't need to prove that it's sound to elide the
///   copy.
///
/// Unaligned values cannot be dropped in place, they must be copied to an aligned
/// location first using [`ptr::read_unaligned`].
///
/// [`ptr::read`]: ../ptr/fn.read.html
/// [`ptr::read_unaligned`]: ../ptr/fn.read_unaligned.html
///
/// # Safety
///
/// Behavior is undefined if any of the following conditions are violated:
///
/// * `to_drop` must be [valid] for reads.
///
/// * `to_drop` must be properly aligned.
///
/// Additionally, if `T` is not [`Copy`], using the pointed-to value after
/// calling `drop_in_place` can cause undefined behavior. Note that `*to_drop =
/// foo` counts as a use because it will cause the value to be dropped
/// again. [`write`] can be used to overwrite data without causing it to be
/// dropped.
///
/// Note that even if `T` has size `0`, the pointer must be non-NULL and properly aligned.
///
/// [valid]: ../ptr/index.html#safety
/// [`Copy`]: ../marker/trait.Copy.html
/// [`write`]: ../ptr/fn.write.html
///
/// # Examples
///
/// Manually remove the last item from a vector:
///
/// ```
/// use std::ptr;
/// use std::rc::Rc;
///
/// let last = Rc::new(1);
/// let weak = Rc::downgrade(&last);
///
/// let mut v = vec![Rc::new(0), last];
///
/// unsafe {
///     // Get a raw pointer to the last element in `v`.
///     let ptr = &mut v[1] as *mut _;
///     // Shorten `v` to prevent the last item from being dropped. We do that first,
///     // to prevent issues if the `drop_in_place` below panics.
///     v.set_len(1);
///     // Without a call `drop_in_place`, the last item would never be dropped,
///     // and the memory it manages would be leaked.
///     ptr::drop_in_place(ptr);
/// }
///
/// assert_eq!(v, &[0.into()]);
///
/// // Ensure that the last item was dropped.
/// assert!(weak.upgrade().is_none());
/// ```
///
/// Notice that the compiler performs this copy automatically when dropping packed structs,
/// i.e., you do not usually have to worry about such issues unless you call `drop_in_place`
/// manually.
#[stable(feature = "drop_in_place", since = "1.8.0")]
#[inline(always)]
pub unsafe fn drop_in_place<T: ?Sized>(to_drop: *mut T) {
    real_drop_in_place(&mut *to_drop)
}

// The real `drop_in_place` -- the one that gets called implicitly when variables go
// out of scope -- should have a safe reference and not a raw pointer as argument
// type.  When we drop a local variable, we access it with a pointer that behaves
// like a safe reference; transmuting that to a raw pointer does not mean we can
// actually access it with raw pointers.
#[lang = "drop_in_place"]
#[allow(unconditional_recursion)]
unsafe fn real_drop_in_place<T: ?Sized>(to_drop: &mut T) {
    // Code here does not matter - this is replaced by the
    // real drop glue by the compiler.
    real_drop_in_place(to_drop)
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
#[rustc_promotable]
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
#[rustc_promotable]
pub const fn null_mut<T>() -> *mut T { 0 as *mut T }

#[repr(C)]
pub(crate) union Repr<T> {
    pub(crate) rust: *const [T],
    rust_mut: *mut [T],
    pub(crate) raw: FatPtr<T>,
}

#[repr(C)]
pub(crate) struct FatPtr<T> {
    data: *const T,
    pub(crate) len: usize,
}

/// Forms a slice from a pointer and a length.
///
/// The `len` argument is the number of **elements**, not the number of bytes.
///
/// # Examples
///
/// ```rust
/// #![feature(slice_from_raw_parts)]
/// use std::ptr;
///
/// // create a slice pointer when starting out with a pointer to the first element
/// let mut x = [5, 6, 7];
/// let ptr = &mut x[0] as *mut _;
/// let slice = ptr::slice_from_raw_parts_mut(ptr, 3);
/// assert_eq!(unsafe { &*slice }[2], 7);
/// ```
#[inline]
#[unstable(feature = "slice_from_raw_parts", reason = "recently added", issue = "36925")]
pub fn slice_from_raw_parts<T>(data: *const T, len: usize) -> *const [T] {
    unsafe { Repr { raw: FatPtr { data, len } }.rust }
}

/// Performs the same functionality as [`from_raw_parts`], except that a
/// mutable slice is returned.
///
/// See the documentation of [`from_raw_parts`] for more details.
///
/// [`from_raw_parts`]: ../../std/slice/fn.from_raw_parts.html
#[inline]
#[unstable(feature = "slice_from_raw_parts", reason = "recently added", issue = "36925")]
pub fn slice_from_raw_parts_mut<T>(data: *mut T, len: usize) -> *mut [T] {
    unsafe { Repr { raw: FatPtr { data, len } }.rust_mut }
}

/// Swaps the values at two mutable locations of the same type, without
/// deinitializing either.
///
/// But for the following two exceptions, this function is semantically
/// equivalent to [`mem::swap`]:
///
/// * It operates on raw pointers instead of references. When references are
///   available, [`mem::swap`] should be preferred.
///
/// * The two pointed-to values may overlap. If the values do overlap, then the
///   overlapping region of memory from `x` will be used. This is demonstrated
///   in the second example below.
///
/// [`mem::swap`]: ../mem/fn.swap.html
///
/// # Safety
///
/// Behavior is undefined if any of the following conditions are violated:
///
/// * Both `x` and `y` must be [valid] for reads and writes.
///
/// * Both `x` and `y` must be properly aligned.
///
/// Note that even if `T` has size `0`, the pointers must be non-NULL and properly aligned.
///
/// [valid]: ../ptr/index.html#safety
///
/// # Examples
///
/// Swapping two non-overlapping regions:
///
/// ```
/// use std::ptr;
///
/// let mut array = [0, 1, 2, 3];
///
/// let x = array[0..].as_mut_ptr() as *mut [u32; 2]; // this is `array[0..2]`
/// let y = array[2..].as_mut_ptr() as *mut [u32; 2]; // this is `array[2..4]`
///
/// unsafe {
///     ptr::swap(x, y);
///     assert_eq!([2, 3, 0, 1], array);
/// }
/// ```
///
/// Swapping two overlapping regions:
///
/// ```
/// use std::ptr;
///
/// let mut array = [0, 1, 2, 3];
///
/// let x = array[0..].as_mut_ptr() as *mut [u32; 3]; // this is `array[0..3]`
/// let y = array[1..].as_mut_ptr() as *mut [u32; 3]; // this is `array[1..4]`
///
/// unsafe {
///     ptr::swap(x, y);
///     // The indices `1..3` of the slice overlap between `x` and `y`.
///     // Reasonable results would be for to them be `[2, 3]`, so that indices `0..3` are
///     // `[1, 2, 3]` (matching `y` before the `swap`); or for them to be `[0, 1]`
///     // so that indices `1..4` are `[0, 1, 2]` (matching `x` before the `swap`).
///     // This implementation is defined to make the latter choice.
///     assert_eq!([1, 0, 1, 2], array);
/// }
/// ```
#[inline]
#[stable(feature = "rust1", since = "1.0.0")]
pub unsafe fn swap<T>(x: *mut T, y: *mut T) {
    // Give ourselves some scratch space to work with.
    // We do not have to worry about drops: `MaybeUninit` does nothing when dropped.
    let mut tmp = MaybeUninit::<T>::uninit();

    // Perform the swap
    copy_nonoverlapping(x, tmp.as_mut_ptr(), 1);
    copy(y, x, 1); // `x` and `y` may overlap
    copy_nonoverlapping(tmp.as_ptr(), y, 1);
}

/// Swaps `count * size_of::<T>()` bytes between the two regions of memory
/// beginning at `x` and `y`. The two regions must *not* overlap.
///
/// # Safety
///
/// Behavior is undefined if any of the following conditions are violated:
///
/// * Both `x` and `y` must be [valid] for reads and writes of `count *
///   size_of::<T>()` bytes.
///
/// * Both `x` and `y` must be properly aligned.
///
/// * The region of memory beginning at `x` with a size of `count *
///   size_of::<T>()` bytes must *not* overlap with the region of memory
///   beginning at `y` with the same size.
///
/// Note that even if the effectively copied size (`count * size_of::<T>()`) is `0`,
/// the pointers must be non-NULL and properly aligned.
///
/// [valid]: ../ptr/index.html#safety
///
/// # Examples
///
/// Basic usage:
///
/// ```
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
#[stable(feature = "swap_nonoverlapping", since = "1.27.0")]
pub unsafe fn swap_nonoverlapping<T>(x: *mut T, y: *mut T, count: usize) {
    let x = x as *mut u8;
    let y = y as *mut u8;
    let len = mem::size_of::<T>() * count;
    swap_nonoverlapping_bytes(x, y, len)
}

#[inline]
pub(crate) unsafe fn swap_nonoverlapping_one<T>(x: *mut T, y: *mut T) {
    // For types smaller than the block optimization below,
    // just swap directly to avoid pessimizing codegen.
    if mem::size_of::<T>() < 32 {
        let z = read(x);
        copy_nonoverlapping(y, x, 1);
        write(y, z);
    } else {
        swap_nonoverlapping(x, y, 1);
    }
}

#[inline]
unsafe fn swap_nonoverlapping_bytes(x: *mut u8, y: *mut u8, len: usize) {
    // The approach here is to utilize simd to swap x & y efficiently. Testing reveals
    // that swapping either 32 bytes or 64 bytes at a time is most efficient for Intel
    // Haswell E processors. LLVM is more able to optimize if we give a struct a
    // #[repr(simd)], even if we don't actually use this struct directly.
    //
    // FIXME repr(simd) broken on emscripten and redox
    #[cfg_attr(not(any(target_os = "emscripten", target_os = "redox")), repr(simd))]
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
        let mut t = mem::MaybeUninit::<Block>::uninit();
        let t = t.as_mut_ptr() as *mut u8;
        let x = x.add(i);
        let y = y.add(i);

        // Swap a block of bytes of x & y, using t as a temporary buffer
        // This should be optimized into efficient SIMD operations where available
        copy_nonoverlapping(x, t, block_size);
        copy_nonoverlapping(y, x, block_size);
        copy_nonoverlapping(t, y, block_size);
        i += block_size;
    }

    if i < len {
        // Swap any remaining bytes
        let mut t = mem::MaybeUninit::<UnalignedBlock>::uninit();
        let rem = len - i;

        let t = t.as_mut_ptr() as *mut u8;
        let x = x.add(i);
        let y = y.add(i);

        copy_nonoverlapping(x, t, rem);
        copy_nonoverlapping(y, x, rem);
        copy_nonoverlapping(t, y, rem);
    }
}

/// Moves `src` into the pointed `dst`, returning the previous `dst` value.
///
/// Neither value is dropped.
///
/// This function is semantically equivalent to [`mem::replace`] except that it
/// operates on raw pointers instead of references. When references are
/// available, [`mem::replace`] should be preferred.
///
/// [`mem::replace`]: ../mem/fn.replace.html
///
/// # Safety
///
/// Behavior is undefined if any of the following conditions are violated:
///
/// * `dst` must be [valid] for writes.
///
/// * `dst` must be properly aligned.
///
/// Note that even if `T` has size `0`, the pointer must be non-NULL and properly aligned.
///
/// [valid]: ../ptr/index.html#safety
///
/// # Examples
///
/// ```
/// use std::ptr;
///
/// let mut rust = vec!['b', 'u', 's', 't'];
///
/// // `mem::replace` would have the same effect without requiring the unsafe
/// // block.
/// let b = unsafe {
///     ptr::replace(&mut rust[0], 'r')
/// };
///
/// assert_eq!(b, 'b');
/// assert_eq!(rust, &['r', 'u', 's', 't']);
/// ```
#[inline]
#[stable(feature = "rust1", since = "1.0.0")]
pub unsafe fn replace<T>(dst: *mut T, mut src: T) -> T {
    mem::swap(&mut *dst, &mut src); // cannot overlap
    src
}

/// Reads the value from `src` without moving it. This leaves the
/// memory in `src` unchanged.
///
/// # Safety
///
/// Behavior is undefined if any of the following conditions are violated:
///
/// * `src` must be [valid] for reads.
///
/// * `src` must be properly aligned. Use [`read_unaligned`] if this is not the
///   case.
///
/// Note that even if `T` has size `0`, the pointer must be non-NULL and properly aligned.
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
///
/// Manually implement [`mem::swap`]:
///
/// ```
/// use std::ptr;
///
/// fn swap<T>(a: &mut T, b: &mut T) {
///     unsafe {
///         // Create a bitwise copy of the value at `a` in `tmp`.
///         let tmp = ptr::read(a);
///
///         // Exiting at this point (either by explicitly returning or by
///         // calling a function which panics) would cause the value in `tmp` to
///         // be dropped while the same value is still referenced by `a`. This
///         // could trigger undefined behavior if `T` is not `Copy`.
///
///         // Create a bitwise copy of the value at `b` in `a`.
///         // This is safe because mutable references cannot alias.
///         ptr::copy_nonoverlapping(b, a, 1);
///
///         // As above, exiting here could trigger undefined behavior because
///         // the same value is referenced by `a` and `b`.
///
///         // Move `tmp` into `b`.
///         ptr::write(b, tmp);
///
///         // `tmp` has been moved (`write` takes ownership of its second argument),
///         // so nothing is dropped implicitly here.
///     }
/// }
///
/// let mut foo = "foo".to_owned();
/// let mut bar = "bar".to_owned();
///
/// swap(&mut foo, &mut bar);
///
/// assert_eq!(foo, "bar");
/// assert_eq!(bar, "foo");
/// ```
///
/// ## Ownership of the Returned Value
///
/// `read` creates a bitwise copy of `T`, regardless of whether `T` is [`Copy`].
/// If `T` is not [`Copy`], using both the returned value and the value at
/// `*src` can violate memory safety. Note that assigning to `*src` counts as a
/// use because it will attempt to drop the value at `*src`.
///
/// [`write`] can be used to overwrite data without causing it to be dropped.
///
/// ```
/// use std::ptr;
///
/// let mut s = String::from("foo");
/// unsafe {
///     // `s2` now points to the same underlying memory as `s`.
///     let mut s2: String = ptr::read(&s);
///
///     assert_eq!(s2, "foo");
///
///     // Assigning to `s2` causes its original value to be dropped. Beyond
///     // this point, `s` must no longer be used, as the underlying memory has
///     // been freed.
///     s2 = String::default();
///     assert_eq!(s2, "");
///
///     // Assigning to `s` would cause the old value to be dropped again,
///     // resulting in undefined behavior.
///     // s = String::from("bar"); // ERROR
///
///     // `ptr::write` can be used to overwrite a value without dropping it.
///     ptr::write(&mut s, String::from("bar"));
/// }
///
/// assert_eq!(s, "bar");
/// ```
///
/// [`mem::swap`]: ../mem/fn.swap.html
/// [valid]: ../ptr/index.html#safety
/// [`Copy`]: ../marker/trait.Copy.html
/// [`read_unaligned`]: ./fn.read_unaligned.html
/// [`write`]: ./fn.write.html
#[inline]
#[stable(feature = "rust1", since = "1.0.0")]
pub unsafe fn read<T>(src: *const T) -> T {
    let mut tmp = MaybeUninit::<T>::uninit();
    copy_nonoverlapping(src, tmp.as_mut_ptr(), 1);
    tmp.assume_init()
}

/// Reads the value from `src` without moving it. This leaves the
/// memory in `src` unchanged.
///
/// Unlike [`read`], `read_unaligned` works with unaligned pointers.
///
/// # Safety
///
/// Behavior is undefined if any of the following conditions are violated:
///
/// * `src` must be [valid] for reads.
///
/// Like [`read`], `read_unaligned` creates a bitwise copy of `T`, regardless of
/// whether `T` is [`Copy`]. If `T` is not [`Copy`], using both the returned
/// value and the value at `*src` can [violate memory safety][read-ownership].
///
/// Note that even if `T` has size `0`, the pointer must be non-NULL.
///
/// [`Copy`]: ../marker/trait.Copy.html
/// [`read`]: ./fn.read.html
/// [`write_unaligned`]: ./fn.write_unaligned.html
/// [read-ownership]: ./fn.read.html#ownership-of-the-returned-value
/// [valid]: ../ptr/index.html#safety
///
/// ## On `packed` structs
///
/// It is currently impossible to create raw pointers to unaligned fields
/// of a packed struct.
///
/// Attempting to create a raw pointer to an `unaligned` struct field with
/// an expression such as `&packed.unaligned as *const FieldType` creates an
/// intermediate unaligned reference before converting that to a raw pointer.
/// That this reference is temporary and immediately cast is inconsequential
/// as the compiler always expects references to be properly aligned.
/// As a result, using `&packed.unaligned as *const FieldType` causes immediate
/// *undefined behavior* in your program.
///
/// An example of what not to do and how this relates to `read_unaligned` is:
///
/// ```no_run
/// #[repr(packed, C)]
/// struct Packed {
///     _padding: u8,
///     unaligned: u32,
/// }
///
/// let packed = Packed {
///     _padding: 0x00,
///     unaligned: 0x01020304,
/// };
///
/// let v = unsafe {
///     // Here we attempt to take the address of a 32-bit integer which is not aligned.
///     let unaligned =
///         // A temporary unaligned reference is created here which results in
///         // undefined behavior regardless of whether the reference is used or not.
///         &packed.unaligned
///         // Casting to a raw pointer doesn't help; the mistake already happened.
///         as *const u32;
///
///     let v = std::ptr::read_unaligned(unaligned);
///
///     v
/// };
/// ```
///
/// Accessing unaligned fields directly with e.g. `packed.unaligned` is safe however.
// FIXME: Update docs based on outcome of RFC #2582 and friends.
#[inline]
#[stable(feature = "ptr_unaligned", since = "1.17.0")]
pub unsafe fn read_unaligned<T>(src: *const T) -> T {
    let mut tmp = MaybeUninit::<T>::uninit();
    copy_nonoverlapping(src as *const u8,
                        tmp.as_mut_ptr() as *mut u8,
                        mem::size_of::<T>());
    tmp.assume_init()
}

/// Overwrites a memory location with the given value without reading or
/// dropping the old value.
///
/// `write` does not drop the contents of `dst`. This is safe, but it could leak
/// allocations or resources, so care should be taken not to overwrite an object
/// that should be dropped.
///
/// Additionally, it does not drop `src`. Semantically, `src` is moved into the
/// location pointed to by `dst`.
///
/// This is appropriate for initializing uninitialized memory, or overwriting
/// memory that has previously been [`read`] from.
///
/// [`read`]: ./fn.read.html
///
/// # Safety
///
/// Behavior is undefined if any of the following conditions are violated:
///
/// * `dst` must be [valid] for writes.
///
/// * `dst` must be properly aligned. Use [`write_unaligned`] if this is not the
///   case.
///
/// Note that even if `T` has size `0`, the pointer must be non-NULL and properly aligned.
///
/// [valid]: ../ptr/index.html#safety
/// [`write_unaligned`]: ./fn.write_unaligned.html
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
///
/// Manually implement [`mem::swap`]:
///
/// ```
/// use std::ptr;
///
/// fn swap<T>(a: &mut T, b: &mut T) {
///     unsafe {
///         // Create a bitwise copy of the value at `a` in `tmp`.
///         let tmp = ptr::read(a);
///
///         // Exiting at this point (either by explicitly returning or by
///         // calling a function which panics) would cause the value in `tmp` to
///         // be dropped while the same value is still referenced by `a`. This
///         // could trigger undefined behavior if `T` is not `Copy`.
///
///         // Create a bitwise copy of the value at `b` in `a`.
///         // This is safe because mutable references cannot alias.
///         ptr::copy_nonoverlapping(b, a, 1);
///
///         // As above, exiting here could trigger undefined behavior because
///         // the same value is referenced by `a` and `b`.
///
///         // Move `tmp` into `b`.
///         ptr::write(b, tmp);
///
///         // `tmp` has been moved (`write` takes ownership of its second argument),
///         // so nothing is dropped implicitly here.
///     }
/// }
///
/// let mut foo = "foo".to_owned();
/// let mut bar = "bar".to_owned();
///
/// swap(&mut foo, &mut bar);
///
/// assert_eq!(foo, "bar");
/// assert_eq!(bar, "foo");
/// ```
///
/// [`mem::swap`]: ../mem/fn.swap.html
#[inline]
#[stable(feature = "rust1", since = "1.0.0")]
pub unsafe fn write<T>(dst: *mut T, src: T) {
    intrinsics::move_val_init(&mut *dst, src)
}

/// Overwrites a memory location with the given value without reading or
/// dropping the old value.
///
/// Unlike [`write`], the pointer may be unaligned.
///
/// `write_unaligned` does not drop the contents of `dst`. This is safe, but it
/// could leak allocations or resources, so care should be taken not to overwrite
/// an object that should be dropped.
///
/// Additionally, it does not drop `src`. Semantically, `src` is moved into the
/// location pointed to by `dst`.
///
/// This is appropriate for initializing uninitialized memory, or overwriting
/// memory that has previously been read with [`read_unaligned`].
///
/// [`write`]: ./fn.write.html
/// [`read_unaligned`]: ./fn.read_unaligned.html
///
/// # Safety
///
/// Behavior is undefined if any of the following conditions are violated:
///
/// * `dst` must be [valid] for writes.
///
/// Note that even if `T` has size `0`, the pointer must be non-NULL.
///
/// [valid]: ../ptr/index.html#safety
///
/// ## On `packed` structs
///
/// It is currently impossible to create raw pointers to unaligned fields
/// of a packed struct.
///
/// Attempting to create a raw pointer to an `unaligned` struct field with
/// an expression such as `&packed.unaligned as *const FieldType` creates an
/// intermediate unaligned reference before converting that to a raw pointer.
/// That this reference is temporary and immediately cast is inconsequential
/// as the compiler always expects references to be properly aligned.
/// As a result, using `&packed.unaligned as *const FieldType` causes immediate
/// *undefined behavior* in your program.
///
/// An example of what not to do and how this relates to `write_unaligned` is:
///
/// ```no_run
/// #[repr(packed, C)]
/// struct Packed {
///     _padding: u8,
///     unaligned: u32,
/// }
///
/// let v = 0x01020304;
/// let mut packed: Packed = unsafe { std::mem::zeroed() };
///
/// let v = unsafe {
///     // Here we attempt to take the address of a 32-bit integer which is not aligned.
///     let unaligned =
///         // A temporary unaligned reference is created here which results in
///         // undefined behavior regardless of whether the reference is used or not.
///         &mut packed.unaligned
///         // Casting to a raw pointer doesn't help; the mistake already happened.
///         as *mut u32;
///
///     std::ptr::write_unaligned(unaligned, v);
///
///     v
/// };
/// ```
///
/// Accessing unaligned fields directly with e.g. `packed.unaligned` is safe however.
// FIXME: Update docs based on outcome of RFC #2582 and friends.
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
/// [`write_volatile`]: ./fn.write_volatile.html
///
/// # Notes
///
/// Rust does not currently have a rigorously and formally defined memory model,
/// so the precise semantics of what "volatile" means here is subject to change
/// over time. That being said, the semantics will almost always end up pretty
/// similar to [C11's definition of volatile][c11].
///
/// The compiler shouldn't change the relative order or number of volatile
/// memory operations. However, volatile memory operations on zero-sized types
/// (e.g., if a zero-sized type is passed to `read_volatile`) are noops
/// and may be ignored.
///
/// [c11]: http://www.open-std.org/jtc1/sc22/wg14/www/docs/n1570.pdf
///
/// # Safety
///
/// Behavior is undefined if any of the following conditions are violated:
///
/// * `src` must be [valid] for reads.
///
/// * `src` must be properly aligned.
///
/// Like [`read`], `read_volatile` creates a bitwise copy of `T`, regardless of
/// whether `T` is [`Copy`]. If `T` is not [`Copy`], using both the returned
/// value and the value at `*src` can [violate memory safety][read-ownership].
/// However, storing non-[`Copy`] types in volatile memory is almost certainly
/// incorrect.
///
/// Note that even if `T` has size `0`, the pointer must be non-NULL and properly aligned.
///
/// [valid]: ../ptr/index.html#safety
/// [`Copy`]: ../marker/trait.Copy.html
/// [`read`]: ./fn.read.html
/// [read-ownership]: ./fn.read.html#ownership-of-the-returned-value
///
/// Just like in C, whether an operation is volatile has no bearing whatsoever
/// on questions involving concurrent access from multiple threads. Volatile
/// accesses behave exactly like non-atomic accesses in that regard. In particular,
/// a race between a `read_volatile` and any write operation to the same location
/// is undefined behavior.
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
/// `write_volatile` does not drop the contents of `dst`. This is safe, but it
/// could leak allocations or resources, so care should be taken not to overwrite
/// an object that should be dropped.
///
/// Additionally, it does not drop `src`. Semantically, `src` is moved into the
/// location pointed to by `dst`.
///
/// [`read_volatile`]: ./fn.read_volatile.html
///
/// # Notes
///
/// Rust does not currently have a rigorously and formally defined memory model,
/// so the precise semantics of what "volatile" means here is subject to change
/// over time. That being said, the semantics will almost always end up pretty
/// similar to [C11's definition of volatile][c11].
///
/// The compiler shouldn't change the relative order or number of volatile
/// memory operations. However, volatile memory operations on zero-sized types
/// (e.g., if a zero-sized type is passed to `write_volatile`) are noops
/// and may be ignored.
///
/// [c11]: http://www.open-std.org/jtc1/sc22/wg14/www/docs/n1570.pdf
///
/// # Safety
///
/// Behavior is undefined if any of the following conditions are violated:
///
/// * `dst` must be [valid] for writes.
///
/// * `dst` must be properly aligned.
///
/// Note that even if `T` has size `0`, the pointer must be non-NULL and properly aligned.
///
/// [valid]: ../ptr/index.html#safety
///
/// Just like in C, whether an operation is volatile has no bearing whatsoever
/// on questions involving concurrent access from multiple threads. Volatile
/// accesses behave exactly like non-atomic accesses in that regard. In particular,
/// a race between a `write_volatile` and any other operation (reading or writing)
/// on the same location is undefined behavior.
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
    /// let s: &str = "Follow the rabbit";
    /// let ptr: *const u8 = s.as_ptr();
    /// assert!(!ptr.is_null());
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn is_null(self) -> bool {
        // Compare via a cast to a thin pointer, so fat pointers are only
        // considering their "data" part for null-ness.
        (self as *const u8) == null()
    }

    /// Cast to a pointer to a different type
    #[unstable(feature = "ptr_cast", issue = "60602")]
    #[inline]
    pub const fn cast<U>(self) -> *const U {
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
    /// non-NULL, then it is properly aligned, dereferencable (for the whole
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
    /// let ptr: *const u8 = &10u8 as *const u8;
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
    /// let ptr: *const u8 = &10u8 as *const u8;
    ///
    /// unsafe {
    ///     let val_back = &*ptr;
    ///     println!("We got back the value: {}!", val_back);
    /// }
    /// ```
    #[stable(feature = "ptr_as_ref", since = "1.9.0")]
    #[inline]
    pub unsafe fn as_ref<'a>(self) -> Option<&'a T> {
        if self.is_null() {
            None
        } else {
            Some(&*self)
        }
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
    ///   byte past the end of the same allocated object.
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
    /// Consider using `wrapping_offset` instead if these constraints are
    /// difficult to satisfy. The only advantage of this method is that it
    /// enables more aggressive compiler optimizations.
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
    ///
    /// `count` is in units of T; e.g., a `count` of 3 represents a pointer
    /// offset of `3 * size_of::<T>()` bytes.
    ///
    /// # Safety
    ///
    /// The resulting pointer does not need to be in bounds, but it is
    /// potentially hazardous to dereference (which requires `unsafe`).
    /// In particular, the resulting pointer may *not* be used to access a
    /// different allocated object than the one `self` points to. In other
    /// words, `x.wrapping_offset(y.wrapping_offset_from(x))` is
    /// *not* the same as `y`, and dereferencing it is undefined behavior
    /// unless `x` and `y` point into the same allocated object.
    ///
    /// Always use `.offset(count)` instead when possible, because `offset`
    /// allows the compiler to optimize better. If you need to cross object
    /// boundaries, cast the pointer to an integer and do the arithmetic there.
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
    /// This function is the inverse of [`offset`].
    ///
    /// [`offset`]: #method.offset
    /// [`wrapping_offset_from`]: #method.wrapping_offset_from
    ///
    /// # Safety
    ///
    /// If any of the following conditions are violated, the result is Undefined
    /// Behavior:
    ///
    /// * Both the starting and other pointer must be either in bounds or one
    ///   byte past the end of the same allocated object.
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
    #[unstable(feature = "ptr_offset_from", issue = "41079")]
    #[inline]
    pub unsafe fn offset_from(self, origin: *const T) -> isize where T: Sized {
        let pointee_size = mem::size_of::<T>();
        assert!(0 < pointee_size && pointee_size <= isize::max_value() as usize);

        // This is the same sequence that Clang emits for pointer subtraction.
        // It can be neither `nsw` nor `nuw` because the input is treated as
        // unsigned but then the output is treated as signed, so neither works.
        let d = isize::wrapping_sub(self as _, origin as _);
        intrinsics::exact_div(d, pointee_size as _)
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
    /// let a = [0; 5];
    /// let ptr1: *const i32 = &a[1];
    /// let ptr2: *const i32 = &a[3];
    /// assert_eq!(ptr2.wrapping_offset_from(ptr1), 2);
    /// assert_eq!(ptr1.wrapping_offset_from(ptr2), -2);
    /// assert_eq!(ptr1.wrapping_offset(2), ptr2);
    /// assert_eq!(ptr2.wrapping_offset(-2), ptr1);
    ///
    /// let ptr1: *const i32 = 3 as _;
    /// let ptr2: *const i32 = 13 as _;
    /// assert_eq!(ptr2.wrapping_offset_from(ptr1), 2);
    /// ```
    #[unstable(feature = "ptr_wrapping_offset_from", issue = "41079")]
    #[inline]
    pub fn wrapping_offset_from(self, origin: *const T) -> isize where T: Sized {
        let pointee_size = mem::size_of::<T>();
        assert!(0 < pointee_size && pointee_size <= isize::max_value() as usize);

        let d = isize::wrapping_sub(self as _, origin as _);
        d.wrapping_div(pointee_size as _)
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
    ///   byte past the end of the same allocated object.
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
    /// Consider using `wrapping_offset` instead if these constraints are
    /// difficult to satisfy. The only advantage of this method is that it
    /// enables more aggressive compiler optimizations.
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
    #[inline]
    pub unsafe fn add(self, count: usize) -> Self
        where T: Sized,
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
    ///   byte past the end of the same allocated object.
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
    /// Consider using `wrapping_offset` instead if these constraints are
    /// difficult to satisfy. The only advantage of this method is that it
    /// enables more aggressive compiler optimizations.
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
    #[inline]
    pub unsafe fn sub(self, count: usize) -> Self
        where T: Sized,
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
    /// Always use `.add(count)` instead when possible, because `add`
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
    #[inline]
    pub fn wrapping_add(self, count: usize) -> Self
        where T: Sized,
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
    /// Always use `.sub(count)` instead when possible, because `sub`
    /// allows the compiler to optimize better.
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
    #[inline]
    pub fn wrapping_sub(self, count: usize) -> Self
        where T: Sized,
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
        where T: Sized,
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
        where T: Sized,
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
        where T: Sized,
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
        where T: Sized,
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
        where T: Sized,
    {
        copy_nonoverlapping(self, dest, count)
    }

    /// Computes the offset that needs to be applied to the pointer in order to make it aligned to
    /// `align`.
    ///
    /// If it is not possible to align the pointer, the implementation returns
    /// `usize::max_value()`.
    ///
    /// The offset is expressed in number of `T` elements, and not bytes. The value returned can be
    /// used with the `offset` or `offset_to` methods.
    ///
    /// There are no guarantees whatsover that offsetting the pointer will not overflow or go
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
    pub fn align_offset(self, align: usize) -> usize where T: Sized {
        if !align.is_power_of_two() {
            panic!("align_offset: align is not a power-of-two");
        }
        unsafe {
            align_offset(self, align)
        }
    }
}


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

    /// Cast to a pointer to a different type
    #[unstable(feature = "ptr_cast", issue = "60602")]
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
    /// non-NULL, then it is properly aligned, dereferencable (for the whole
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
        if self.is_null() {
            None
        } else {
            Some(&*self)
        }
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
    ///   byte past the end of the same allocated object.
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
    /// Consider using `wrapping_offset` instead if these constraints are
    /// difficult to satisfy. The only advantage of this method is that it
    /// enables more aggressive compiler optimizations.
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
    /// `count` is in units of T; e.g., a `count` of 3 represents a pointer
    /// offset of `3 * size_of::<T>()` bytes.
    ///
    /// # Safety
    ///
    /// The resulting pointer does not need to be in bounds, but it is
    /// potentially hazardous to dereference (which requires `unsafe`).
    /// In particular, the resulting pointer may *not* be used to access a
    /// different allocated object than the one `self` points to. In other
    /// words, `x.wrapping_offset(y.wrapping_offset_from(x))` is
    /// *not* the same as `y`, and dereferencing it is undefined behavior
    /// unless `x` and `y` point into the same allocated object.
    ///
    /// Always use `.offset(count)` instead when possible, because `offset`
    /// allows the compiler to optimize better. If you need to cross object
    /// boundaries, cast the pointer to an integer and do the arithmetic there.
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
    /// As with [`as_ref`], this is unsafe because it cannot verify the validity
    /// of the returned pointer, nor can it ensure that the lifetime `'a`
    /// returned is indeed a valid lifetime for the contained data.
    ///
    /// When calling this method, you have to ensure that if the pointer is
    /// non-NULL, then it is properly aligned, dereferencable (for the whole
    /// size of `T`) and points to an initialized instance of `T`. This applies
    /// even if the result of this method is unused!
    /// (The part about being initialized is not yet fully decided, but until
    /// it is the only safe approach is to ensure that they are indeed initialized.)
    ///
    /// Additionally, the lifetime `'a` returned is arbitrarily chosen and does
    /// not necessarily reflect the actual lifetime of the data. It is up to the
    /// caller to ensure that for the duration of this lifetime, the memory this
    /// pointer points to does not get accessed through any other pointer.
    ///
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
    #[stable(feature = "ptr_as_ref", since = "1.9.0")]
    #[inline]
    pub unsafe fn as_mut<'a>(self) -> Option<&'a mut T> {
        if self.is_null() {
            None
        } else {
            Some(&mut *self)
        }
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
    ///   byte past the end of the same allocated object.
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
    #[inline]
    pub unsafe fn offset_from(self, origin: *const T) -> isize where T: Sized {
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
    pub fn wrapping_offset_from(self, origin: *const T) -> isize where T: Sized {
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
    ///   byte past the end of the same allocated object.
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
    /// Consider using `wrapping_offset` instead if these constraints are
    /// difficult to satisfy. The only advantage of this method is that it
    /// enables more aggressive compiler optimizations.
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
    #[inline]
    pub unsafe fn add(self, count: usize) -> Self
        where T: Sized,
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
    ///   byte past the end of the same allocated object.
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
    /// Consider using `wrapping_offset` instead if these constraints are
    /// difficult to satisfy. The only advantage of this method is that it
    /// enables more aggressive compiler optimizations.
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
    #[inline]
    pub unsafe fn sub(self, count: usize) -> Self
        where T: Sized,
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
    /// Always use `.add(count)` instead when possible, because `add`
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
    #[inline]
    pub fn wrapping_add(self, count: usize) -> Self
        where T: Sized,
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
    /// Always use `.sub(count)` instead when possible, because `sub`
    /// allows the compiler to optimize better.
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
    #[inline]
    pub fn wrapping_sub(self, count: usize) -> Self
        where T: Sized,
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
        where T: Sized,
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
        where T: Sized,
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
        where T: Sized,
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
        where T: Sized,
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
        where T: Sized,
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
        where T: Sized,
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
        where T: Sized,
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
        where T: Sized,
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
        where T: Sized,
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
        where T: Sized,
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
        where T: Sized,
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
        where T: Sized,
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
        where T: Sized,
    {
        swap(self, with)
    }

    /// Computes the offset that needs to be applied to the pointer in order to make it aligned to
    /// `align`.
    ///
    /// If it is not possible to align the pointer, the implementation returns
    /// `usize::max_value()`.
    ///
    /// The offset is expressed in number of `T` elements, and not bytes. The value returned can be
    /// used with the `offset` or `offset_to` methods.
    ///
    /// There are no guarantees whatsover that offsetting the pointer will not overflow or go
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
    pub fn align_offset(self, align: usize) -> usize where T: Sized {
        if !align.is_power_of_two() {
            panic!("align_offset: align is not a power-of-two");
        }
        unsafe {
            align_offset(self, align)
        }
    }
}

/// Align pointer `p`.
///
/// Calculate offset (in terms of elements of `stride` stride) that has to be applied
/// to pointer `p` so that pointer `p` would get aligned to `a`.
///
/// Note: This implementation has been carefully tailored to not panic. It is UB for this to panic.
/// The only real change that can be made here is change of `INV_TABLE_MOD_16` and associated
/// constants.
///
/// If we ever decide to make it possible to call the intrinsic with `a` that is not a
/// power-of-two, it will probably be more prudent to just change to a naive implementation rather
/// than trying to adapt this to accommodate that change.
///
/// Any questions go to @nagisa.
#[lang="align_offset"]
pub(crate) unsafe fn align_offset<T: Sized>(p: *const T, a: usize) -> usize {
    /// Calculate multiplicative modular inverse of `x` modulo `m`.
    ///
    /// This implementation is tailored for align_offset and has following preconditions:
    ///
    /// * `m` is a power-of-two;
    /// * `x < m`; (if `x ≥ m`, pass in `x % m` instead)
    ///
    /// Implementation of this function shall not panic. Ever.
    #[inline]
    fn mod_inv(x: usize, m: usize) -> usize {
        /// Multiplicative modular inverse table modulo 2⁴ = 16.
        ///
        /// Note, that this table does not contain values where inverse does not exist (i.e., for
        /// `0⁻¹ mod 16`, `2⁻¹ mod 16`, etc.)
        const INV_TABLE_MOD_16: [u8; 8] = [1, 11, 13, 7, 9, 3, 5, 15];
        /// Modulo for which the `INV_TABLE_MOD_16` is intended.
        const INV_TABLE_MOD: usize = 16;
        /// INV_TABLE_MOD²
        const INV_TABLE_MOD_SQUARED: usize = INV_TABLE_MOD * INV_TABLE_MOD;

        let table_inverse = INV_TABLE_MOD_16[(x & (INV_TABLE_MOD - 1)) >> 1] as usize;
        if m <= INV_TABLE_MOD {
            table_inverse & (m - 1)
        } else {
            // We iterate "up" using the following formula:
            //
            // $$ xy ≡ 1 (mod 2ⁿ) → xy (2 - xy) ≡ 1 (mod 2²ⁿ) $$
            //
            // until 2²ⁿ ≥ m. Then we can reduce to our desired `m` by taking the result `mod m`.
            let mut inverse = table_inverse;
            let mut going_mod = INV_TABLE_MOD_SQUARED;
            loop {
                // y = y * (2 - xy) mod n
                //
                // Note, that we use wrapping operations here intentionally – the original formula
                // uses e.g., subtraction `mod n`. It is entirely fine to do them `mod
                // usize::max_value()` instead, because we take the result `mod n` at the end
                // anyway.
                inverse = inverse.wrapping_mul(
                    2usize.wrapping_sub(x.wrapping_mul(inverse))
                ) & (going_mod - 1);
                if going_mod > m {
                    return inverse & (m - 1);
                }
                going_mod = going_mod.wrapping_mul(going_mod);
            }
        }
    }

    let stride = mem::size_of::<T>();
    let a_minus_one = a.wrapping_sub(1);
    let pmoda = p as usize & a_minus_one;

    if pmoda == 0 {
        // Already aligned. Yay!
        return 0;
    }

    if stride <= 1 {
        return if stride == 0 {
            // If the pointer is not aligned, and the element is zero-sized, then no amount of
            // elements will ever align the pointer.
            !0
        } else {
            a.wrapping_sub(pmoda)
        };
    }

    let smoda = stride & a_minus_one;
    // a is power-of-two so cannot be 0. stride = 0 is handled above.
    let gcdpow = intrinsics::cttz_nonzero(stride).min(intrinsics::cttz_nonzero(a));
    let gcd = 1usize << gcdpow;

    if p as usize & (gcd - 1) == 0 {
        // This branch solves for the following linear congruence equation:
        //
        // $$ p + so ≡ 0 mod a $$
        //
        // $p$ here is the pointer value, $s$ – stride of `T`, $o$ offset in `T`s, and $a$ – the
        // requested alignment.
        //
        // g = gcd(a, s)
        // o = (a - (p mod a))/g * ((s/g)⁻¹ mod a)
        //
        // The first term is “the relative alignment of p to a”, the second term is “how does
        // incrementing p by s bytes change the relative alignment of p”. Division by `g` is
        // necessary to make this equation well formed if $a$ and $s$ are not co-prime.
        //
        // Furthermore, the result produced by this solution is not “minimal”, so it is necessary
        // to take the result $o mod lcm(s, a)$. We can replace $lcm(s, a)$ with just a $a / g$.
        let j = a.wrapping_sub(pmoda) >> gcdpow;
        let k = smoda >> gcdpow;
        return intrinsics::unchecked_rem(j.wrapping_mul(mod_inv(k, a)), a >> gcdpow);
    }

    // Cannot be aligned at all.
    usize::max_value()
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

/// Compares raw pointers for equality.
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
/// assert!(ptr::eq(five_ref, same_five_ref));
///
/// assert!(five_ref == other_five_ref);
/// assert!(!ptr::eq(five_ref, other_five_ref));
/// ```
///
/// Slices are also compared by their length (fat pointers):
///
/// ```
/// let a = [1, 2, 3];
/// assert!(std::ptr::eq(&a[..3], &a[..3]));
/// assert!(!std::ptr::eq(&a[..2], &a[..3]));
/// assert!(!std::ptr::eq(&a[0..2], &a[1..3]));
/// ```
///
/// Traits are also compared by their implementation:
///
/// ```
/// #[repr(transparent)]
/// struct Wrapper { member: i32 }
///
/// trait Trait {}
/// impl Trait for Wrapper {}
/// impl Trait for i32 {}
///
/// fn main() {
///     let wrapper = Wrapper { member: 10 };
///
///     // Pointers have equal addresses.
///     assert!(std::ptr::eq(
///         &wrapper as *const Wrapper as *const u8,
///         &wrapper.member as *const i32 as *const u8
///     ));
///
///     // Objects have equal addresses, but `Trait` has different implementations.
///     assert!(!std::ptr::eq(
///         &wrapper as &dyn Trait,
///         &wrapper.member as &dyn Trait,
///     ));
///     assert!(!std::ptr::eq(
///         &wrapper as &dyn Trait as *const dyn Trait,
///         &wrapper.member as &dyn Trait as *const dyn Trait,
///     ));
///
///     // Converting the reference to a `*const u8` compares by address.
///     assert!(std::ptr::eq(
///         &wrapper as &dyn Trait as *const dyn Trait as *const u8,
///         &wrapper.member as &dyn Trait as *const dyn Trait as *const u8,
///     ));
/// }
/// ```
#[stable(feature = "ptr_eq", since = "1.17.0")]
#[inline]
pub fn eq<T: ?Sized>(a: *const T, b: *const T) -> bool {
    a == b
}

/// Hash a raw pointer.
///
/// This can be used to hash a `&T` reference (which coerces to `*const T` implicitly)
/// by its address rather than the value it points to
/// (which is what the `Hash for &T` implementation does).
///
/// # Examples
///
/// ```
/// use std::collections::hash_map::DefaultHasher;
/// use std::hash::{Hash, Hasher};
/// use std::ptr;
///
/// let five = 5;
/// let five_ref = &five;
///
/// let mut hasher = DefaultHasher::new();
/// ptr::hash(five_ref, &mut hasher);
/// let actual = hasher.finish();
///
/// let mut hasher = DefaultHasher::new();
/// (five_ref as *const i32).hash(&mut hasher);
/// let expected = hasher.finish();
///
/// assert_eq!(actual, expected);
/// ```
#[stable(feature = "ptr_hash", since = "1.35.0")]
pub fn hash<T: ?Sized, S: hash::Hasher>(hashee: *const T, into: &mut S) {
    use crate::hash::Hash;
    hashee.hash(into);
}

// Impls for function pointers
macro_rules! fnptr_impls_safety_abi {
    ($FnTy: ty, $($Arg: ident),*) => {
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
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                fmt::Pointer::fmt(&(*self as *const ()), f)
            }
        }

        #[stable(feature = "fnptr_impls", since = "1.4.0")]
        impl<Ret, $($Arg),*> fmt::Debug for $FnTy {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                fmt::Pointer::fmt(&(*self as *const ()), f)
            }
        }
    }
}

macro_rules! fnptr_impls_args {
    ($($Arg: ident),+) => {
        fnptr_impls_safety_abi! { extern "Rust" fn($($Arg),+) -> Ret, $($Arg),+ }
        fnptr_impls_safety_abi! { extern "C" fn($($Arg),+) -> Ret, $($Arg),+ }
        fnptr_impls_safety_abi! { extern "C" fn($($Arg),+ , ...) -> Ret, $($Arg),+ }
        fnptr_impls_safety_abi! { unsafe extern "Rust" fn($($Arg),+) -> Ret, $($Arg),+ }
        fnptr_impls_safety_abi! { unsafe extern "C" fn($($Arg),+) -> Ret, $($Arg),+ }
        fnptr_impls_safety_abi! { unsafe extern "C" fn($($Arg),+ , ...) -> Ret, $($Arg),+ }
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
