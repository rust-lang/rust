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
//! * For a pointer to be valid, it is necessary, but not always sufficient, that the pointer
//!   be *dereferenceable*: the memory range of the given size starting at the pointer must all be
//!   within the bounds of a single allocated object. Note that in Rust,
//!   every (stack-allocated) variable is considered a separate allocated object.
//! * Even for operations of [size zero][zst], the pointer must not be pointing to deallocated
//!   memory, i.e., deallocation makes pointers invalid even for zero-sized operations. However,
//!   casting any non-zero integer *literal* to a pointer is valid for zero-sized accesses, even if
//!   some memory happens to exist at that address and gets deallocated. This corresponds to writing
//!   your own allocator: allocating zero-sized objects is not very hard. The canonical way to
//!   obtain a pointer that is valid for zero-sized accesses is [`NonNull::dangling`].
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
//! [zst]: ../../nomicon/exotic-sizes.html#zero-sized-types-zsts
//! [atomic operations]: crate::sync::atomic
//! [`offset`]: ../../std/primitive.pointer.html#method.offset

#![stable(feature = "rust1", since = "1.0.0")]

use crate::cmp::Ordering;
use crate::fmt;
use crate::hash;
use crate::intrinsics::{self, abort, is_aligned_and_not_null, is_nonoverlapping};
use crate::mem::{self, MaybeUninit};

#[stable(feature = "rust1", since = "1.0.0")]
#[doc(inline)]
pub use crate::intrinsics::copy_nonoverlapping;

#[stable(feature = "rust1", since = "1.0.0")]
#[doc(inline)]
pub use crate::intrinsics::copy;

#[stable(feature = "rust1", since = "1.0.0")]
#[doc(inline)]
pub use crate::intrinsics::write_bytes;

mod non_null;
#[stable(feature = "nonnull", since = "1.25.0")]
pub use non_null::NonNull;

mod unique;
#[unstable(feature = "ptr_internals", issue = "none")]
pub use unique::Unique;

mod const_ptr;
mod mut_ptr;

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
///   dropping manually allocated memory (e.g., in the implementations of
///   `Box`/`Rc`/`Vec`), as the compiler doesn't need to prove that it's
///   sound to elide the copy.
///
/// * It can be used to drop [pinned] data when `T` is not `repr(packed)`
///   (pinned data must not be moved before it is dropped).
///
/// Unaligned values cannot be dropped in place, they must be copied to an aligned
/// location first using [`ptr::read_unaligned`]. For packed structs, this move is
/// done automatically by the compiler. This means the fields of packed structs
/// are not dropped in-place.
///
/// [`ptr::read`]: self::read
/// [`ptr::read_unaligned`]: self::read_unaligned
/// [pinned]: crate::pin
///
/// # Safety
///
/// Behavior is undefined if any of the following conditions are violated:
///
/// * `to_drop` must be [valid] for both reads and writes.
///
/// * `to_drop` must be properly aligned.
///
/// * The value `to_drop` points to must be valid for dropping, which may mean it must uphold
///   additional invariants - this is type-dependent.
///
/// Additionally, if `T` is not [`Copy`], using the pointed-to value after
/// calling `drop_in_place` can cause undefined behavior. Note that `*to_drop =
/// foo` counts as a use because it will cause the value to be dropped
/// again. [`write()`] can be used to overwrite data without causing it to be
/// dropped.
///
/// Note that even if `T` has size `0`, the pointer must be non-NULL and properly aligned.
///
/// [valid]: self#safety
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
#[lang = "drop_in_place"]
#[allow(unconditional_recursion)]
pub unsafe fn drop_in_place<T: ?Sized>(to_drop: *mut T) {
    // Code here does not matter - this is replaced by the
    // real drop glue by the compiler.

    // SAFETY: see comment above
    unsafe { drop_in_place(to_drop) }
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
#[inline(always)]
#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_promotable]
#[rustc_const_stable(feature = "const_ptr_null", since = "1.32.0")]
pub const fn null<T>() -> *const T {
    0 as *const T
}

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
#[inline(always)]
#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_promotable]
#[rustc_const_stable(feature = "const_ptr_null", since = "1.32.0")]
pub const fn null_mut<T>() -> *mut T {
    0 as *mut T
}

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

// Manual impl needed to avoid `T: Clone` bound.
impl<T> Clone for FatPtr<T> {
    fn clone(&self) -> Self {
        *self
    }
}

// Manual impl needed to avoid `T: Copy` bound.
impl<T> Copy for FatPtr<T> {}

/// Forms a raw slice from a pointer and a length.
///
/// The `len` argument is the number of **elements**, not the number of bytes.
///
/// This function is safe, but actually using the return value is unsafe.
/// See the documentation of [`slice::from_raw_parts`] for slice safety requirements.
///
/// [`slice::from_raw_parts`]: crate::slice::from_raw_parts
///
/// # Examples
///
/// ```rust
/// use std::ptr;
///
/// // create a slice pointer when starting out with a pointer to the first element
/// let x = [5, 6, 7];
/// let raw_pointer = x.as_ptr();
/// let slice = ptr::slice_from_raw_parts(raw_pointer, 3);
/// assert_eq!(unsafe { &*slice }[2], 7);
/// ```
#[inline]
#[stable(feature = "slice_from_raw_parts", since = "1.42.0")]
#[rustc_const_unstable(feature = "const_slice_from_raw_parts", issue = "67456")]
pub const fn slice_from_raw_parts<T>(data: *const T, len: usize) -> *const [T] {
    // SAFETY: Accessing the value from the `Repr` union is safe since *const [T]
    // and FatPtr have the same memory layouts. Only std can make this
    // guarantee.
    unsafe { Repr { raw: FatPtr { data, len } }.rust }
}

/// Performs the same functionality as [`slice_from_raw_parts`], except that a
/// raw mutable slice is returned, as opposed to a raw immutable slice.
///
/// See the documentation of [`slice_from_raw_parts`] for more details.
///
/// This function is safe, but actually using the return value is unsafe.
/// See the documentation of [`slice::from_raw_parts_mut`] for slice safety requirements.
///
/// [`slice::from_raw_parts_mut`]: crate::slice::from_raw_parts_mut
///
/// # Examples
///
/// ```rust
/// use std::ptr;
///
/// let x = &mut [5, 6, 7];
/// let raw_pointer = x.as_mut_ptr();
/// let slice = ptr::slice_from_raw_parts_mut(raw_pointer, 3);
///
/// unsafe {
///     (*slice)[2] = 99; // assign a value at an index in the slice
/// };
///
/// assert_eq!(unsafe { &*slice }[2], 99);
/// ```
#[inline]
#[stable(feature = "slice_from_raw_parts", since = "1.42.0")]
#[rustc_const_unstable(feature = "const_slice_from_raw_parts", issue = "67456")]
pub const fn slice_from_raw_parts_mut<T>(data: *mut T, len: usize) -> *mut [T] {
    // SAFETY: Accessing the value from the `Repr` union is safe since *mut [T]
    // and FatPtr have the same memory layouts
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
/// # Safety
///
/// Behavior is undefined if any of the following conditions are violated:
///
/// * Both `x` and `y` must be [valid] for both reads and writes.
///
/// * Both `x` and `y` must be properly aligned.
///
/// Note that even if `T` has size `0`, the pointers must be non-NULL and properly aligned.
///
/// [valid]: self#safety
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
    // SAFETY: the caller must guarantee that `x` and `y` are
    // valid for writes and properly aligned. `tmp` cannot be
    // overlapping either `x` or `y` because `tmp` was just allocated
    // on the stack as a separate allocated object.
    unsafe {
        copy_nonoverlapping(x, tmp.as_mut_ptr(), 1);
        copy(y, x, 1); // `x` and `y` may overlap
        copy_nonoverlapping(tmp.as_ptr(), y, 1);
    }
}

/// Swaps `count * size_of::<T>()` bytes between the two regions of memory
/// beginning at `x` and `y`. The two regions must *not* overlap.
///
/// # Safety
///
/// Behavior is undefined if any of the following conditions are violated:
///
/// * Both `x` and `y` must be [valid] for both reads and writes of `count *
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
/// [valid]: self#safety
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
    if cfg!(debug_assertions)
        && !(is_aligned_and_not_null(x)
            && is_aligned_and_not_null(y)
            && is_nonoverlapping(x, y, count))
    {
        // Not panicking to keep codegen impact smaller.
        abort();
    }

    let x = x as *mut u8;
    let y = y as *mut u8;
    let len = mem::size_of::<T>() * count;
    // SAFETY: the caller must guarantee that `x` and `y` are
    // valid for writes and properly aligned.
    unsafe { swap_nonoverlapping_bytes(x, y, len) }
}

#[inline]
pub(crate) unsafe fn swap_nonoverlapping_one<T>(x: *mut T, y: *mut T) {
    // For types smaller than the block optimization below,
    // just swap directly to avoid pessimizing codegen.
    if mem::size_of::<T>() < 32 {
        // SAFETY: the caller must guarantee that `x` and `y` are valid
        // for writes, properly aligned, and non-overlapping.
        unsafe {
            let z = read(x);
            copy_nonoverlapping(y, x, 1);
            write(y, z);
        }
    } else {
        // SAFETY: the caller must uphold the safety contract for `swap_nonoverlapping`.
        unsafe { swap_nonoverlapping(x, y, 1) };
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

        // SAFETY: As `i < len`, and as the caller must guarantee that `x` and `y` are valid
        // for `len` bytes, `x + i` and `y + i` must be valid adresses, which fulfills the
        // safety contract for `add`.
        //
        // Also, the caller must guarantee that `x` and `y` are valid for writes, properly aligned,
        // and non-overlapping, which fulfills the safety contract for `copy_nonoverlapping`.
        unsafe {
            let x = x.add(i);
            let y = y.add(i);

            // Swap a block of bytes of x & y, using t as a temporary buffer
            // This should be optimized into efficient SIMD operations where available
            copy_nonoverlapping(x, t, block_size);
            copy_nonoverlapping(y, x, block_size);
            copy_nonoverlapping(t, y, block_size);
        }
        i += block_size;
    }

    if i < len {
        // Swap any remaining bytes
        let mut t = mem::MaybeUninit::<UnalignedBlock>::uninit();
        let rem = len - i;

        let t = t.as_mut_ptr() as *mut u8;

        // SAFETY: see previous safety comment.
        unsafe {
            let x = x.add(i);
            let y = y.add(i);

            copy_nonoverlapping(x, t, rem);
            copy_nonoverlapping(y, x, rem);
            copy_nonoverlapping(t, y, rem);
        }
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
/// # Safety
///
/// Behavior is undefined if any of the following conditions are violated:
///
/// * `dst` must be [valid] for both reads and writes.
///
/// * `dst` must be properly aligned.
///
/// * `dst` must point to a properly initialized value of type `T`.
///
/// Note that even if `T` has size `0`, the pointer must be non-NULL and properly aligned.
///
/// [valid]: self#safety
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
    // SAFETY: the caller must guarantee that `dst` is valid to be
    // cast to a mutable reference (valid for writes, aligned, initialized),
    // and cannot overlap `src` since `dst` must point to a distinct
    // allocated object.
    unsafe {
        mem::swap(&mut *dst, &mut src); // cannot overlap
    }
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
/// * `src` must point to a properly initialized value of type `T`.
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
/// [`write()`] can be used to overwrite data without causing it to be dropped.
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
/// [valid]: self#safety
#[inline]
#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_const_unstable(feature = "const_ptr_read", issue = "80377")]
pub const unsafe fn read<T>(src: *const T) -> T {
    // `copy_nonoverlapping` takes care of debug_assert.
    let mut tmp = MaybeUninit::<T>::uninit();
    // SAFETY: the caller must guarantee that `src` is valid for reads.
    // `src` cannot overlap `tmp` because `tmp` was just allocated on
    // the stack as a separate allocated object.
    //
    // Also, since we just wrote a valid value into `tmp`, it is guaranteed
    // to be properly initialized.
    unsafe {
        copy_nonoverlapping(src, tmp.as_mut_ptr(), 1);
        tmp.assume_init()
    }
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
/// * `src` must point to a properly initialized value of type `T`.
///
/// Like [`read`], `read_unaligned` creates a bitwise copy of `T`, regardless of
/// whether `T` is [`Copy`]. If `T` is not [`Copy`], using both the returned
/// value and the value at `*src` can [violate memory safety][read-ownership].
///
/// Note that even if `T` has size `0`, the pointer must be non-NULL.
///
/// [read-ownership]: read#ownership-of-the-returned-value
/// [valid]: self#safety
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
///
/// # Examples
///
/// Read an usize value from a byte buffer:
///
/// ```
/// use std::mem;
///
/// fn read_usize(x: &[u8]) -> usize {
///     assert!(x.len() >= mem::size_of::<usize>());
///
///     let ptr = x.as_ptr() as *const usize;
///
///     unsafe { ptr.read_unaligned() }
/// }
/// ```
#[inline]
#[stable(feature = "ptr_unaligned", since = "1.17.0")]
#[rustc_const_unstable(feature = "const_ptr_read", issue = "80377")]
pub const unsafe fn read_unaligned<T>(src: *const T) -> T {
    // `copy_nonoverlapping` takes care of debug_assert.
    let mut tmp = MaybeUninit::<T>::uninit();
    // SAFETY: the caller must guarantee that `src` is valid for reads.
    // `src` cannot overlap `tmp` because `tmp` was just allocated on
    // the stack as a separate allocated object.
    //
    // Also, since we just wrote a valid value into `tmp`, it is guaranteed
    // to be properly initialized.
    unsafe {
        copy_nonoverlapping(src as *const u8, tmp.as_mut_ptr() as *mut u8, mem::size_of::<T>());
        tmp.assume_init()
    }
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
/// [valid]: self#safety
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
#[inline]
#[stable(feature = "rust1", since = "1.0.0")]
pub unsafe fn write<T>(dst: *mut T, src: T) {
    if cfg!(debug_assertions) && !is_aligned_and_not_null(dst) {
        // Not panicking to keep codegen impact smaller.
        abort();
    }
    // SAFETY: the caller must uphold the safety contract for `move_val_init`.
    unsafe { intrinsics::move_val_init(&mut *dst, src) }
}

/// Overwrites a memory location with the given value without reading or
/// dropping the old value.
///
/// Unlike [`write()`], the pointer may be unaligned.
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
/// # Safety
///
/// Behavior is undefined if any of the following conditions are violated:
///
/// * `dst` must be [valid] for writes.
///
/// Note that even if `T` has size `0`, the pointer must be non-NULL.
///
/// [valid]: self#safety
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
///
/// # Examples
///
/// Write an usize value to a byte buffer:
///
/// ```
/// use std::mem;
///
/// fn write_usize(x: &mut [u8], val: usize) {
///     assert!(x.len() >= mem::size_of::<usize>());
///
///     let ptr = x.as_mut_ptr() as *mut usize;
///
///     unsafe { ptr.write_unaligned(val) }
/// }
/// ```
#[inline]
#[stable(feature = "ptr_unaligned", since = "1.17.0")]
pub unsafe fn write_unaligned<T>(dst: *mut T, src: T) {
    // SAFETY: the caller must guarantee that `dst` is valid for writes.
    // `dst` cannot overlap `src` because the caller has mutable access
    // to `dst` while `src` is owned by this function.
    unsafe {
        // `copy_nonoverlapping` takes care of debug_assert.
        copy_nonoverlapping(&src as *const T as *const u8, dst as *mut u8, mem::size_of::<T>());
    }
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
/// * `src` must point to a properly initialized value of type `T`.
///
/// Like [`read`], `read_volatile` creates a bitwise copy of `T`, regardless of
/// whether `T` is [`Copy`]. If `T` is not [`Copy`], using both the returned
/// value and the value at `*src` can [violate memory safety][read-ownership].
/// However, storing non-[`Copy`] types in volatile memory is almost certainly
/// incorrect.
///
/// Note that even if `T` has size `0`, the pointer must be non-NULL and properly aligned.
///
/// [valid]: self#safety
/// [read-ownership]: read#ownership-of-the-returned-value
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
    if cfg!(debug_assertions) && !is_aligned_and_not_null(src) {
        // Not panicking to keep codegen impact smaller.
        abort();
    }
    // SAFETY: the caller must uphold the safety contract for `volatile_load`.
    unsafe { intrinsics::volatile_load(src) }
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
/// [valid]: self#safety
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
    if cfg!(debug_assertions) && !is_aligned_and_not_null(dst) {
        // Not panicking to keep codegen impact smaller.
        abort();
    }
    // SAFETY: the caller must uphold the safety contract for `volatile_store`.
    unsafe {
        intrinsics::volatile_store(dst, src);
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
#[lang = "align_offset"]
pub(crate) unsafe fn align_offset<T: Sized>(p: *const T, a: usize) -> usize {
    // FIXME(#75598): Direct use of these intrinsics improves codegen significantly at opt-level <=
    // 1, where the method versions of these operations are not inlined.
    use intrinsics::{
        unchecked_shl, unchecked_shr, unchecked_sub, wrapping_add, wrapping_mul, wrapping_sub,
    };

    /// Calculate multiplicative modular inverse of `x` modulo `m`.
    ///
    /// This implementation is tailored for `align_offset` and has following preconditions:
    ///
    /// * `m` is a power-of-two;
    /// * `x < m`; (if `x ≥ m`, pass in `x % m` instead)
    ///
    /// Implementation of this function shall not panic. Ever.
    #[inline]
    unsafe fn mod_inv(x: usize, m: usize) -> usize {
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
        // SAFETY: `m` is required to be a power-of-two, hence non-zero.
        let m_minus_one = unsafe { unchecked_sub(m, 1) };
        if m <= INV_TABLE_MOD {
            table_inverse & m_minus_one
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
                // usize::MAX` instead, because we take the result `mod n` at the end
                // anyway.
                inverse = wrapping_mul(inverse, wrapping_sub(2usize, wrapping_mul(x, inverse)));
                if going_mod >= m {
                    return inverse & m_minus_one;
                }
                going_mod = wrapping_mul(going_mod, going_mod);
            }
        }
    }

    let stride = mem::size_of::<T>();
    // SAFETY: `a` is a power-of-two, therefore non-zero.
    let a_minus_one = unsafe { unchecked_sub(a, 1) };
    if stride == 1 {
        // `stride == 1` case can be computed more simply through `-p (mod a)`, but doing so
        // inhibits LLVM's ability to select instructions like `lea`. Instead we compute
        //
        //    round_up_to_next_alignment(p, a) - p
        //
        // which distributes operations around the load-bearing, but pessimizing `and` sufficiently
        // for LLVM to be able to utilize the various optimizations it knows about.
        return wrapping_sub(
            wrapping_add(p as usize, a_minus_one) & wrapping_sub(0, a),
            p as usize,
        );
    }

    let pmoda = p as usize & a_minus_one;
    if pmoda == 0 {
        // Already aligned. Yay!
        return 0;
    } else if stride == 0 {
        // If the pointer is not aligned, and the element is zero-sized, then no amount of
        // elements will ever align the pointer.
        return usize::MAX;
    }

    let smoda = stride & a_minus_one;
    // SAFETY: a is power-of-two hence non-zero. stride == 0 case is handled above.
    let gcdpow = unsafe { intrinsics::cttz_nonzero(stride).min(intrinsics::cttz_nonzero(a)) };
    // SAFETY: gcdpow has an upper-bound that’s at most the number of bits in an usize.
    let gcd = unsafe { unchecked_shl(1usize, gcdpow) };

    // SAFETY: gcd is always greater or equal to 1.
    if p as usize & unsafe { unchecked_sub(gcd, 1) } == 0 {
        // This branch solves for the following linear congruence equation:
        //
        // ` p + so = 0 mod a `
        //
        // `p` here is the pointer value, `s` - stride of `T`, `o` offset in `T`s, and `a` - the
        // requested alignment.
        //
        // With `g = gcd(a, s)`, and the above condition asserting that `p` is also divisible by
        // `g`, we can denote `a' = a/g`, `s' = s/g`, `p' = p/g`, then this becomes equivalent to:
        //
        // ` p' + s'o = 0 mod a' `
        // ` o = (a' - (p' mod a')) * (s'^-1 mod a') `
        //
        // The first term is "the relative alignment of `p` to `a`" (divided by the `g`), the second
        // term is "how does incrementing `p` by `s` bytes change the relative alignment of `p`" (again
        // divided by `g`).
        // Division by `g` is necessary to make the inverse well formed if `a` and `s` are not
        // co-prime.
        //
        // Furthermore, the result produced by this solution is not "minimal", so it is necessary
        // to take the result `o mod lcm(s, a)`. We can replace `lcm(s, a)` with just a `a'`.

        // SAFETY: `gcdpow` has an upper-bound not greater than the number of trailing 0-bits in
        // `a`.
        let a2 = unsafe { unchecked_shr(a, gcdpow) };
        // SAFETY: `a2` is non-zero. Shifting `a` by `gcdpow` cannot shift out any of the set bits
        // in `a` (of which it has exactly one).
        let a2minus1 = unsafe { unchecked_sub(a2, 1) };
        // SAFETY: `gcdpow` has an upper-bound not greater than the number of trailing 0-bits in
        // `a`.
        let s2 = unsafe { unchecked_shr(smoda, gcdpow) };
        // SAFETY: `gcdpow` has an upper-bound not greater than the number of trailing 0-bits in
        // `a`. Furthermore, the subtraction cannot overflow, because `a2 = a >> gcdpow` will
        // always be strictly greater than `(p % a) >> gcdpow`.
        let minusp2 = unsafe { unchecked_sub(a2, unchecked_shr(pmoda, gcdpow)) };
        // SAFETY: `a2` is a power-of-two, as proven above. `s2` is strictly less than `a2`
        // because `(s % a) >> gcdpow` is strictly less than `a >> gcdpow`.
        return wrapping_mul(minusp2, unsafe { mod_inv(s2, a2) }) & a2minus1;
    }

    // Cannot be aligned at all.
    usize::MAX
}

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
/// let wrapper = Wrapper { member: 10 };
///
/// // Pointers have equal addresses.
/// assert!(std::ptr::eq(
///     &wrapper as *const Wrapper as *const u8,
///     &wrapper.member as *const i32 as *const u8
/// ));
///
/// // Objects have equal addresses, but `Trait` has different implementations.
/// assert!(!std::ptr::eq(
///     &wrapper as &dyn Trait,
///     &wrapper.member as &dyn Trait,
/// ));
/// assert!(!std::ptr::eq(
///     &wrapper as &dyn Trait as *const dyn Trait,
///     &wrapper.member as &dyn Trait as *const dyn Trait,
/// ));
///
/// // Converting the reference to a `*const u8` compares by address.
/// assert!(std::ptr::eq(
///     &wrapper as &dyn Trait as *const dyn Trait as *const u8,
///     &wrapper.member as &dyn Trait as *const dyn Trait as *const u8,
/// ));
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
                // HACK: The intermediate cast as usize is required for AVR
                // so that the address space of the source function pointer
                // is preserved in the final function pointer.
                //
                // https://github.com/avr-rust/rust/issues/143
                fmt::Pointer::fmt(&(*self as usize as *const ()), f)
            }
        }

        #[stable(feature = "fnptr_impls", since = "1.4.0")]
        impl<Ret, $($Arg),*> fmt::Debug for $FnTy {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                // HACK: The intermediate cast as usize is required for AVR
                // so that the address space of the source function pointer
                // is preserved in the final function pointer.
                //
                // https://github.com/avr-rust/rust/issues/143
                fmt::Pointer::fmt(&(*self as usize as *const ()), f)
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

fnptr_impls_args! {}
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

/// Create a `const` raw pointer to a place, without creating an intermediate reference.
///
/// Creating a reference with `&`/`&mut` is only allowed if the pointer is properly aligned
/// and points to initialized data. For cases where those requirements do not hold,
/// raw pointers should be used instead. However, `&expr as *const _` creates a reference
/// before casting it to a raw pointer, and that reference is subject to the same rules
/// as all other references. This macro can create a raw pointer *without* creating
/// a reference first.
///
/// # Example
///
/// ```
/// #![feature(raw_ref_macros)]
/// use std::ptr;
///
/// #[repr(packed)]
/// struct Packed {
///     f1: u8,
///     f2: u16,
/// }
///
/// let packed = Packed { f1: 1, f2: 2 };
/// // `&packed.f2` would create an unaligned reference, and thus be Undefined Behavior!
/// let raw_f2 = ptr::raw_const!(packed.f2);
/// assert_eq!(unsafe { raw_f2.read_unaligned() }, 2);
/// ```
#[unstable(feature = "raw_ref_macros", issue = "73394")]
#[rustc_macro_transparency = "semitransparent"]
#[allow_internal_unstable(raw_ref_op)]
pub macro raw_const($e:expr) {
    &raw const $e
}

/// Create a `mut` raw pointer to a place, without creating an intermediate reference.
///
/// Creating a reference with `&`/`&mut` is only allowed if the pointer is properly aligned
/// and points to initialized data. For cases where those requirements do not hold,
/// raw pointers should be used instead. However, `&mut expr as *mut _` creates a reference
/// before casting it to a raw pointer, and that reference is subject to the same rules
/// as all other references. This macro can create a raw pointer *without* creating
/// a reference first.
///
/// # Example
///
/// ```
/// #![feature(raw_ref_macros)]
/// use std::ptr;
///
/// #[repr(packed)]
/// struct Packed {
///     f1: u8,
///     f2: u16,
/// }
///
/// let mut packed = Packed { f1: 1, f2: 2 };
/// // `&mut packed.f2` would create an unaligned reference, and thus be Undefined Behavior!
/// let raw_f2 = ptr::raw_mut!(packed.f2);
/// unsafe { raw_f2.write_unaligned(42); }
/// assert_eq!({packed.f2}, 42); // `{...}` forces copying the field instead of creating a reference.
/// ```
#[unstable(feature = "raw_ref_macros", issue = "73394")]
#[rustc_macro_transparency = "semitransparent"]
#[allow_internal_unstable(raw_ref_op)]
pub macro raw_mut($e:expr) {
    &raw mut $e
}
