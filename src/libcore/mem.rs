//! Basic functions for dealing with memory.
//!
//! This module contains functions for querying the size and alignment of
//! types, initializing and manipulating memory.

#![stable(feature = "rust1", since = "1.0.0")]

use clone;
use cmp;
use fmt;
use hash;
use intrinsics;
use marker::{Copy, PhantomData, Sized};
use ptr;
use ops::{Deref, DerefMut};

#[stable(feature = "rust1", since = "1.0.0")]
#[doc(inline)]
pub use intrinsics::transmute;

/// Takes ownership and "forgets" about the value **without running its destructor**.
///
/// Any resources the value manages, such as heap memory or a file handle, will linger
/// forever in an unreachable state. However, it does not guarantee that pointers
/// to this memory will remain valid.
///
/// * If you want to leak memory, see [`Box::leak`][leak].
/// * If you want to obtain a raw pointer to the memory, see [`Box::into_raw`][into_raw].
/// * If you want to dispose of a value properly, running its destructor, see
/// [`mem::drop`][drop].
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
/// That said, leaking resources such as memory or I/O objects is usually undesirable,
/// so `forget` is only recommended for specialized use cases like those shown below.
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
/// Leak an I/O object, never closing the file:
///
/// ```no_run
/// use std::mem;
/// use std::fs::File;
///
/// let file = File::open("foo.txt").unwrap();
/// mem::forget(file);
/// ```
///
/// The practical use cases for `forget` are rather specialized and mainly come
/// up in unsafe or FFI code.
///
/// ## Use case 1
///
/// You have created an uninitialized value using [`mem::uninitialized`][uninit].
/// You must either initialize or `forget` it on every computation path before
/// Rust drops it automatically, like at the end of a scope or after a panic.
/// Running the destructor on an uninitialized value would be [undefined behavior][ub].
///
/// ```
/// use std::mem;
/// use std::ptr;
///
/// # let some_condition = false;
/// unsafe {
///     let mut uninit_vec: Vec<u32> = mem::uninitialized();
///
///     if some_condition {
///         // Initialize the variable.
///         ptr::write(&mut uninit_vec, Vec::new());
///     } else {
///         // Forget the uninitialized value so its destructor doesn't run.
///         mem::forget(uninit_vec);
///     }
/// }
/// ```
///
/// ## Use case 2
///
/// You have duplicated the bytes making up a value, without doing a proper
/// [`Clone`][clone]. You need the value's destructor to run only once,
/// because a double `free` is undefined behavior.
///
/// An example is a possible implementation of [`mem::swap`][swap]:
///
/// ```
/// use std::mem;
/// use std::ptr;
///
/// # #[allow(dead_code)]
/// fn swap<T>(x: &mut T, y: &mut T) {
///     unsafe {
///         // Give ourselves some scratch space to work with
///         let mut t: T = mem::uninitialized();
///
///         // Perform the swap, `&mut` pointers never alias
///         ptr::copy_nonoverlapping(&*x, &mut t, 1);
///         ptr::copy_nonoverlapping(&*y, x, 1);
///         ptr::copy_nonoverlapping(&t, y, 1);
///
///         // y and t now point to the same thing, but we need to completely
///         // forget `t` because we do not want to run the destructor for `T`
///         // on its value, which is still owned somewhere outside this function.
///         mem::forget(t);
///     }
/// }
/// ```
///
/// [drop]: fn.drop.html
/// [uninit]: fn.uninitialized.html
/// [clone]: ../clone/trait.Clone.html
/// [swap]: fn.swap.html
/// [box]: ../../std/boxed/struct.Box.html
/// [leak]: ../../std/boxed/struct.Box.html#method.leak
/// [into_raw]: ../../std/boxed/struct.Box.html#method.into_raw
/// [ub]: ../../reference/behavior-considered-undefined.html
#[inline]
#[stable(feature = "rust1", since = "1.0.0")]
pub fn forget<T>(t: T) {
    ManuallyDrop::new(t);
}

/// Like [`forget`], but also accepts unsized values.
///
/// This function is just a shim intended to be removed when the `unsized_locals` feature gets
/// stabilized.
///
/// [`forget`]: fn.forget.html
#[inline]
#[unstable(feature = "forget_unsized", issue = "0")]
pub fn forget_unsized<T: ?Sized>(t: T) {
    unsafe { intrinsics::forget(t) }
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
/// [alignment]: ./fn.align_of.html
#[inline]
#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_promotable]
pub const fn size_of<T>() -> usize {
    intrinsics::size_of::<T>()
}

/// Returns the size of the pointed-to value in bytes.
///
/// This is usually the same as `size_of::<T>()`. However, when `T` *has* no
/// statically-known size, e.g., a slice [`[T]`][slice] or a [trait object],
/// then `size_of_val` can be used to get the dynamically-known size.
///
/// [slice]: ../../std/primitive.slice.html
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
#[stable(feature = "rust1", since = "1.0.0")]
pub fn size_of_val<T: ?Sized>(val: &T) -> usize {
    unsafe { intrinsics::size_of_val(val) }
}

/// Returns the [ABI]-required minimum alignment of a type.
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
#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_deprecated(reason = "use `align_of` instead", since = "1.2.0")]
pub fn min_align_of<T>() -> usize {
    intrinsics::min_align_of::<T>()
}

/// Returns the [ABI]-required minimum alignment of the type of the value that `val` points to.
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
#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_deprecated(reason = "use `align_of_val` instead", since = "1.2.0")]
pub fn min_align_of_val<T: ?Sized>(val: &T) -> usize {
    unsafe { intrinsics::min_align_of_val(val) }
}

/// Returns the [ABI]-required minimum alignment of a type.
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
#[inline]
#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_promotable]
pub const fn align_of<T>() -> usize {
    intrinsics::min_align_of::<T>()
}

/// Returns the [ABI]-required minimum alignment of the type of the value that `val` points to.
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
#[stable(feature = "rust1", since = "1.0.0")]
pub fn align_of_val<T: ?Sized>(val: &T) -> usize {
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
/// Note that `ptr::drop_in_place` already performs this check, so if your workload
/// can be reduced to some small number of drop_in_place calls, using this is
/// unnecessary. In particular note that you can drop_in_place a slice, and that
/// will do a single needs_drop check for all the values.
///
/// Types like Vec therefore just `drop_in_place(&mut self[..])` without using
/// needs_drop explicitly. Types like HashMap, on the other hand, have to drop
/// values one at a time and should use this API.
///
///
/// # Examples
///
/// Here's an example of how a collection might make use of needs_drop:
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
#[stable(feature = "needs_drop", since = "1.21.0")]
#[rustc_const_unstable(feature = "const_needs_drop")]
pub const fn needs_drop<T>() -> bool {
    intrinsics::needs_drop::<T>()
}

/// Creates a value whose bytes are all zero.
///
/// This has the same effect as allocating space with
/// [`mem::uninitialized`][uninit] and then zeroing it out. It is useful for
/// FFI sometimes, but should generally be avoided.
///
/// There is no guarantee that an all-zero byte-pattern represents a valid value of
/// some type `T`. If `T` has a destructor and the value is destroyed (due to
/// a panic or the end of a scope) before being initialized, then the destructor
/// will run on zeroed data, likely leading to [undefined behavior][ub].
///
/// See also the documentation for [`mem::uninitialized`][uninit], which has
/// many of the same caveats.
///
/// [uninit]: fn.uninitialized.html
/// [ub]: ../../reference/behavior-considered-undefined.html
///
/// # Examples
///
/// ```
/// use std::mem;
///
/// let x: i32 = unsafe { mem::zeroed() };
/// assert_eq!(0, x);
/// ```
#[inline]
#[stable(feature = "rust1", since = "1.0.0")]
pub unsafe fn zeroed<T>() -> T {
    intrinsics::panic_if_uninhabited::<T>();
    intrinsics::init()
}

/// Bypasses Rust's normal memory-initialization checks by pretending to
/// produce a value of type `T`, while doing nothing at all.
///
/// **This is incredibly dangerous and should not be done lightly. Deeply
/// consider initializing your memory with a default value instead.**
///
/// This is useful for FFI functions and initializing arrays sometimes,
/// but should generally be avoided.
///
/// # Undefined behavior
///
/// It is [undefined behavior][ub] to read uninitialized memory, even just an
/// uninitialized boolean. For instance, if you branch on the value of such
/// a boolean, your program may take one, both, or neither of the branches.
///
/// Writing to the uninitialized value is similarly dangerous. Rust believes the
/// value is initialized, and will therefore try to [`Drop`] the uninitialized
/// value and its fields if you try to overwrite it in a normal manner. The only way
/// to safely initialize an uninitialized value is with [`ptr::write`][write],
/// [`ptr::copy`][copy], or [`ptr::copy_nonoverlapping`][copy_no].
///
/// If the value does implement [`Drop`], it must be initialized before
/// it goes out of scope (and therefore would be dropped). Note that this
/// includes a `panic` occurring and unwinding the stack suddenly.
///
/// If you partially initialize an array, you may need to use
/// [`ptr::drop_in_place`][drop_in_place] to remove the elements you have fully
/// initialized followed by [`mem::forget`][mem_forget] to prevent drop running
/// on the array. If a partially allocated array is dropped this will lead to
/// undefined behaviour.
///
/// # Examples
///
/// Here's how to safely initialize an array of [`Vec`]s.
///
/// ```
/// use std::mem;
/// use std::ptr;
///
/// // Only declare the array. This safely leaves it
/// // uninitialized in a way that Rust will track for us.
/// // However we can't initialize it element-by-element
/// // safely, and we can't use the `[value; 1000]`
/// // constructor because it only works with `Copy` data.
/// let mut data: [Vec<u32>; 1000];
///
/// unsafe {
///     // So we need to do this to initialize it.
///     data = mem::uninitialized();
///
///     // DANGER ZONE: if anything panics or otherwise
///     // incorrectly reads the array here, we will have
///     // Undefined Behavior.
///
///     // It's ok to mutably iterate the data, since this
///     // doesn't involve reading it at all.
///     // (ptr and len are statically known for arrays)
///     for elem in &mut data[..] {
///         // *elem = Vec::new() would try to drop the
///         // uninitialized memory at `elem` -- bad!
///         //
///         // Vec::new doesn't allocate or do really
///         // anything. It's only safe to call here
///         // because we know it won't panic.
///         ptr::write(elem, Vec::new());
///     }
///
///     // SAFE ZONE: everything is initialized.
/// }
///
/// println!("{:?}", &data[0]);
/// ```
///
/// This example emphasizes exactly how delicate and dangerous using `mem::uninitialized`
/// can be. Note that the [`vec!`] macro *does* let you initialize every element with a
/// value that is only [`Clone`], so the following is semantically equivalent and
/// vastly less dangerous, as long as you can live with an extra heap
/// allocation:
///
/// ```
/// let data: Vec<Vec<u32>> = vec![Vec::new(); 1000];
/// println!("{:?}", &data[0]);
/// ```
///
/// This example shows how to handle partially initialized arrays, which could
/// be found in low-level datastructures.
///
/// ```
/// use std::mem;
/// use std::ptr;
///
/// // Count the number of elements we have assigned.
/// let mut data_len: usize = 0;
/// let mut data: [String; 1000];
///
/// unsafe {
///     data = mem::uninitialized();
///
///     for elem in &mut data[0..500] {
///         ptr::write(elem, String::from("hello"));
///         data_len += 1;
///     }
///
///     // For each item in the array, drop if we allocated it.
///     for i in &mut data[0..data_len] {
///         ptr::drop_in_place(i);
///     }
/// }
/// // Forget the data. If this is allowed to drop, you may see a crash such as:
/// // 'mem_uninit_test(2457,0x7fffb55dd380) malloc: *** error for object
/// // 0x7ff3b8402920: pointer being freed was not allocated'
/// mem::forget(data);
/// ```
///
/// [`Vec`]: ../../std/vec/struct.Vec.html
/// [`vec!`]: ../../std/macro.vec.html
/// [`Clone`]: ../../std/clone/trait.Clone.html
/// [ub]: ../../reference/behavior-considered-undefined.html
/// [write]: ../ptr/fn.write.html
/// [drop_in_place]: ../ptr/fn.drop_in_place.html
/// [mem_zeroed]: fn.zeroed.html
/// [mem_forget]: fn.forget.html
/// [copy]: ../intrinsics/fn.copy.html
/// [copy_no]: ../intrinsics/fn.copy_nonoverlapping.html
/// [`Drop`]: ../ops/trait.Drop.html
#[inline]
#[rustc_deprecated(since = "2.0.0", reason = "use `mem::MaybeUninit::uninitialized` instead")]
#[stable(feature = "rust1", since = "1.0.0")]
pub unsafe fn uninitialized<T>() -> T {
    intrinsics::panic_if_uninhabited::<T>();
    intrinsics::uninit()
}

/// Swaps the values at two mutable locations, without deinitializing either one.
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
pub fn swap<T>(x: &mut T, y: &mut T) {
    unsafe {
        ptr::swap_nonoverlapping_one(x, y);
    }
}

/// Moves `src` into the referenced `dest`, returning the previous `dest` value.
///
/// Neither value is dropped.
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
/// assert_eq!(2, old_v.len());
/// assert_eq!(3, v.len());
/// ```
///
/// `replace` allows consumption of a struct field by replacing it with another value.
/// Without `replace` you can run into issues like these:
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
/// `self.buf`. But `replace` can be used to disassociate the original value of `self.buf` from
/// `self`, allowing it to be returned:
///
/// ```
/// # #![allow(dead_code)]
/// use std::mem;
///
/// # struct Buffer<T> { buf: Vec<T> }
/// impl<T> Buffer<T> {
///     fn get_and_reset(&mut self) -> Vec<T> {
///         mem::replace(&mut self.buf, Vec::new())
///     }
/// }
/// ```
///
/// [`Clone`]: ../../std/clone/trait.Clone.html
#[inline]
#[stable(feature = "rust1", since = "1.0.0")]
pub fn replace<T>(dest: &mut T, mut src: T) -> T {
    swap(dest, &mut src);
    src
}

/// Disposes of a value.
///
/// This does call the argument's implementation of [`Drop`][drop].
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
/// [drop]: ../ops/trait.Drop.html
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
/// [`RefCell`]: ../../std/cell/struct.RefCell.html
/// [`Copy`]: ../../std/marker/trait.Copy.html
#[inline]
#[stable(feature = "rust1", since = "1.0.0")]
pub fn drop<T>(_x: T) { }

/// Interprets `src` as having type `&U`, and then reads `src` without moving
/// the contained value.
///
/// This function will unsafely assume the pointer `src` is valid for
/// [`size_of::<U>`][size_of] bytes by transmuting `&T` to `&U` and then reading
/// the `&U`. It will also unsafely create a copy of the contained value instead of
/// moving out of `src`.
///
/// It is not a compile-time error if `T` and `U` have different sizes, but it
/// is highly encouraged to only invoke this function where `T` and `U` have the
/// same size. This function triggers [undefined behavior][ub] if `U` is larger than
/// `T`.
///
/// [ub]: ../../reference/behavior-considered-undefined.html
/// [size_of]: fn.size_of.html
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
/// let foo_slice = [10u8];
///
/// unsafe {
///     // Copy the data from 'foo_slice' and treat it as a 'Foo'
///     let mut foo_struct: Foo = mem::transmute_copy(&foo_slice);
///     assert_eq!(foo_struct.bar, 10);
///
///     // Modify the copied data
///     foo_struct.bar = 20;
///     assert_eq!(foo_struct.bar, 20);
/// }
///
/// // The contents of 'foo_slice' should not have changed
/// assert_eq!(foo_slice, [10]);
/// ```
#[inline]
#[stable(feature = "rust1", since = "1.0.0")]
pub unsafe fn transmute_copy<T, U>(src: &T) -> U {
    ptr::read_unaligned(src as *const T as *const U)
}

/// Opaque type representing the discriminant of an enum.
///
/// See the [`discriminant`] function in this module for more information.
///
/// [`discriminant`]: fn.discriminant.html
#[stable(feature = "discriminant_value", since = "1.21.0")]
pub struct Discriminant<T>(u64, PhantomData<fn() -> T>);

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
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        fmt.debug_tuple("Discriminant")
           .field(&self.0)
           .finish()
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
/// of some variant will not change between compilations with the same compiler.
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
/// assert!(mem::discriminant(&Foo::A("bar")) == mem::discriminant(&Foo::A("baz")));
/// assert!(mem::discriminant(&Foo::B(1))     == mem::discriminant(&Foo::B(2)));
/// assert!(mem::discriminant(&Foo::B(3))     != mem::discriminant(&Foo::C(3)));
/// ```
#[stable(feature = "discriminant_value", since = "1.21.0")]
pub fn discriminant<T>(v: &T) -> Discriminant<T> {
    unsafe {
        Discriminant(intrinsics::discriminant_value(v), PhantomData)
    }
}

/// A wrapper to inhibit compiler from automatically calling `T`’s destructor.
///
/// This wrapper is 0-cost.
///
/// # Examples
///
/// This wrapper helps with explicitly documenting the drop order dependencies between fields of
/// the type:
///
/// ```rust
/// use std::mem::ManuallyDrop;
/// struct Peach;
/// struct Banana;
/// struct Melon;
/// struct FruitBox {
///     // Immediately clear there’s something non-trivial going on with these fields.
///     peach: ManuallyDrop<Peach>,
///     melon: Melon, // Field that’s independent of the other two.
///     banana: ManuallyDrop<Banana>,
/// }
///
/// impl Drop for FruitBox {
///     fn drop(&mut self) {
///         unsafe {
///             // Explicit ordering in which field destructors are run specified in the intuitive
///             // location – the destructor of the structure containing the fields.
///             // Moreover, one can now reorder fields within the struct however much they want.
///             ManuallyDrop::drop(&mut self.peach);
///             ManuallyDrop::drop(&mut self.banana);
///         }
///         // After destructor for `FruitBox` runs (this function), the destructor for Melon gets
///         // invoked in the usual manner, as it is not wrapped in `ManuallyDrop`.
///     }
/// }
/// ```
#[stable(feature = "manually_drop", since = "1.20.0")]
#[lang = "manually_drop"]
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct ManuallyDrop<T: ?Sized> {
    value: T,
}

impl<T> ManuallyDrop<T> {
    /// Wrap a value to be manually dropped.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use std::mem::ManuallyDrop;
    /// ManuallyDrop::new(Box::new(()));
    /// ```
    #[stable(feature = "manually_drop", since = "1.20.0")]
    #[inline(always)]
    pub const fn new(value: T) -> ManuallyDrop<T> {
        ManuallyDrop { value }
    }

    /// Extracts the value from the `ManuallyDrop` container.
    ///
    /// This allows the value to be dropped again.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use std::mem::ManuallyDrop;
    /// let x = ManuallyDrop::new(Box::new(()));
    /// let _: Box<()> = ManuallyDrop::into_inner(x); // This drops the `Box`.
    /// ```
    #[stable(feature = "manually_drop", since = "1.20.0")]
    #[inline(always)]
    pub const fn into_inner(slot: ManuallyDrop<T>) -> T {
        slot.value
    }

    /// Takes the contained value out.
    ///
    /// This method is primarily intended for moving out values in drop.
    /// Instead of using [`ManuallyDrop::drop`] to manually drop the value,
    /// you can use this method to take the value and use it however desired.
    /// `Drop` will be invoked on the returned value following normal end-of-scope rules.
    ///
    /// If you have ownership of the container, you can use [`ManuallyDrop::into_inner`] instead.
    ///
    /// # Safety
    ///
    /// This function semantically moves out the contained value without preventing further usage.
    /// It is up to the user of this method to ensure that this container is not used again.
    ///
    /// [`ManuallyDrop::drop`]: #method.drop
    /// [`ManuallyDrop::into_inner`]: #method.into_inner
    #[must_use = "if you don't need the value, you can use `ManuallyDrop::drop` instead"]
    #[unstable(feature = "manually_drop_take", issue = "55422")]
    #[inline]
    pub unsafe fn take(slot: &mut ManuallyDrop<T>) -> T {
        ManuallyDrop::into_inner(ptr::read(slot))
    }
}

impl<T: ?Sized> ManuallyDrop<T> {
    /// Manually drops the contained value.
    ///
    /// If you have ownership of the value, you can use [`ManuallyDrop::into_inner`] instead.
    ///
    /// # Safety
    ///
    /// This function runs the destructor of the contained value and thus the wrapped value
    /// now represents uninitialized data. It is up to the user of this method to ensure the
    /// uninitialized data is not actually used.
    ///
    /// [`ManuallyDrop::into_inner`]: #method.into_inner
    #[stable(feature = "manually_drop", since = "1.20.0")]
    #[inline]
    pub unsafe fn drop(slot: &mut ManuallyDrop<T>) {
        ptr::drop_in_place(&mut slot.value)
    }
}

#[stable(feature = "manually_drop", since = "1.20.0")]
impl<T: ?Sized> Deref for ManuallyDrop<T> {
    type Target = T;
    #[inline(always)]
    fn deref(&self) -> &T {
        &self.value
    }
}

#[stable(feature = "manually_drop", since = "1.20.0")]
impl<T: ?Sized> DerefMut for ManuallyDrop<T> {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut T {
        &mut self.value
    }
}

/// A newtype to construct uninitialized instances of `T`.
///
/// The compiler, in general, assumes that variables are properly initialized
/// at their respective type. For example, a variable of reference type must
/// be aligned and non-NULL. This is an invariant that must *always* be upheld,
/// even in unsafe code. As a consequence, zero-initializing a variable of reference
/// type causes instantaneous undefined behavior, no matter whether that reference
/// ever gets used to access memory:
///
/// ```rust,no_run
/// use std::mem;
///
/// let x: &i32 = unsafe { mem::zeroed() }; // undefined behavior!
/// ```
///
/// This is exploited by the compiler for various optimizations, such as eliding
/// run-time checks and optimizing `enum` layout.
///
/// Not initializing memory at all (instead of zero--initializing it) causes the same
/// issue: after all, the initial value of the variable might just happen to be
/// one that violates the invariant.
///
/// `MaybeUninit` serves to enable unsafe code to deal with uninitialized data:
/// it is a signal to the compiler indicating that the data here might *not*
/// be initialized:
///
/// ```rust
/// #![feature(maybe_uninit)]
/// use std::mem::MaybeUninit;
///
/// // Create an explicitly uninitialized reference.
/// let mut x = MaybeUninit::<&i32>::uninitialized();
/// // Set it to a valid value.
/// x.set(&0);
/// // Extract the initialized data -- this is only allowed *after* properly
/// // initializing `x`!
/// let x = unsafe { x.into_initialized() };
/// ```
///
/// The compiler then knows to not optimize this code.
#[allow(missing_debug_implementations)]
#[unstable(feature = "maybe_uninit", issue = "53491")]
// NOTE after stabilizing `MaybeUninit` proceed to deprecate `mem::{uninitialized,zeroed}`
pub union MaybeUninit<T> {
    uninit: (),
    value: ManuallyDrop<T>,
}

impl<T> MaybeUninit<T> {
    /// Create a new `MaybeUninit` initialized with the given value.
    ///
    /// Note that dropping a `MaybeUninit` will never call `T`'s drop code.
    /// It is your responsibility to make sure `T` gets dropped if it got initialized.
    #[unstable(feature = "maybe_uninit", issue = "53491")]
    #[inline(always)]
    pub const fn new(val: T) -> MaybeUninit<T> {
        MaybeUninit { value: ManuallyDrop::new(val) }
    }

    /// Creates a new `MaybeUninit` in an uninitialized state.
    ///
    /// Note that dropping a `MaybeUninit` will never call `T`'s drop code.
    /// It is your responsibility to make sure `T` gets dropped if it got initialized.
    #[unstable(feature = "maybe_uninit", issue = "53491")]
    #[inline(always)]
    pub const fn uninitialized() -> MaybeUninit<T> {
        MaybeUninit { uninit: () }
    }

    /// Creates a new `MaybeUninit` in an uninitialized state, with the memory being
    /// filled with `0` bytes. It depends on `T` whether that already makes for
    /// proper initialization. For example, `MaybeUninit<usize>::zeroed()` is initialized,
    /// but `MaybeUninit<&'static i32>::zeroed()` is not because references must not
    /// be null.
    ///
    /// Note that dropping a `MaybeUninit` will never call `T`'s drop code.
    /// It is your responsibility to make sure `T` gets dropped if it got initialized.
    #[unstable(feature = "maybe_uninit", issue = "53491")]
    #[inline]
    pub fn zeroed() -> MaybeUninit<T> {
        let mut u = MaybeUninit::<T>::uninitialized();
        unsafe {
            u.as_mut_ptr().write_bytes(0u8, 1);
        }
        u
    }

    /// Sets the value of the `MaybeUninit`. This overwrites any previous value without dropping it.
    /// For your convenience, this also returns a mutable reference to the (now safely initialized)
    /// contents of `self`.
    #[unstable(feature = "maybe_uninit", issue = "53491")]
    #[inline(always)]
    pub fn set(&mut self, val: T) -> &mut T {
        unsafe {
            self.value = ManuallyDrop::new(val);
            self.get_mut()
        }
    }

    /// Extracts the value from the `MaybeUninit` container. This is a great way
    /// to ensure that the data will get dropped, because the resulting `T` is
    /// subject to the usual drop handling.
    ///
    /// # Unsafety
    ///
    /// It is up to the caller to guarantee that the `MaybeUninit` really is in an initialized
    /// state, otherwise this will immediately cause undefined behavior.
    #[unstable(feature = "maybe_uninit", issue = "53491")]
    #[inline(always)]
    pub unsafe fn into_initialized(self) -> T {
        intrinsics::panic_if_uninhabited::<T>();
        ManuallyDrop::into_inner(self.value)
    }

    /// Deprecated alternative to `into_initialized`. Will never get stabilized.
    /// Exists only to transition stdsimd to `into_initialized`.
    #[inline(always)]
    #[allow(unused)]
    pub(crate) unsafe fn into_inner(self) -> T {
        self.into_initialized()
    }

    /// Gets a reference to the contained value.
    ///
    /// # Unsafety
    ///
    /// It is up to the caller to guarantee that the `MaybeUninit` really is in an initialized
    /// state, otherwise this will immediately cause undefined behavior.
    #[unstable(feature = "maybe_uninit", issue = "53491")]
    #[inline(always)]
    pub unsafe fn get_ref(&self) -> &T {
        &*self.value
    }

    /// Gets a mutable reference to the contained value.
    ///
    /// # Unsafety
    ///
    /// It is up to the caller to guarantee that the `MaybeUninit` really is in an initialized
    /// state, otherwise this will immediately cause undefined behavior.
    // FIXME(#53491): We currently rely on the above being incorrect, i.e., we have references
    // to uninitialized data (e.g., in `libcore/fmt/float.rs`).  We should make
    // a final decision about the rules before stabilization.
    #[unstable(feature = "maybe_uninit", issue = "53491")]
    #[inline(always)]
    pub unsafe fn get_mut(&mut self) -> &mut T {
        &mut *self.value
    }

    /// Gets a pointer to the contained value. Reading from this pointer or turning it
    /// into a reference will be undefined behavior unless the `MaybeUninit` is initialized.
    #[unstable(feature = "maybe_uninit", issue = "53491")]
    #[inline(always)]
    pub fn as_ptr(&self) -> *const T {
        unsafe { &*self.value as *const T }
    }

    /// Get sa mutable pointer to the contained value. Reading from this pointer or turning it
    /// into a reference will be undefined behavior unless the `MaybeUninit` is initialized.
    #[unstable(feature = "maybe_uninit", issue = "53491")]
    #[inline(always)]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        unsafe { &mut *self.value as *mut T }
    }

    /// Gets a pointer to the first element of the array.
    #[unstable(feature = "maybe_uninit", issue = "53491")]
    #[inline(always)]
    pub fn first_ptr(this: &[MaybeUninit<T>]) -> *const T {
        this as *const [MaybeUninit<T>] as *const T
    }

    /// Gets a mutable pointer to the first element of the array.
    #[unstable(feature = "maybe_uninit", issue = "53491")]
    #[inline(always)]
    pub fn first_ptr_mut(this: &mut [MaybeUninit<T>]) -> *mut T {
        this as *mut [MaybeUninit<T>] as *mut T
    }
}
