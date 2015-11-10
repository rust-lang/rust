// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Basic functions for dealing with memory
//!
//! This module contains functions for querying the size and alignment of
//! types, initializing and manipulating memory.

#![stable(feature = "rust1", since = "1.0.0")]

use marker::Sized;
use intrinsics;
use ptr;

#[stable(feature = "rust1", since = "1.0.0")]
pub use intrinsics::transmute;

/// Leaks a value into the void, consuming ownership and never running its
/// destructor.
///
/// This function will take ownership of its argument, but is distinct from the
/// `mem::drop` function in that it **does not run the destructor**, leaking the
/// value and any resources that it owns.
///
/// There's only a few reasons to use this function. They mainly come
/// up in unsafe code or FFI code.
///
/// * You have an uninitialized value, perhaps for performance reasons, and
///   need to prevent the destructor from running on it.
/// * You have two copies of a value (like when writing something like
///   [`mem::swap`][swap]), but need the destructor to only run once to
///   prevent a double `free`.
/// * Transferring resources across [FFI][ffi] boundaries.
///
/// [swap]: fn.swap.html
/// [ffi]: ../../book/ffi.html
///
/// # Safety
///
/// This function is not marked as `unsafe` as Rust does not guarantee that the
/// `Drop` implementation for a value will always run. Note, however, that
/// leaking resources such as memory or I/O objects is likely not desired, so
/// this function is only recommended for specialized use cases.
///
/// The safety of this function implies that when writing `unsafe` code
/// yourself care must be taken when leveraging a destructor that is required to
/// run to preserve memory safety. There are known situations where the
/// destructor may not run (such as if ownership of the object with the
/// destructor is returned) which must be taken into account.
///
/// # Other forms of Leakage
///
/// It's important to point out that this function is not the only method by
/// which a value can be leaked in safe Rust code. Other known sources of
/// leakage are:
///
/// * `Rc` and `Arc` cycles
/// * `mpsc::{Sender, Receiver}` cycles (they use `Arc` internally)
/// * Panicking destructors are likely to leak local resources
///
/// # Example
///
/// Leak some heap memory by never deallocating it:
///
/// ```rust
/// use std::mem;
///
/// let heap_memory = Box::new(3);
/// mem::forget(heap_memory);
/// ```
///
/// Leak an I/O object, never closing the file:
///
/// ```rust,no_run
/// use std::mem;
/// use std::fs::File;
///
/// let file = File::open("foo.txt").unwrap();
/// mem::forget(file);
/// ```
///
/// The `mem::swap` function uses `mem::forget` to good effect:
///
/// ```rust
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
#[stable(feature = "rust1", since = "1.0.0")]
pub fn forget<T>(t: T) {
    unsafe { intrinsics::forget(t) }
}

/// Returns the size of a type in bytes.
///
/// # Examples
///
/// ```
/// use std::mem;
///
/// assert_eq!(4, mem::size_of::<i32>());
/// ```
#[inline]
#[stable(feature = "rust1", since = "1.0.0")]
pub fn size_of<T>() -> usize {
    unsafe { intrinsics::size_of::<T>() }
}

/// Returns the size of the type that `val` points to in bytes.
///
/// # Examples
///
/// ```
/// use std::mem;
///
/// assert_eq!(4, mem::size_of_val(&5i32));
/// ```
#[inline]
#[stable(feature = "rust1", since = "1.0.0")]
pub fn size_of_val<T: ?Sized>(val: &T) -> usize {
    unsafe { intrinsics::size_of_val(val) }
}

/// Returns the ABI-required minimum alignment of a type
///
/// This is the alignment used for struct fields. It may be smaller than the preferred alignment.
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
#[deprecated(reason = "use `align_of` instead", since = "1.2.0")]
pub fn min_align_of<T>() -> usize {
    unsafe { intrinsics::min_align_of::<T>() }
}

/// Returns the ABI-required minimum alignment of the type of the value that `val` points to
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
#[deprecated(reason = "use `align_of_val` instead", since = "1.2.0")]
pub fn min_align_of_val<T: ?Sized>(val: &T) -> usize {
    unsafe { intrinsics::min_align_of_val(val) }
}

/// Returns the alignment in memory for a type.
///
/// This is the alignment used for struct fields. It may be smaller than the preferred alignment.
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
pub fn align_of<T>() -> usize {
    unsafe { intrinsics::min_align_of::<T>() }
}

/// Returns the ABI-required minimum alignment of the type of the value that `val` points to
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

/// Creates a value initialized to zero.
///
/// This function is similar to allocating space for a local variable and zeroing it out (an unsafe
/// operation).
///
/// Care must be taken when using this function, if the type `T` has a destructor and the value
/// falls out of scope (due to unwinding or returning) before being initialized, then the
/// destructor will run on zeroed data, likely leading to crashes.
///
/// This is useful for FFI functions sometimes, but should generally be avoided.
///
/// # Examples
///
/// ```
/// use std::mem;
///
/// let x: i32 = unsafe { mem::zeroed() };
/// ```
#[inline]
#[stable(feature = "rust1", since = "1.0.0")]
pub unsafe fn zeroed<T>() -> T {
    intrinsics::init()
}

/// Creates a value initialized to an unspecified series of bytes.
///
/// The byte sequence usually indicates that the value at the memory
/// in question has been dropped. Thus, *if* T carries a drop flag,
/// any associated destructor will not be run when the value falls out
/// of scope.
///
/// Some code at one time used the `zeroed` function above to
/// accomplish this goal.
///
/// This function is expected to be deprecated with the transition
/// to non-zeroing drop.
#[inline]
#[unstable(feature = "filling_drop", issue = "5016")]
pub unsafe fn dropped<T>() -> T {
    #[inline(always)]
    unsafe fn dropped_impl<T>() -> T { intrinsics::init_dropped() }

    dropped_impl()
}

/// Bypasses Rust's normal memory-initialization checks by pretending to
/// produce a value of type T, while doing nothing at all.
///
/// **This is incredibly dangerous, and should not be done lightly. Deeply
/// consider initializing your memory with a default value instead.**
///
/// This is useful for FFI functions and initializing arrays sometimes,
/// but should generally be avoided.
///
/// # Undefined Behavior
///
/// It is Undefined Behavior to read uninitialized memory. Even just an
/// uninitialized boolean. For instance, if you branch on the value of such
/// a boolean your program may take one, both, or neither of the branches.
///
/// Note that this often also includes *writing* to the uninitialized value.
/// Rust believes the value is initialized, and will therefore try to Drop
/// the uninitialized value and its fields if you try to overwrite the memory
/// in a normal manner. The only way to safely initialize an arbitrary
/// uninitialized value is with one of the `ptr` functions: `write`, `copy`, or
/// `copy_nonoverlapping`. This isn't necessary if `T` is a primitive
/// or otherwise only contains types that don't implement Drop.
///
/// If this value *does* need some kind of Drop, it must be initialized before
/// it goes out of scope (and therefore would be dropped). Note that this
/// includes a `panic` occurring and unwinding the stack suddenly.
///
/// # Examples
///
/// Here's how to safely initialize an array of `Vec`s.
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
/// This example emphasizes exactly how delicate and dangerous doing this is.
/// Note that the `vec!` macro *does* let you initialize every element with a
/// value that is only `Clone`, so the following is semantically equivalent and
/// vastly less dangerous, as long as you can live with an extra heap
/// allocation:
///
/// ```
/// let data: Vec<Vec<u32>> = vec![Vec::new(); 1000];
/// println!("{:?}", &data[0]);
/// ```
#[inline]
#[stable(feature = "rust1", since = "1.0.0")]
pub unsafe fn uninitialized<T>() -> T {
    intrinsics::uninit()
}

/// Swap the values at two mutable locations of the same type, without deinitializing or copying
/// either one.
///
/// # Examples
///
/// ```
/// use std::mem;
///
/// let x = &mut 5;
/// let y = &mut 42;
///
/// mem::swap(x, y);
///
/// assert_eq!(42, *x);
/// assert_eq!(5, *y);
/// ```
#[inline]
#[stable(feature = "rust1", since = "1.0.0")]
pub fn swap<T>(x: &mut T, y: &mut T) {
    unsafe {
        // Give ourselves some scratch space to work with
        let mut t: T = uninitialized();

        // Perform the swap, `&mut` pointers never alias
        ptr::copy_nonoverlapping(&*x, &mut t, 1);
        ptr::copy_nonoverlapping(&*y, x, 1);
        ptr::copy_nonoverlapping(&t, y, 1);

        // y and t now point to the same thing, but we need to completely
        // forget `t` because we do not want to run the destructor for `T`
        // on its value, which is still owned somewhere outside this function.
        forget(t);
    }
}

/// Replaces the value at a mutable location with a new one, returning the old value, without
/// deinitializing or copying either one.
///
/// This is primarily used for transferring and swapping ownership of a value in a mutable
/// location.
///
/// # Examples
///
/// A simple example:
///
/// ```
/// use std::mem;
///
/// let mut v: Vec<i32> = Vec::new();
///
/// mem::replace(&mut v, Vec::new());
/// ```
///
/// This function allows consumption of one field of a struct by replacing it with another value.
/// The normal approach doesn't always work:
///
/// ```rust,ignore
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
/// Note that `T` does not necessarily implement `Clone`, so it can't even clone and reset
/// `self.buf`. But `replace` can be used to disassociate the original value of `self.buf` from
/// `self`, allowing it to be returned:
///
/// ```
/// # #![allow(dead_code)]
/// use std::mem;
/// # struct Buffer<T> { buf: Vec<T> }
/// impl<T> Buffer<T> {
///     fn get_and_reset(&mut self) -> Vec<T> {
///         mem::replace(&mut self.buf, Vec::new())
///     }
/// }
/// ```
#[inline]
#[stable(feature = "rust1", since = "1.0.0")]
pub fn replace<T>(dest: &mut T, mut src: T) -> T {
    swap(dest, &mut src);
    src
}

/// Disposes of a value.
///
/// While this does call the argument's implementation of `Drop`, it will not
/// release any borrows, as borrows are based on lexical scope.
///
/// This effectively does nothing for
/// [types which implement `Copy`](../../book/ownership.html#copy-types),
/// e.g. integers. Such values are copied and _then_ moved into the function,
/// so the value persists after this function call.
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
/// Borrows are based on lexical scope, so this produces an error:
///
/// ```ignore
/// let mut v = vec![1, 2, 3];
/// let x = &v[0];
///
/// drop(x); // explicitly drop the reference, but the borrow still exists
///
/// v.push(4); // error: cannot borrow `v` as mutable because it is also
///            // borrowed as immutable
/// ```
///
/// An inner scope is needed to fix this:
///
/// ```
/// let mut v = vec![1, 2, 3];
///
/// {
///     let x = &v[0];
///
///     drop(x); // this is now redundant, as `x` is going out of scope anyway
/// }
///
/// v.push(4); // no problems
/// ```
///
/// Since `RefCell` enforces the borrow rules at runtime, `drop()` can
/// seemingly release a borrow of one:
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
/// Integers and other types implementing `Copy` are unaffected by `drop()`
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
#[inline]
#[stable(feature = "rust1", since = "1.0.0")]
pub fn drop<T>(_x: T) { }

macro_rules! repeat_u8_as_u32 {
    ($name:expr) => { (($name as u32) << 24 |
                       ($name as u32) << 16 |
                       ($name as u32) <<  8 |
                       ($name as u32)) }
}
macro_rules! repeat_u8_as_u64 {
    ($name:expr) => { ((repeat_u8_as_u32!($name) as u64) << 32 |
                       (repeat_u8_as_u32!($name) as u64)) }
}

// NOTE: Keep synchronized with values used in librustc_trans::trans::adt.
//
// In particular, the POST_DROP_U8 marker must never equal the
// DTOR_NEEDED_U8 marker.
//
// For a while pnkfelix was using 0xc1 here.
// But having the sign bit set is a pain, so 0x1d is probably better.
//
// And of course, 0x00 brings back the old world of zero'ing on drop.
#[unstable(feature = "filling_drop", issue = "5016")]
#[allow(missing_docs)]
pub const POST_DROP_U8: u8 = 0x1d;
#[unstable(feature = "filling_drop", issue = "5016")]
#[allow(missing_docs)]
pub const POST_DROP_U32: u32 = repeat_u8_as_u32!(POST_DROP_U8);
#[unstable(feature = "filling_drop", issue = "5016")]
#[allow(missing_docs)]
pub const POST_DROP_U64: u64 = repeat_u8_as_u64!(POST_DROP_U8);

#[cfg(target_pointer_width = "32")]
#[unstable(feature = "filling_drop", issue = "5016")]
#[allow(missing_docs)]
pub const POST_DROP_USIZE: usize = POST_DROP_U32 as usize;
#[cfg(target_pointer_width = "64")]
#[unstable(feature = "filling_drop", issue = "5016")]
#[allow(missing_docs)]
pub const POST_DROP_USIZE: usize = POST_DROP_U64 as usize;

/// Interprets `src` as `&U`, and then reads `src` without moving the contained
/// value.
///
/// This function will unsafely assume the pointer `src` is valid for
/// `sizeof(U)` bytes by transmuting `&T` to `&U` and then reading the `&U`. It
/// will also unsafely create a copy of the contained value instead of moving
/// out of `src`.
///
/// It is not a compile-time error if `T` and `U` have different sizes, but it
/// is highly encouraged to only invoke this function where `T` and `U` have the
/// same size. This function triggers undefined behavior if `U` is larger than
/// `T`.
///
/// # Examples
///
/// ```
/// use std::mem;
///
/// let one = unsafe { mem::transmute_copy(&1) };
///
/// assert_eq!(1, one);
/// ```
#[inline]
#[stable(feature = "rust1", since = "1.0.0")]
pub unsafe fn transmute_copy<T, U>(src: &T) -> U {
    // FIXME(#23542) Replace with type ascription.
    #![allow(trivial_casts)]
    ptr::read(src as *const T as *const U)
}
